/**
 * npu_query.c — the grammar-oracle bridge (the N6's "Ollama").
 *
 * Reproduces the host autoregressive recall loop on the device: a rule-name prompt
 * is fed token-by-token; at each step the CPU does the int8 embedding Gather to build
 * the NPU input [C,L], the NPU runs the conv body, the CPU takes argmax over the vocab
 * to pick the next token, and the loop continues until EOS. The generated tokens are
 * detokenized into the rule body string.
 */
#include "npu_query.h"
#include "network_embed.h"    /* g_embed_table[VOCAB*DIM], EMBED_DIM */
#include "network_tokens.h"   /* g_tok_decode, g_rule_prompt, TOK_EOS_ID, ... */
#include <string.h>

/* Per-token CPU<->NPU dialog logger (implemented in grammar_runner.cpp via the C++
 * TerminalLogger). One line per autoregressive step = one NPU epoch. */
extern void NPU_LogStep(int rule_idx, int pos, int tok_id, const char *piece);
extern void lvgl_port_n6_display_freeze(int freeze);   /* per-epoch display gate: LTDC fetch off during the NPU run */
extern volatile int g_lvgl_ok;                          /* 1 once the LVGL display is initialised */
extern volatile int g_npu_quiet;                        /* 1 = suppress the per-token UART dialog ("just infer + display" experiment) */

void NPU_QueryRule(stai_network *net, int8_t *in_buf, const int8_t *out_buf,
                   int rule_idx, char *out, int out_max)
{
    const int L = 32;    /* sequence length  (model fixed; HW header relabels the dims) */
    const int C = 256;   /* embedding channels */
    const int V = 374;   /* vocab */

    int16_t cur[32];
    int plen = g_rule_prompt_len[rule_idx];
    for (int i = 0; i < L; i++)
        cur[i] = (i < plen) ? g_rule_prompt[rule_idx][i] : (int16_t)TOK_EOS_ID;

    out[0] = '\0';
    int olen = 0;

    for (int pos = plen - 1; pos < L - 1; pos++)
    {
        /* int8 embedding Gather -> NPU input [C,L], channel-major:
         * in[c*L + l] = embedding(cur[l])[c] = g_embed_table[cur[l]*EMBED_DIM + c]. */
        for (int l = 0; l < L; l++)
        {
            const int8_t *row = &g_embed_table[(int)cur[l] * EMBED_DIM];
            for (int c = 0; c < C; c++)
                in_buf[c * L + l] = row[c];
        }

        /* Per-epoch display gate REQUIRED. Test proved it: removing it left the LTDC fetching the front
         * buffer during the NPU's AXI flood -> the live scanout corrupts continuously and the panel is
         * blank all the time (ST mechanism: FB + LTDC regs stay intact, ISR underrun flag = 0, but the
         * FETCHED pixels are wrong). The double-buffer fixes render-vs-read; this gate is the ONLY thing
         * that fixes NPU-vs-read, by stopping the LTDC fetch (Layer1 off) for each epoch's ~ms. The
         * per-epoch layer toggle is a brief flicker, but it is the mitigation given the FB must stay in
         * shared PSRAM (no free non-NPU SRAM bank fits 768K). Between epochs the panel refreshes clean. */
        if (g_lvgl_ok) lvgl_port_n6_display_freeze(1);
        int npu_run_ok = (stai_network_run(net, STAI_MODE_SYNC) == STAI_SUCCESS);
        if (g_lvgl_ok) lvgl_port_n6_display_freeze(0);
        if (!npu_run_ok)
        {
            strncpy(out, "[npu run err]", out_max - 1);
            out[out_max - 1] = '\0';
            return;
        }

        /* argmax over the vocab at the current position: logits[v][pos] = out[v*L + pos]. */
        int best = 0;
        int8_t bv = out_buf[0 * L + pos];
        for (int v = 1; v < V; v++)
        {
            int8_t x = out_buf[v * L + pos];
            if (x > bv) { bv = x; best = v; }
        }

        /* line-by-line CPU<->NPU dialog: this step's embedding Gather + NPU epoch + argmax.
         * g_npu_quiet gates it: the log is a BLOCKING colour UART line per token and no display
         * refresh runs during this loop, so logging only stretches the flicker window. Silencing it
         * = "just infer + display" (shorter inference, less per-cycle glitch degradation). */
        if (!g_npu_quiet)
            NPU_LogStep(rule_idx, pos, best, (best == TOK_EOS_ID) ? "<eos>" : g_tok_decode[best]);

        if (best == TOK_EOS_ID)
            break;

        /* append the detokenized piece. */
        const char *piece = g_tok_decode[best];
        int pl = (int)strlen(piece);
        if (olen + pl < out_max - 1)
        {
            memcpy(out + olen, piece, pl);
            olen += pl;
            out[olen] = '\0';
        }

        if (pos + 1 < L)
            cur[pos + 1] = (int16_t)best;
    }
}
