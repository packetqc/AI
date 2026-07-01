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
#include "mcu_cache.h"        /* self-gating M55 D-cache maintenance (no-op while D-cache off) */
#include <string.h>

/* Per-token CPU<->NPU dialog logger (implemented in grammar_runner.cpp via the C++
 * TerminalLogger). One line per autoregressive step = one NPU epoch. */
extern void NPU_LogStep(int rule_idx, int pos, int tok_id, const char *piece);
extern void lvgl_port_n6_display_freeze(int freeze);   /* per-epoch display gate: LTDC fetch off during the NPU run */
extern volatile int g_lvgl_ok;                          /* 1 once the LVGL display is initialised */
extern volatile int g_npu_quiet;                        /* 1 = suppress the per-token UART dialog ("just infer + display" experiment) */
extern volatile int g_npu_gate;                         /* stage 2: 1 = per-epoch LTDC gate ON (old), 0 = gate DROPPED (default) */

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

        /* D-cache coherency (flicker fix — stage 1): the CPU just filled in_buf THROUGH the M55 cache,
         * so clean it back to memory before the NPU reads the embedding over its AXI master — otherwise
         * (once stage-1 enables D-cache) the NPU sees stale input and the answer is wrong. ST's own
         * contract makes cleaning app-filled input buffers the application's responsibility; the runtime
         * only invalidates the output (network.c). Self-gating: mcu_cache_clean_range checks SCB->CCR.DC,
         * so this is a no-op while D-cache is off (today) and a real SCB_CleanDCache_by_Addr once it's on.
         * in_buf holds C*L int8 = 8192 B (a multiple of the 32-B cache line). */
        mcu_cache_clean_range((uint32_t)(uintptr_t)in_buf,
                              (uint32_t)(uintptr_t)in_buf + (uint32_t)(C * L));

        /* Per-epoch display gate — stage 2: now DROPPED by default (g_npu_gate=0). The gate was a
         * stopgap that stopped the LTDC fetch (Layer1 off) for each epoch because, WITHOUT it and with
         * D-cache OFF, the CPU working set + NPU AXI flood together starved the LTDC's PSRAM scanout ->
         * continuous live-scanout corruption (FB + regs intact, underrun flag 0, but fetched pixels
         * wrong). Stage 1 enabled D-cache, pulling the CPU working set off the AXI bus (the reference's
         * low-traffic pattern), so the LTDC should now scan cleanly through the NPU's flood alone with
         * NO gate and NO per-epoch flicker. g_npu_gate is left as a runtime A/B switch: set it to 1 via
         * GDB to restore the old gate on-device (no reflash) and compare scanout cleanliness. */
        if (g_lvgl_ok && g_npu_gate) lvgl_port_n6_display_freeze(1);
        int npu_run_ok = (stai_network_run(net, STAI_MODE_SYNC) == STAI_SUCCESS);
        if (g_lvgl_ok && g_npu_gate) lvgl_port_n6_display_freeze(0);
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
