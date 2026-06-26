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

void NPU_QueryRule(stai_network *net, int8_t *in_buf, const int8_t *out_buf,
                   int rule_idx, char *out, int out_max)
{
    const int L = STAI_NETWORK_IN_1_HEIGHT;    /* sequence length = 32 */
    const int C = STAI_NETWORK_IN_1_CHANNEL;   /* embedding channels = 256 */
    const int V = STAI_NETWORK_OUT_1_CHANNEL;  /* vocab = 374 */

    int16_t cur[STAI_NETWORK_IN_1_HEIGHT];
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

        if (stai_network_run(net, STAI_MODE_SYNC) != STAI_SUCCESS)
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
