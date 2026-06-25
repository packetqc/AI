/**
 * llm_fsbl_cnn.c — FSBL inference wrapper for the NPU-NATIVE Conv1D model.
 *
 * Drop-in replacement for the transformer llm_fsbl.c. Same public API
 * (LLM_FSBL_Init / RunOnce / Generate / Deinit), but matched to the CNN network:
 *   input  : int8[1,256,32]  CHANNEL_FIRST  (256=hidden ch, 32=seq), QLinear(0.02167,-12)
 *   output : int8[1,374,32]  CHANNEL_FIRST  (374=vocab ch, 32=seq),  QLinear(0.1484,-54)
 *
 * The whole body runs on the Neural-ART NPU (analyze: 5/5 pure-HW epochs, 0 SW).
 * CPU does only the int8 embedding lookup (llm_embed_cnn.h) and the argmax.
 * No BOS slot, no attention mask — the TCN is a plain conv stack.
 */
#include "llm_fsbl.h"
#include "llm_tokenizer.h"
#include "llm_embed.h"          /* llm_embed_cnn.h installed as llm_embed.h */

#include <string.h>
#include "stai_network.h"
#include "stai.h"

STAI_NETWORK_CONTEXT_DECLARE(s_net_ctx, STAI_NETWORK_CONTEXT_SIZE);
static stai_ptr s_inp[STAI_NETWORK_IN_NUM];
static stai_ptr s_out[STAI_NETWORK_OUT_NUM];
static int      s_initialised = 0;

/* channel-first index helpers for [C][W] layout (W = LLM_SEQ_LEN = 32) */
#define IDX(c, p)  ((c) * LLM_SEQ_LEN + (p))

int LLM_FSBL_Init(void)
{
    stai_return_code err = stai_network_init(s_net_ctx);
    if (err != STAI_SUCCESS) return -(int)err;
    stai_size n;
    err = stai_network_get_inputs(s_net_ctx, s_inp, &n);
    if (err != STAI_SUCCESS) return -(int)err;
    err = stai_network_get_outputs(s_net_ctx, s_out, &n);
    if (err != STAI_SUCCESS) return -(int)err;
    s_initialised = 1;
    return 0;
}

void LLM_FSBL_Deinit(void)
{
    if (s_initialised) { stai_network_deinit(s_net_ctx); s_initialised = 0; }
}

/* Run one forward pass. Predicts next token at position (n_tok-1) — the last
 * real position, where the conv's right context is padding (matches training). */
int LLM_FSBL_RunOnce(const int32_t *input_ids, const int32_t *attn_mask,
                     int seq_len, int32_t *next_token_id)
{
    (void)attn_mask;
    if (!s_initialised) return -1;

    int8_t *in = (int8_t *)s_inp[0];           /* [256][32] channel-first */
    int cpy = (seq_len < LLM_SEQ_LEN) ? seq_len : LLM_SEQ_LEN;

    /* fill real tokens into columns 0..cpy-1 */
    for (int p = 0; p < cpy; p++) {
        int tok = input_ids[p];
        if (tok < 0 || tok >= LLM_VOCAB_SIZE) tok = LLM_EOS_TOKEN;
        const int8_t *row = llm_embed_table[tok];     /* [256] */
        for (int c = 0; c < LLM_HIDDEN_SIZE; c++) in[IDX(c, p)] = row[c];
    }
    /* pad remaining columns with the EOS embedding (right-side padding) */
    for (int p = cpy; p < LLM_SEQ_LEN; p++) {
        const int8_t *row = llm_embed_table[LLM_EOS_TOKEN];
        for (int c = 0; c < LLM_HIDDEN_SIZE; c++) in[IDX(c, p)] = row[c];
    }

    if (stai_network_run(s_net_ctx, STAI_MODE_SYNC) != STAI_SUCCESS) return -2;

    /* argmax over the 374 vocab channels at the last real position */
    const int8_t *out = (const int8_t *)s_out[0];     /* [374][32] */
    int last = (cpy > 0) ? cpy - 1 : 0;
    int best = 0; int8_t mx = out[IDX(0, last)];
    for (int v = 1; v < LLM_VOCAB_SIZE; v++) {
        int8_t lv = out[IDX(v, last)];
        if (lv > mx) { mx = lv; best = v; }
    }
    *next_token_id = best;
    return 0;
}

/* autoregressive generation — same contract as the transformer wrapper */
int LLM_FSBL_Generate(const char *prompt, char *out_buf, int out_max, int max_new_tokens)
{
    int32_t ids[LLM_SEQ_LEN] = {0};
    int32_t mask[LLM_SEQ_LEN] = {0};
    int n = LLM_Tokenize(prompt, ids, mask, LLM_SEQ_LEN);
    if (n <= 0) return 0;

    int out_pos = 0, made = 0;
    while (made < max_new_tokens) {
        int32_t nt;
        if (LLM_FSBL_RunOnce(ids, mask, n, &nt) != 0) return -1;
        if (nt == LLM_EOS_TOKEN) break;
        const char *s = LLM_TokenStr(nt);
        int l = (int)strlen(s);
        if (out_pos + l < out_max - 1) { memcpy(out_buf + out_pos, s, l); out_pos += l; }
        made++;
        if (n < LLM_SEQ_LEN) { ids[n++] = nt; }
        else { memmove(ids, ids + 1, (LLM_SEQ_LEN - 1) * sizeof(int32_t)); ids[LLM_SEQ_LEN-1] = nt; }
    }
    out_buf[out_pos] = '\0';
    return made;
}
