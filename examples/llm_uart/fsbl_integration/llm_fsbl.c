/**
 * llm_fsbl.c
 *
 * FSBL-mode STAI inference wrapper — uses stai_network_* directly.
 * No dependency on app_x-cube-ai.c, aiValidation, or bsp_ai.h.
 *
 * Memory placement:
 *   s_net_ctx        → .bss in AXISRAM2 RAM  (STAI context, small)
 *   s_activation     → .AI_RAM in AXISRAM3   (331 KB activation buffer)
 *   s_states, s_inp/out → .bss in AXISRAM2 RAM
 */

#include "llm_fsbl.h"
#include "llm_tokenizer.h"

#include <string.h>
#include "network.h"
#include "user_init.h"
#include "stai.h"

/* ── Network context in AXISRAM2 RAM ───────────────────────────────────── */

STAI_NETWORK_CONTEXT_DECLARE(s_net_ctx, STAI_NETWORK_CONTEXT_SIZE);

/* ── Activation buffer in AXISRAM3 (added to linker via .AI_RAM section) ── */

#if defined(__ICCARM__)
#define AI_RAM_ATTR  _Pragma("location=\".AI_RAM\"")
#elif defined(__GNUC__) || defined(__CC_ARM)
#define AI_RAM_ATTR  __attribute__((section(".AI_RAM")))
#endif

STAI_ALIGNED(32)
AI_RAM_ATTR
static uint8_t s_activation[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];

/* ── Small state buffer and pointer arrays in AXISRAM2 RAM ─────────────── */

STAI_ALIGNED(32) static uint8_t s_states[4];

static stai_ptr s_act_ptrs[]   = { (stai_ptr)s_activation };
static stai_ptr s_state_ptrs[] = { (stai_ptr)s_states };
static stai_ptr s_inp[STAI_NETWORK_IN_NUM];
static stai_ptr s_out[STAI_NETWORK_OUT_NUM];
static int      s_initialised  = 0;

/* ── Helpers ────────────────────────────────────────────────────────────── */

static int argmax_i8(const int8_t *arr, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[best]) best = i;
    return best;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int LLM_FSBL_Init(void)
{
    stai_return_code err;
    stai_size n;

    /* user_stai_network_init: calls stai_network_init + handles weights */
    err = user_stai_network_init(s_net_ctx);
    if (err != STAI_SUCCESS) return -(int)err;

    err = stai_network_set_activations(s_net_ctx, s_act_ptrs,
                                       STAI_NETWORK_ACTIVATIONS_NUM);
    if (err != STAI_SUCCESS) return -(int)err;

    err = stai_network_set_states(s_net_ctx, s_state_ptrs,
                                  STAI_NETWORK_STATES_NUM);
    if (err != STAI_SUCCESS) return -(int)err;

    /* Obtain pointers to the network's own input/output tensor buffers */
    err = stai_network_get_inputs(s_net_ctx, s_inp, &n);
    if (err != STAI_SUCCESS) return -(int)err;

    err = stai_network_get_outputs(s_net_ctx, s_out, &n);
    if (err != STAI_SUCCESS) return -(int)err;

    s_initialised = 1;
    return 0;
}

void LLM_FSBL_Deinit(void)
{
    if (s_initialised) {
        stai_network_deinit(s_net_ctx);
        s_initialised = 0;
    }
}

int LLM_FSBL_RunOnce(const int32_t *input_ids, const int32_t *attn_mask,
                     int seq_len, int32_t *next_token_id)
{
    if (!s_initialised) return -1;

    /* Fill network's own input tensors (int32, 32 elements each) */
    memcpy((int32_t *)s_inp[0], input_ids, LLM_SEQ_LEN * sizeof(int32_t));
    memcpy((int32_t *)s_inp[1], attn_mask, LLM_SEQ_LEN * sizeof(int32_t));

    stai_return_code err = stai_network_run(s_net_ctx, STAI_MODE_SYNC);
    if (err != STAI_SUCCESS) return -(int)err;

    /* Argmax at last real token position — output int8[1×32×1×374] channel-last */
    const int8_t *logits = (const int8_t *)s_out[0];
    int last = (seq_len > 0 && seq_len <= LLM_SEQ_LEN) ? seq_len - 1 : LLM_SEQ_LEN - 1;
    *next_token_id = (int32_t)argmax_i8(logits + last * LLM_VOCAB_SIZE, LLM_VOCAB_SIZE);

    return 0;
}

int LLM_FSBL_Generate(const char *prompt, char *out_buf, int out_max, int max_new_tokens)
{
    int32_t ids[LLM_SEQ_LEN]  = {0};
    int32_t mask[LLM_SEQ_LEN] = {0};
    int out_pos    = 0;
    int new_tokens = 0;

    int n = LLM_Tokenize(prompt, ids, mask, LLM_SEQ_LEN);
    if (n <= 0) return 0;

    while (new_tokens < max_new_tokens) {
        int32_t next_tok;
        if (LLM_FSBL_RunOnce(ids, mask, n, &next_tok) != 0) return -1;

        if (next_tok == LLM_EOS_TOKEN) break;

        const char *tok_str = LLM_TokenStr(next_tok);
        int tok_len = (int)strlen(tok_str);
        if (out_pos + tok_len < out_max - 1) {
            memcpy(out_buf + out_pos, tok_str, tok_len);
            out_pos += tok_len;
        }
        new_tokens++;

        /* Slide context window */
        if (n < LLM_SEQ_LEN) {
            ids[n]  = next_tok;
            mask[n] = 1;
            n++;
        } else {
            memmove(ids,  ids  + 1, (LLM_SEQ_LEN - 1) * sizeof(int32_t));
            memmove(mask, mask + 1, (LLM_SEQ_LEN - 1) * sizeof(int32_t));
            ids[LLM_SEQ_LEN - 1]  = next_tok;
            mask[LLM_SEQ_LEN - 1] = 1;
        }
    }

    out_buf[out_pos] = '\0';
    return new_tokens;
}
