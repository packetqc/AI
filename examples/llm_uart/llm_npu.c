/**
 * llm_npu.c
 *
 * STAI inference wrapper — replaces aiValidationProcess with a real
 * autoregressive text-generation loop using the on-device Qwen2 model.
 *
 * Memory layout (no extra buffers allocated here):
 *   - network context   : declared in app_x-cube-ai.c  (extern via stai_mnetwork_*)
 *   - activation RAM    : data_activations[] from app_x-cube-ai.c (extern)
 *   - state buffers     : data_states[]      from app_x-cube-ai.c (extern)
 *   - input/output ptrs : obtained from stai_mnetwork_get_inputs/outputs at runtime
 *                         (the network owns the buffers inside activation RAM)
 *
 * Output tensor: int8[1 × 32 × 1 × 374], channel-last
 *   argmax is valid directly on int8 because scale > 0 (monotonic mapping).
 *   logits[pos, v] = arr[(pos * 374) + v]
 */

#include "llm_npu.h"
#include "llm_tokenizer.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* STAI multi-network API from app_x-cube-ai.c */
#include "app_x-cube-ai.h"
#include "network.h"
#include "bsp_ai.h"   /* huart1 / UartHandle */

extern stai_ptr data_activations[];
extern stai_ptr data_states[];

/* ── Static state ─────────────────────────────────────────────────────── */

static stai_ptr s_net_handle = NULL;

/* Pointers INTO the network's own activation-RAM tensors (set at init) */
static stai_ptr s_inp[STAI_NETWORK_IN_NUM];
static stai_ptr s_out[STAI_NETWORK_OUT_NUM];

/* ── Helpers ──────────────────────────────────────────────────────────── */

static void uart_print(const char *s)
{
    HAL_UART_Transmit(&UartHandle, (const uint8_t *)s, (uint16_t)strlen(s), 1000);
}

/* Read one line from UART into buf (blocking, echoes chars, ends on \r or \n) */
static int uart_readline(char *buf, int max_len)
{
    int n = 0;
    uint8_t ch;
    while (n < max_len - 1) {
        if (HAL_UART_Receive(&UartHandle, &ch, 1, HAL_MAX_DELAY) != HAL_OK)
            break;
        HAL_UART_Transmit(&UartHandle, &ch, 1, 100);  /* echo */
        if (ch == '\r' || ch == '\n') break;
        if (ch == 0x08 && n > 0) { n--; continue; }  /* backspace */
        buf[n++] = (char)ch;
    }
    buf[n] = '\0';
    uart_print("\r\n");
    return n;
}

/* Argmax over int8 array — valid substitute for float argmax (scale > 0) */
static int argmax_i8(const int8_t *arr, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[best]) best = i;
    return best;
}

/* ── Public API ───────────────────────────────────────────────────────── */

int LLM_NPU_Init(void)
{
    stai_return_code err;

    err = stai_mnetwork_init(STAI_NETWORK_MODEL_NAME, &s_net_handle);
    if (err != STAI_SUCCESS) return -(int)err;

    err = stai_mnetwork_set_activations(s_net_handle, data_activations);
    if (err != STAI_SUCCESS) goto fail;

    err = stai_mnetwork_set_states(s_net_handle, data_states);
    if (err != STAI_SUCCESS) goto fail;

    /* Obtain network's own input/output buffer addresses in activation RAM */
    err = stai_mnetwork_get_inputs(s_net_handle, s_inp);
    if (err != STAI_SUCCESS) goto fail;

    err = stai_mnetwork_get_outputs(s_net_handle, s_out);
    if (err != STAI_SUCCESS) goto fail;

    return 0;

fail:
    stai_mnetwork_deinit(s_net_handle);
    s_net_handle = NULL;
    return -(int)err;
}

void LLM_NPU_Deinit(void)
{
    if (s_net_handle) {
        stai_mnetwork_deinit(s_net_handle);
        s_net_handle = NULL;
    }
}

int LLM_NPU_RunOnce(const int32_t *input_ids, const int32_t *attn_mask,
                    int seq_len, int32_t *next_token_id)
{
    if (!s_net_handle) return -1;

    /* Copy into network's internal input tensors (int32, 32 elements each) */
    int32_t *net_ids  = (int32_t *)s_inp[0];
    int32_t *net_mask = (int32_t *)s_inp[1];

    memcpy(net_ids,  input_ids, LLM_SEQ_LEN * sizeof(int32_t));
    memcpy(net_mask, attn_mask, LLM_SEQ_LEN * sizeof(int32_t));

    stai_return_code err = stai_mnetwork_run(s_net_handle);
    if (err != STAI_SUCCESS) return -(int)err;

    /* Output: int8[1 × 32 × 1 × 374], channel-last.
     * Logits at the last real token position: offset = (seq_len-1) × 374 */
    const int8_t *logits = (const int8_t *)s_out[0];
    int last = (seq_len > 0 && seq_len <= LLM_SEQ_LEN) ? seq_len - 1 : LLM_SEQ_LEN - 1;
    *next_token_id = (int32_t)argmax_i8(logits + last * LLM_VOCAB_SIZE, LLM_VOCAB_SIZE);

    return 0;
}

int LLM_Generate(const char *prompt, char *out_buf, int out_max, int max_new_tokens)
{
    int32_t ids[LLM_SEQ_LEN]  = {0};
    int32_t mask[LLM_SEQ_LEN] = {0};
    int out_pos = 0;
    int new_tokens = 0;

    int n = LLM_Tokenize(prompt, ids, mask, LLM_SEQ_LEN);
    if (n <= 0) return 0;

    while (new_tokens < max_new_tokens) {
        int32_t next_tok;
        if (LLM_NPU_RunOnce(ids, mask, n, &next_tok) != 0) return -1;

        if (next_tok == LLM_EOS_TOKEN) break;

        /* Append decoded text to output buffer */
        const char *tok_str = LLM_TokenStr(next_tok);
        int tok_len = (int)strlen(tok_str);
        if (out_pos + tok_len < out_max - 1) {
            memcpy(out_buf + out_pos, tok_str, tok_len);
            out_pos += tok_len;
        }
        new_tokens++;

        /* Slide context window: shift left if full, append new token */
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

void LLM_Process(void)
{
    char prompt[128];
    char response[256];

    uart_print("\r\n=== Qwen2 LLM on STM32N6 NPU ===\r\n");
    uart_print("Type a prompt and press Enter (empty line to quit).\r\n\r\n");

    for (;;) {
        uart_print("> ");
        int n = uart_readline(prompt, sizeof(prompt));
        if (n == 0) break;

        int ntok = LLM_Generate(prompt, response, sizeof(response), 32);
        if (ntok < 0) {
            uart_print("[inference error]\r\n");
        } else {
            uart_print(response);
            uart_print("\r\n");
        }
    }

    uart_print("Done.\r\n");
}
