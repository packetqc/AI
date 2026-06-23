/**
 * llm_fsbl.h
 *
 * FSBL-mode STAI inference wrapper.
 * Uses stai_network_* directly (no app_x-cube-ai.c mnetwork layer needed).
 * Activation buffer placed in .AI_RAM section → AXISRAM3 (0x34200000).
 *
 * Include path requirements (add to FSBL Makefile C_INCLUDES):
 *   -I../../Appli/AI/App          (network.h, user_init.h, stai.h)
 *   -I../../Appli/Core/Inc        (llm_tokenizer.h, llm_test.h)
 *   -I../../Middlewares/ST/AI/Inc (stai.h, stai_debug.h)
 */
#ifndef LLM_FSBL_H
#define LLM_FSBL_H

#include <stdint.h>

/**
 * Initialise STAI context, activation buffer, weights, and buffer pointers.
 * Call once from FSBL main USER CODE before the test loop.
 * Returns 0 on success, negative STAI error code on failure.
 */
int LLM_FSBL_Init(void);

/**
 * Run one inference pass: fill input_ids/attn_mask → run → argmax last position.
 * Returns 0 on success.
 */
int LLM_FSBL_RunOnce(const int32_t *input_ids, const int32_t *attn_mask,
                     int seq_len, int32_t *next_token_id);

/**
 * Tokenize prompt → autoregressive generate up to max_new_tokens.
 * Decoded tokens appended to out_buf (NUL-terminated).
 * Returns number of tokens generated, or negative on error.
 */
int LLM_FSBL_Generate(const char *prompt, char *out_buf, int out_max,
                      int max_new_tokens);

/**
 * Deinit the network context.
 */
void LLM_FSBL_Deinit(void);

#endif /* LLM_FSBL_H */
