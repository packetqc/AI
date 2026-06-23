/**
 * llm_npu.h
 *
 * STAI inference wrapper for the on-device Qwen2 LLM.
 *
 * Typical flow:
 *   1. Call LLM_NPU_Init() once (replaces aiValidationInit).
 *   2. Call LLM_Generate() for each prompt.
 *   3. Optionally call LLM_NPU_Deinit() on shutdown.
 *
 * Integration:
 *   In app_x-cube-ai.c, inside the USER CODE blocks:
 *     STM32CubeAI_Studio_AI_Init    → call LLM_NPU_Init()
 *     STM32CubeAI_Studio_AI_Process → call LLM_Process()   [interactive loop]
 *   Or call LLM_TestCalc() from main.c for the self-test.
 */
#ifndef LLM_NPU_H
#define LLM_NPU_H

#include <stdint.h>

/* Output QLinear parameters (from model_npu_qdq.onnx export) */
#define LLM_OUT_SCALE      0.069436282f
#define LLM_OUT_ZERO_POINT (-52)

/**
 * Initialise the STAI network and prepare input/output buffer pointers.
 * Must be called before any Generate call.
 * Returns 0 on success, negative on STAI error.
 */
int LLM_NPU_Init(void);

/**
 * Run one forward pass with preloaded ids/mask (length LLM_SEQ_LEN = 32).
 * Writes the predicted next-token ID to *next_token_id.
 * Returns 0 on success.
 */
int LLM_NPU_RunOnce(const int32_t *input_ids, const int32_t *attn_mask,
                    int seq_len, int32_t *next_token_id);

/**
 * Tokenize prompt, then autoregressively generate up to max_new_tokens.
 * Decoded output (NUL-terminated) written to out_buf[0..out_max-1].
 * Returns number of new tokens generated, or negative on error.
 */
int LLM_Generate(const char *prompt, char *out_buf, int out_max, int max_new_tokens);

/**
 * Release the STAI network handle.
 */
void LLM_NPU_Deinit(void);

/**
 * Blocking interactive UART loop: read prompt → generate → print.
 * Call from STM32CubeAI_Studio_AI_Process() to replace validation loop.
 */
void LLM_Process(void);

#endif /* LLM_NPU_H */
