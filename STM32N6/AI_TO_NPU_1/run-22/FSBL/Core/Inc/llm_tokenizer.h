/**
 * llm_tokenizer.h
 *
 * BPE tokenizer for the on-device Qwen2 model (vocab_size=374, 117 merges).
 * Encode ASCII text to token IDs; decode token IDs back to strings.
 *
 * Integration: add llm_tokenizer.c to your project's Makefile source list.
 */
#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#define LLM_VOCAB_SIZE   374
#define LLM_MERGES_NUM   117
#define LLM_SEQ_LEN      32
#define LLM_EOS_TOKEN    0    /* <|endoftext|> */

/**
 * Encode text into a fixed-length token ID + attention mask buffer.
 *
 * @param text     NUL-terminated ASCII input string
 * @param ids      Output: int32[max_len], filled from position 0 (padded with 0)
 * @param mask     Output: int32[max_len], 1 for real tokens, 0 for padding
 * @param max_len  Buffer length (use LLM_SEQ_LEN = 32)
 * @return         Number of real (non-padding) tokens encoded (≤ max_len)
 */
int LLM_Tokenize(const char *text, int32_t *ids, int32_t *mask, int max_len);

/**
 * Return a NUL-terminated string for a token ID (points to static table).
 * Returns "?" for out-of-range IDs.
 */
const char *LLM_TokenStr(int32_t token_id);

#endif /* LLM_TOKENIZER_H */
