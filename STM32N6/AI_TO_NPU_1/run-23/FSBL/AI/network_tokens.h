#ifndef NETWORK_TOKENS_H
#define NETWORK_TOKENS_H
#include <stdint.h>

#define TOK_VOCAB   374
#define TOK_EOS_ID  0
#define TOK_NUM_RULES 5
#define TOK_MAX_PROMPT 2

extern const char* const g_tok_decode[TOK_VOCAB];      /* id -> display string */
extern const char* const g_rule_names[TOK_NUM_RULES];  /* rule name strings */
extern const int16_t     g_rule_prompt[TOK_NUM_RULES][TOK_MAX_PROMPT]; /* token ids, -1 pad */
extern const uint8_t     g_rule_prompt_len[TOK_NUM_RULES];

#endif
