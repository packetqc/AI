#ifndef NPU_QUERY_H
#define NPU_QUERY_H

#include "network.h"   /* stai_network, STAI_* dims */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Autoregressive grammar-oracle query: rule_idx (0..TOK_NUM_RULES-1) -> decoded
 * rule body string into out[out_max]. Uses the already-initialized network + its
 * preallocated int8 input/output buffers. This is the device equivalent of the
 * host get_ollama_answer() — the NPU is the model host (the N6's "Ollama"). */
void NPU_QueryRule(stai_network *net, int8_t *in_buf, const int8_t *out_buf,
                   int rule_idx, char *out, int out_max);

#ifdef __cplusplus
}
#endif

#endif /* NPU_QUERY_H */
