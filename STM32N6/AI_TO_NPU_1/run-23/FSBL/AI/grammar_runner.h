#ifndef GRAMMAR_RUNNER_H
#define GRAMMAR_RUNNER_H

#include "network.h"   /* stai_network */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parse + evaluate input_str (e.g. "3 + 4") via the NPU grammar oracle: the runner
 * queries the NPU for each grammar rule it needs (multi-step dialog, logged), parses
 * the input with those rules, and evaluates the tree. Returns the numeric result;
 * *ok = 1 on success, 0 on parse failure. This is the device port of GrammarRunner. */
long Grammar_Calc(stai_network *net, int8_t *in_buf, const int8_t *out_buf,
                  const char *input_str, int *ok);

#ifdef __cplusplus
}
#endif

#endif /* GRAMMAR_RUNNER_H */
