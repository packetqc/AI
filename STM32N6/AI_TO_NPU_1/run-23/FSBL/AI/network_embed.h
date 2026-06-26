#ifndef NETWORK_EMBED_H
#define NETWORK_EMBED_H
#include <stdint.h>

/* int8 token embedding table [vocab][dim] = g_embed_table[token*EMBED_DIM + ch].
 * Pre-quantized with the NPU input scale=0.029472799971699715 zero_point=-4. */
#define EMBED_VOCAB 374
#define EMBED_DIM   256

extern const int8_t g_embed_table[374 * 256];

#endif /* NETWORK_EMBED_H */
