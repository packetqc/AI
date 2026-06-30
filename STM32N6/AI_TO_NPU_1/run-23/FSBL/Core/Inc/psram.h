/**
 * psram.h — minimal external PSRAM (XSPI1 @0x90000000) enablement + self-test for run-23.
 */
#ifndef PSRAM_H
#define PSRAM_H

#ifdef __cplusplus
extern "C" {
#endif

#define PSRAM_BASE   0x90000000UL          /* XSPI1 PSRAM, memory-mapped */

/* Map the PSRAM (BSP init + memory-mapped mode + prefetch off). Returns 0 on success. */
int PSRAM_Init(void);

/* Write/clean/invalidate/read a few patterns across the framebuffer region; prints PASS/FAIL
 * over the VCP. Returns 0 if all patterns read back. Only call after PSRAM_Init() succeeded. */
int PSRAM_SelfTest(void);

#ifdef __cplusplus
}
#endif

#endif /* PSRAM_H */
