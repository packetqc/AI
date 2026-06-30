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

/* Mark the PSRAM (0x90000000, 4 MB) Normal Non-Cacheable via MPU region 0 + enable the MPU with the
 * default map (PRIVDEFENA) for everything else. Stops the LTDC-reads-stale-D-cache glitch on the
 * framebuffer. Call once after PSRAM_Init(), before the framebuffer is used. */
void PSRAM_Mpu(void);

#ifdef __cplusplus
}
#endif

#endif /* PSRAM_H */
