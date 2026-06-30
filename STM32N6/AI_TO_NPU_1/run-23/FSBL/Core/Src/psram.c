/**
 * psram.c — minimal external PSRAM (XSPI1 @0x90000000) enablement + self-test for run-23.
 *
 * run-23's NPU runs from AXISRAM, leaving the on-board PSRAM free. We map it (so the LTDC
 * framebuffer + app data can live there) with the board BSP — the same minimal sequence the
 * reference project uses — then self-test it before relying on it. The LTDC reads the framebuffer
 * over its own AXI master (no M55 D-cache), so callers must clean the D-cache after CPU writes.
 */
#include "psram.h"
#include "stm32n6570_discovery_xspi.h"
#include "main.h"
#include <stdio.h>

int PSRAM_Init(void)
{
    if (BSP_XSPI_RAM_Init(0) != BSP_ERROR_NONE)                   return -1;
    if (BSP_XSPI_RAM_EnableMemoryMappedMode(0) != BSP_ERROR_NONE) return -2;
    /* automatic prefetch off — same as the reference (avoids speculative reads past the device). */
    MODIFY_REG(XSPI1->CR, XSPI_CR_NOPREF, HAL_XSPI_AUTOMATIC_PREFETCH_DISABLE);
    return 0;
}

int PSRAM_SelfTest(void)
{
    static const uint32_t pat[] = { 0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0x00000000u, 0xFFFFFFFFu };
    const uint32_t base = PSRAM_BASE;
    const uint32_t span = 0x00200000UL;        /* 2 MB — covers the 800x480 framebuffer region */
    const uint32_t step = 0x00040000UL;        /* 256 KB stride */
    int errors = 0;
    for (uint32_t k = 0; k < sizeof(pat) / sizeof(pat[0]); ++k) {
        for (uint32_t a = base; a < base + span; a += step) *(volatile uint32_t *)a = pat[k];
        SCB_CleanDCache_by_Addr((uint32_t *)base, span);
        SCB_InvalidateDCache_by_Addr((uint32_t *)base, span);
        for (uint32_t a = base; a < base + span; a += step)
            if (*(volatile uint32_t *)a != pat[k]) ++errors;
    }
    printf("\r\nPSRAM self-test @0x%08lX (2MB / %u patterns): %s\r\n",
           (unsigned long)base, (unsigned)(sizeof(pat) / sizeof(pat[0])),
           errors ? "FAIL" : "PASS");
    return errors ? -1 : 0;
}
