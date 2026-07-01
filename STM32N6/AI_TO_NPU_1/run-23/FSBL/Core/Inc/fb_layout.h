/**
 * fb_layout.h — framebuffer memory layout selector (single switch).
 *
 * Reference memory architecture for LVGL bare-metal + NPU on STM32N6 (see MEMORY_MAP.md).
 * Flip FB_IN_SRAM to move the LTDC double-buffer off the contended external PSRAM into dedicated,
 * NPU-free on-chip AXISRAM banks — the real fix for the inference-time scanout starvation.
 *
 *   FB_IN_SRAM 0  (default) — LEGACY: both buffers in external PSRAM (0x90000000 / +1 MB). The LTDC
 *                 scanout shares the slow PSRAM path with the NPU's AXI flood -> inference flicker,
 *                 mitigated by the per-epoch display gate (g_npu_gate). No network re-gen required.
 *
 *   FB_IN_SRAM 1  — REFERENCE (option B, 100% SRAM): front in AXISRAM1 (0x34000000), back in
 *                 AXISRAM5-6 (0x342E0000). Both on-chip, on their own AXI slave ports, in banks the
 *                 NPU never touches -> the LTDC scans concurrently with inference: glitch-free, full
 *                 color, 60 Hz, gate not needed (kept only as a runtime fallback).
 *                 PREREQUISITE: the option-B network re-gen that FREES AXISRAM1 of the NPU weights
 *                 (weights + activations packed into AXISRAM3-4-7 — see network_regen/). Until that
 *                 re-gen lands, FB_FRONT_ADDR (0x34000000) OVERLAPS the weights copied to 0x34064000;
 *                 do NOT set FB_IN_SRAM=1 before the re-gen, and the banks must be RAMCFG-enabled and
 *                 RISAF-granted to the LTDC master (see fb_sram_enable()).
 */
#ifndef FB_LAYOUT_H
#define FB_LAYOUT_H

#ifndef FB_IN_SRAM
#define FB_IN_SRAM 0
#endif

#if FB_IN_SRAM
  #define FB_FRONT_ADDR  0x34000000UL   /* AXISRAM1 (1 MB) — freed of weights by the option-B re-gen */
  #define FB_BACK_ADDR   0x342E0000UL   /* AXISRAM5-6 (896 KB, free) */
  #define FB_MPU_BASE    0x34000000UL   /* MPU Normal-Non-Cacheable region covering both FB banks */
  #define FB_MPU_LIMIT   0x343BFFFFUL   /* 0x34000000..0x343BFFFF: AXISRAM1 + gap + AXISRAM5-6 */
#else
  #define FB_FRONT_ADDR  0x90000000UL   /* PSRAM */
  #define FB_BACK_ADDR   0x90100000UL   /* PSRAM + 1 MB */
  #define FB_MPU_BASE    0x90000000UL   /* existing PSRAM non-cacheable region (psram.c) */
  #define FB_MPU_LIMIT   0x903FFFFFUL
#endif

#define FB_WIDTH    800U
#define FB_HEIGHT   480U
#define FB_BPP      2U                                   /* RGB565 */
#define FB_SIZE     (FB_WIDTH * FB_HEIGHT * FB_BPP)      /* 768000 bytes per buffer */

#endif /* FB_LAYOUT_H */
