/**
 * fb_sram.h — on-chip SRAM framebuffer bring-up (reference layout, option B).
 *
 * Prepares the AXISRAM banks that hold the LTDC double-buffer for the 100%-SRAM layout
 * (FB_IN_SRAM=1 in fb_layout.h): MPU Normal-Non-Cacheable regions on the FB banks only, and a
 * power/clock sanity check. No-op when FB_IN_SRAM=0 (legacy PSRAM). See MEMORY_MAP.md.
 */
#ifndef FB_SRAM_H
#define FB_SRAM_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Make the on-chip framebuffer banks LTDC-scannable and coherent with D-cache ON:
 *         mark AXISRAM1 (front) and AXISRAM5-6 (back) MPU Normal-Non-Cacheable so LVGL pixel
 *         writes go straight to SRAM and the LTDC never reads a stale cache line — while leaving
 *         the NPU activation banks (AXISRAM3-4) cacheable write-back (ll_aton needs WB there).
 *         The RISAF read grant for the LTDC master is already covered by the NPU all-CID grant
 *         in SystemIsolation_Config; the banks are powered by Enable_NPU_RAM_ForCore.
 *         Call once at boot AFTER the MPU/PSRAM setup, BEFORE lvgl_port_n6_init.
 *         Compiles to an empty function when FB_IN_SRAM=0.
 */
void fb_sram_enable(void);

#ifdef __cplusplus
}
#endif
#endif /* FB_SRAM_H */
