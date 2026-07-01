/**
 * fb_sram.c — on-chip SRAM framebuffer bring-up (reference layout, option B). See fb_sram.h.
 */
#include "fb_sram.h"
#include "fb_layout.h"
#include "main.h"

#if FB_IN_SRAM

void fb_sram_enable(void)
{
    /* Mark ONLY the FB banks Normal-Non-Cacheable so that, with the M55 D-cache ON, LVGL's pixel
     * writes bypass the cache (straight to SRAM) and the LTDC master always fetches fresh pixels.
     * The NPU activation banks (AXISRAM3-4) are deliberately left OUT of these regions — ll_aton
     * needs them cacheable write-back. Two regions because front and back live in different banks. */
    MPU_Region_InitTypeDef     mpu  = {0};
    MPU_Attributes_InitTypeDef attr = {0};

    HAL_MPU_Disable();

    attr.Number     = MPU_ATTRIBUTES_NUMBER1;
    attr.Attributes = INNER_OUTER(MPU_NOT_CACHEABLE);   /* 0x44 — same encoding as the PSRAM region */
    HAL_MPU_ConfigMemoryAttributes(&attr);

    /* Region 1: FB FRONT — AXISRAM1 (0x34000000, 1 MB; FB uses the low 768 KB). */
    mpu.Enable           = MPU_REGION_ENABLE;
    mpu.Number           = MPU_REGION_NUMBER1;
    mpu.BaseAddress      = 0x34000000UL;
    mpu.LimitAddress     = 0x340FFFFFUL;
    mpu.AttributesIndex  = MPU_ATTRIBUTES_NUMBER1;
    mpu.AccessPermission = MPU_REGION_ALL_RW;
    mpu.IsShareable      = MPU_ACCESS_OUTER_SHAREABLE;
    mpu.DisableExec      = MPU_INSTRUCTION_ACCESS_DISABLE;
    HAL_MPU_ConfigRegion(&mpu);

    /* Region 2: FB BACK — AXISRAM5-6 (0x342E0000..0x343BFFFF, 896 KB). */
    mpu.Number       = MPU_REGION_NUMBER2;
    mpu.BaseAddress  = 0x342E0000UL;
    mpu.LimitAddress = 0x343BFFFFUL;
    HAL_MPU_ConfigRegion(&mpu);

    /* Region 0 (PSRAM, from psram.c) is preserved — untouched by the disable/enable cycle. */
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

#else  /* legacy PSRAM framebuffer — nothing to enable on-chip */

void fb_sram_enable(void) { }

#endif /* FB_IN_SRAM */
