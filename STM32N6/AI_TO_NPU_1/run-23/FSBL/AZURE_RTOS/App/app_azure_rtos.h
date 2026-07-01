/**
 * app_azure_rtos.h — minimal ThreadX framework config for run-23.
 *
 * The reference N6_EDGEAI_1 header pulls in USBX + a generated config; run-23 has neither. This trims
 * it to what the calculator FSBL needs: static allocation + a single application byte pool sized for the
 * LVGL render thread + the demo/inference thread stacks.
 */
#ifndef APP_AZURE_RTOS_H
#define APP_AZURE_RTOS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tx_api.h"
#include "app_threadx.h"

/* Static allocation — the app byte pool lives in a fixed buffer (see app_azure_rtos.c). */
#define USE_STATIC_ALLOCATION   1

/* App byte pool: LVGL render thread (32 KB — lv_timer_handler is stack-heavy; bare-metal ran it on the
 * ~50 KB main stack) + demo/inference thread (16 KB) + headroom/alignment. Placed by app_azure_rtos.c
 * into a dedicated .threadx_pool section (freed AXISRAM, not main .bss). */
#define TX_APP_MEM_POOL_SIZE    (80U * 1024U)

#ifdef __cplusplus
}
#endif

#endif /* APP_AZURE_RTOS_H */
