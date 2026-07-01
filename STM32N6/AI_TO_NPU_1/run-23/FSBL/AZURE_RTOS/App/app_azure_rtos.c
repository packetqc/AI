/**
 * app_azure_rtos.c — minimal ThreadX bring-up for run-23 (calculator FSBL).
 *
 * Trimmed from the reference N6_EDGEAI_1 (which also created USBX/USBPD pools). run-23 needs one thing:
 * create the application byte pool, then hand it to App_ThreadX_Init (app_threadx.c) which spawns the
 * LVGL render thread + the demo/inference thread.
 *
 * tx_kernel_enter() (from MX_ThreadX_Init) calls tx_application_define() once, on the timer/system
 * stack, before the scheduler starts. Only ThreadX object creation + thread spawns belong here.
 */
#include "app_azure_rtos.h"

#if (USE_STATIC_ALLOCATION == 1)

/* App byte pool storage. Kept OUT of main .bss (which lives in AXISRAM2 alongside the FSBL image) — a
 * 48 KB block there would push .bss toward the AXISRAM3 NPU region. Placed in a dedicated .threadx_pool
 * section the linker maps into freed AXISRAM (see STM32N657XX_fsbl_custom.ld). */
static UCHAR tx_byte_pool_buffer[TX_APP_MEM_POOL_SIZE]
    __attribute__((section(".threadx_pool"), aligned(8)));
static TX_BYTE_POOL tx_app_byte_pool;

#endif

VOID tx_application_define(VOID *first_unused_memory)
{
#if (USE_STATIC_ALLOCATION == 1)
  (void)first_unused_memory;
  if (tx_byte_pool_create(&tx_app_byte_pool, "run23 app pool",
                          tx_byte_pool_buffer, TX_APP_MEM_POOL_SIZE) == TX_SUCCESS)
  {
    (void)App_ThreadX_Init((VOID *)&tx_app_byte_pool);
  }
#else
  (void)first_unused_memory;
#endif
}
