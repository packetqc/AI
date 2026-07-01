/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_azure_rtos.c
  * @author  MCD Application Team
  * @brief   app_azure_rtos application implementation file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "d1a_diag.h"
#include "app_azure_rtos.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

#if (USE_STATIC_ALLOCATION == 1)

/* USER CODE BEGIN TX_Pool_Buffer */
/* USER CODE END TX_Pool_Buffer */
#if defined ( __ICCARM__ )
#pragma data_alignment=4
#endif
/* phase2.6q — pool buffer relocated to the .late_buf NOLOAD section so
 * the 32 KB allocation does NOT push main .bss past the BSS-cliff at
 * ~0x341505C0 (which would NULL-deref _tx_timer_current_ptr at boot).
 * Future thread stacks allocated from this pool inherit the safe location. */
__ALIGN_BEGIN static UCHAR tx_byte_pool_buffer[TX_APP_MEM_POOL_SIZE]
    __attribute__((section(".late_buf"), aligned(8))) __ALIGN_END;
static TX_BYTE_POOL tx_app_byte_pool;

/* USER CODE BEGIN UX_Pool_Buffer */
/* USER CODE END UX_Pool_Buffer */
#if defined ( __ICCARM__ )
#pragma data_alignment=4
#endif
__ALIGN_BEGIN static UCHAR ux_byte_pool_buffer[UX_APP_MEM_POOL_SIZE] __ALIGN_END;
static TX_BYTE_POOL ux_app_byte_pool;

/* phase-11-tx-D1a — USBPD DPM byte pool. Placed in .late_buf (NOLOAD)
 * like tx_byte_pool_buffer so the 5 KB allocation does NOT push main
 * .bss past the BSS-cliff at ~0x341505C0 (boot NULL-deref hard rule). */
__ALIGN_BEGIN static UCHAR usbpd_byte_pool_buffer[USBPD_DEVICE_APP_MEM_POOL_SIZE]
    __attribute__((section(".late_buf"), aligned(8))) __ALIGN_END;
static TX_BYTE_POOL usbpd_app_byte_pool;

/* iter12 managed-boot: USBPD source chain (PreInitOs + MX_USBPD_Init) is
 * DEFERRED out of tx_application_define into the RX-pump, gated on the
 * RTL-USB2 init window passing — iter10/iter11 HW-proved the RTL-USB2
 * -30/0x23 disruptor is the boot-active USB-C/USBPD subsystem, not the
 * OTG1 host SW. The byte pool is still created at kernel-init (harmless);
 * its address is stashed here for the deferred MX_USBPD_Init() call. */
void *g_usbpd_memory_ptr = (void *)0;

#endif

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/**
  * @brief  Define the initial system.
  * @param  first_unused_memory : Pointer to the first unused memory
  * @retval None
  */
VOID tx_application_define(VOID *first_unused_memory)
{
  /* USER CODE BEGIN  tx_application_define_1*/
  /* G2-E flow markers @ 0x34127014 (distinct from DPM diag @0x34127000):
   * m=1 entry, 2 after App_ThreadX_Init OK, 3 after MX_USBX_Init OK,
   * 4 top of USBPD block, 5 after MX_USBPD_Init. Pinpoints where
   * tx_application_define flow diverges (DPM marker [0] stayed 0). */
  d1a_diag(0x14, 0x0F100001u);
  /* USER CODE END  tx_application_define_1 */
#if (USE_STATIC_ALLOCATION == 1)
  UINT status = TX_SUCCESS;
  VOID *memory_ptr;

  if (tx_byte_pool_create(&tx_app_byte_pool, "Tx App memory pool", tx_byte_pool_buffer, TX_APP_MEM_POOL_SIZE) != TX_SUCCESS)
  {
    /* USER CODE BEGIN TX_Byte_Pool_Error */

    /* USER CODE END TX_Byte_Pool_Error */
  }
  else
  {
    /* USER CODE BEGIN TX_Byte_Pool_Success */

    /* USER CODE END TX_Byte_Pool_Success */

    memory_ptr = (VOID *)&tx_app_byte_pool;
    status = App_ThreadX_Init(memory_ptr);
    if (status != TX_SUCCESS)
    {
      /* USER CODE BEGIN  App_ThreadX_Init_Error */
      while(1)
      {
      }
      /* USER CODE END  App_ThreadX_Init_Error */
    }
    /* USER CODE BEGIN  App_ThreadX_Init_Success */
    d1a_diag(0x14, 0x0F100002u);
    /* USER CODE END  App_ThreadX_Init_Success */

  }

  if (tx_byte_pool_create(&ux_app_byte_pool, "Ux App memory pool", ux_byte_pool_buffer, UX_APP_MEM_POOL_SIZE) != TX_SUCCESS)
  {
    /* USER CODE BEGIN UX_Byte_Pool_Error */

	/* USER CODE END UX_Byte_Pool_Error */
  }
  else
  {
    /* USER CODE BEGIN UX_Byte_Pool_Success */

    /* USER CODE END UX_Byte_Pool_Success */

    memory_ptr = (VOID *)&ux_app_byte_pool;
    status = MX_USBX_Init(memory_ptr);
    if (status != UX_SUCCESS)
    {
      /* USER CODE BEGIN  MX_USBX_Init_Error */
      while(1)
      {
      }
      /* USER CODE END  MX_USBX_Init_Error */
    }
    /* USER CODE BEGIN  MX_USBX_Init_Success */
    d1a_diag(0x14, 0x0F100003u);
    /* USER CODE END  MX_USBX_Init_Success */
  }

  /* phase-11-tx-D1a G2-D — USBPD DPM byte pool + init (mirrors ST
   * USBPD_SRC_UX_Host_MSC app_azure_rtos.c). Starts the UCPD/Type-C
   * source state machine + DPM ThreadX task; on sink (RTL-SDR) attach
   * it sources VBUS via usbpd_pwr_if -> BSP_USBPD_PWR -> TCPP0203.
   * USBPD_OK == 0 (usbpd_def.h); extern decl avoids pulling USBPD
   * headers into this build unit. */
  {
    extern unsigned int MX_USBPD_Init(void *memory_ptr);
    extern unsigned int USBPD_PreInitOs(void); /* FSBL/USBPD/App/usbpd.c */
    d1a_diag(0x14, 0x0F100004u);  /* m4: USBPD block top */

    /* phase-11-tx-D1a G2-E ROOT-CAUSE FIX: USBPD has TWO init entry
     * points. MX_USBPD_Init (below) only does USBPD_DPM_UserInit +
     * USBPD_DPM_InitOS. The CAD/PE core that powers the TCPP0203
     * (BSP_USBPD_PWR_Init) and runs the Type-C source attach SM is
     * USBPD_DPM_InitCore, reached ONLY via USBPD_PreInitOs()
     * (= USBPD_HW_IF_GlobalHwInit + USBPD_DPM_InitCore). SWD diag had
     * proved BSP_USBPD_PWR_Init [6]=0 and CAD_StateMachine_SRC [7]=0
     * (never ran) because PreInitOs was never called. ST main.c calls
     * it before MX_ThreadX_Init, but in our ThreadX SysTick binding
     * that pre-RTOS slot faults (its HAL/I2C delays let SysTick enter
     * _tx_thread_time_slice with the kernel uninitialized). Calling it
     * HERE -- inside tx_application_define, after _tx_initialize, just
     * before MX_USBPD_Init -- is the safe equivalent: kernel pointer
     * is valid so the SysTick path is harmless, and core init still
     * precedes the user/OS-task init exactly as ST orders it. */
    (void)USBPD_PreInitOs;  /* extern kept; called DEFERRED from RX-pump */

    /* iter12 MANAGED BOOT — DEFER the USB-C/USBPD source chain.
     * iter10 (delay ResetPort) + iter11 (defer whole USB1 OTG host —
     * proved dormant 72s) BOTH HW-failed: RTL-USB2 still -30/0x23 +
     * flap. The boot-active disruptor is the USB-C/USBPD source
     * subsystem itself (USBPD_PreInitOs = USBPD_HW_IF_GlobalHwInit +
     * USBPD_DPM_InitCore → CAD/PE/TCPP0203 SM, + MX_USBPD_Init DPM
     * task), which RTL-only never had (no USB-C device → idle). So
     * create the pool here (harmless) but DON'T start USBPD; stash the
     * pool ptr and let the RX-pump fire USBPD_PreInitOs()+MX_USBPD_Init()
     * only AFTER the RTL-USB2 demod window has passed (g_rtlsdr_init_steps
     * & 0x04), then bring USB1 host up. RTL inits in total isolation =
     * the proven-good RTL-only condition. User directive: "init rtl
     * first then hackrf"; "manage the boot". */
    if (tx_byte_pool_create(&usbpd_app_byte_pool, "USBPD App memory pool",
            usbpd_byte_pool_buffer, USBPD_DEVICE_APP_MEM_POOL_SIZE) != TX_SUCCESS)
    {
      /* USER CODE BEGIN USBPD_Byte_Pool_Error */
      d1a_diag(0x14, 0x0F10E004u);  /* pool fail */
      while (1) { }
      /* USER CODE END USBPD_Byte_Pool_Error */
    }
    else
    {
      memory_ptr = (VOID *)&usbpd_app_byte_pool;
      g_usbpd_memory_ptr = (void *)&usbpd_app_byte_pool;  /* deferred MX_USBPD_Init */
      (void)memory_ptr;
    }
  }
#else
/*
 * Using dynamic memory allocation requires to apply some changes to the linker file.
 * ThreadX needs to pass a pointer to the first free memory location in RAM to the tx_application_define() function,
 * using the "first_unused_memory" argument.
 * This require changes in the linker files to expose this memory location.
 * For EWARM add the following section into the .icf file:
     place in RAM_region    { last section FREE_MEM };
 * For MDK-ARM
     - either define the RW_IRAM1 region in the ".sct" file
     - or modify the line below in "tx_initialize_low_level.S to match the memory region being used
        LDR r1, =|Image$$RW_IRAM1$$ZI$$Limit|

 * For STM32CubeIDE add the following section into the .ld file:
     ._threadx_heap :
       {
          . = ALIGN(8);
          __RAM_segment_used_end__ = .;
          . = . + 64K;
          . = ALIGN(8);
        } >RAM_D1 AT> RAM_D1
    * The simplest way to provide memory for ThreadX is to define a new section, see ._threadx_heap above.
    * In the example above the ThreadX heap size is set to 64KBytes.
    * The ._threadx_heap must be located between the .bss and the ._user_heap_stack sections in the linker script.
    * Caution: Make sure that ThreadX does not need more than the provided heap memory (64KBytes in this example).
    * Read more in STM32CubeIDE User Guide, chapter: "Linker script".

 * The "tx_initialize_low_level.S" should be also modified to enable the "USE_DYNAMIC_MEMORY_ALLOCATION" flag.
 */

  /* USER CODE BEGIN DYNAMIC_MEM_ALLOC */
  (void)first_unused_memory;
  /* USER CODE END DYNAMIC_MEM_ALLOC */
#endif

}
