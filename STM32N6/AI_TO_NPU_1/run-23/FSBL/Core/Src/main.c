/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
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
#include "main.h"
#include "extmem_manager.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
// #include "stm32n6570_discovery_xspi.h"
// #include "stm32n6570_discovery_errno.h"
#include "stm32n6570_discovery.h"    /* BSP: COM1 VCP UART + LEDs (board-canonical) */
#include "stai_network.h"            /* STAI API + the TCN NPU model (Neural-ART HW) */
#include "network_tokens.h"          /* device tokenizer support (rule prompts + decode) */
#include "npu_query.h"               /* grammar-oracle bridge (autoregressive NPU recall) */
#include "grammar_runner.h"          /* parse + evaluate via the oracle (the calculator) */
#include "lcd.h"                     /* LTDC + panel bring-up (display) */
#include "psram.h"                   /* external PSRAM map + self-test (framebuffer + app data) */
#include "lvgl_port_n6.h"            /* LVGL 9.x DIRECT-mode port on the PSRAM framebuffer */
#include "fb_layout.h"               /* framebuffer layout selector: PSRAM (legacy) vs 100%-SRAM ref */
#include "fb_sram.h"                 /* on-chip SRAM FB bank MPU/RISAF bring-up (no-op unless FB_IN_SRAM) */
#include "lvgl_scene.h"              /* C2 test scene */
// #include "stm32n6xx_hal_bsec.h"
// #include "stm32n6xx_hal_ramcfg.h"
// #include "llm_fsbl.h"
// #include "llm_test_fsbl.h"
// #include "npu_init.h"
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

CACHEAXI_HandleTypeDef hcacheaxi;

RAMCFG_HandleTypeDef hramcfg_SRAM1;
RAMCFG_HandleTypeDef hramcfg_SRAM2;
RAMCFG_HandleTypeDef hramcfg_SRAM3;
RAMCFG_HandleTypeDef hramcfg_SRAM4;
RAMCFG_HandleTypeDef hramcfg_SRAM5;
RAMCFG_HandleTypeDef hramcfg_SRAM6;

UART_HandleTypeDef huart1;

XSPI_HandleTypeDef hxspi1;
XSPI_HandleTypeDef hxspi2;

/* USER CODE BEGIN PV */
COM_InitTypeDef BspCOMInit;

volatile int debugFlag = 0;  /* 1 = spin in catch loop for MCP GDB attach + release (methodology). */
volatile uint32_t g_boot_stage = 0;  /* init progress marker — read via GDB to localize a stall/error */
volatile int      g_psram_rc   = 99; /* PSRAM_Init() result: 0=mapped+writable, <0=fail, 99=not run */
volatile int      g_lvgl_ok    = 0;  /* 1 once lvgl_port_n6_init succeeded */
volatile uint32_t g_heartbeat  = 0;  /* SW heartbeat counter (bare-metal liveness; HW LED mirrors it) */
/* Display-clean experiment (2026-06-30): 1 = suppress the per-token CPU<->NPU UART dialog
 * (NPU_LogStep) so a rule query is "just infer + display". The dialog is a blocking colour UART
 * line PER TOKEN inside the token loop, and lv_timer_handler never runs during that loop — so the
 * logging only lengthens the window the panel spends in the gate-toggling flicker state. Silencing
 * it shortens each inference cycle (less flicker, less per-cycle degradation), matching the
 * reference's minimal-work-per-frame pattern. Set to 0 to restore the full oracle dialog.
 * TESTED ON DEVICE 2026-06-30: does NOT fix the glitch — it still happens with quiet=1. Confirms
 * the glitch is the intrinsic per-EPOCH NPU<->LTDC AXI contention, independent of UART/CPU churn
 * duration. The real fix is the reference's dirty-region update pattern + D-cache (not this). Kept
 * on (=1) at the user's request. */
volatile int      g_npu_quiet  = 1;

/* Per-epoch LTDC display gate. 0 = gate OFF (default on option-B): the LTDC keeps scanning the
 * framebuffer during inference. 1 = gate ON: Layer1 fetch is disabled (lvgl_port_n6_display_freeze)
 * around the whole NPU inference so the LTDC stops contending for the shared AXI bus.
 *
 * HISTORY / WHY THE DEFAULT IS NOW OFF:
 *  - On the LEGACY PSRAM framebuffer (FB_IN_SRAM=0), gate OFF caused catastrophic whole-screen scanline
 *    corruption every inference — the NPU's two 64-bit AXI masters starved the LTDC's slow PSRAM
 *    scanout (invisible to SWD; only a camera saw it). There the gate was REQUIRED.
 *  - Option-B (FB_IN_SRAM=1) moves the double-buffer into dedicated NPU-free AXISRAM banks (front
 *    AXISRAM1, back AXISRAM5-6) — this IS the "AXISRAM scanout buffer" reference fix. With the FB on
 *    its own on-chip AXI slave ports + D-cache on, the LTDC scans cleanly THROUGH the inference:
 *    camera A/B (2026-06, "bye flipper") showed ZERO glitch with the gate OFF and a clean calculator UI.
 *  - With the gate ON, display_freeze() disables Layer1 for the ENTIRE stai_network_run(). That is
 *    imperceptible when inference is ~ms (all-HW epochs) but the current network has SW-fallback conv
 *    epochs that take ~2 s, so the gate blanks the panel to the bg tint for ~2 s every cycle — the
 *    reported "screen goes blank for a lot of heartbeats then comes back". OFF removes that blank.
 *
 * Kept as a runtime A/B switch: set it to 1 via GDB write_memory (D-cache-coherent through the MCP
 * write) to restore the old freeze-based gate on-device with no reflash. See MEMORY_MAP.md § Display. */
volatile int      g_npu_gate   = 0;

/* NOTE: interconnect QoS was tested as a config-only alternative to the gate and DISPROVEN on device
 * (2026-07-01, IPEVO camera): lowering the NPU's NPUNIC read QoS (SYSCFG->NPUNICQOSCR NPU1/NPU2
 * ARQOSR) to 2 and even 0 did NOT stop the whole-screen scanline corruption during inference. The
 * contention is bandwidth saturation, not priority arbitration — the NPU's sustained AXI flood
 * consumes the interconnect regardless of QoS rank. The per-epoch gate above remains the solution. */

/* NPU model (TCN, STAI). Opaque context buffer — sized by the generated header, 8-aligned. */
__attribute__((aligned(STAI_NETWORK_CONTEXT_ALIGNMENT)))
static uint8_t s_network_ctx[STAI_NETWORK_CONTEXT_SIZE];
static stai_network *g_network = NULL;
/* The Neural-ART HW network bakes its activations in its memory pool (ACTIVATIONS_NUM=0) —
 * no app-provided activation buffer needed. */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_XSPI1_Init(void);
static void MX_XSPI2_Init(void);
static void MX_BSEC_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_RAMCFG_Init(void);
static void MX_ICACHE_Init(void);
static void MX_CACHEAXI_Init(void);
static void SystemIsolation_Config(void);
/* USER CODE BEGIN PFP */
static void Enable_NPU_RAM_ForCore(void);
static void Enable_AXICACHE_RAM_ForCore(void);
static void OpenDebug(void);
extern void TerminalLogger_SmokeTest(void);  /* C++ TerminalLogger (terminal_logger.cpp), extern "C" */
extern void Cpp_STL_SmokeTest(void);         /* C++ STL infra test (cpp_runtime.cpp), extern "C" */
extern void NPU_SetVerbose(int on);          /* per-token NPU dialog toggle (grammar_runner.cpp) */
// void LLM_Repl_Run(void);   /* interactive NPU grammar REPL (llm_repl.cpp) */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Direct-register USART1 TX — HAL_UART_Transmit fails silently in the FSBL boot context
 * (HAL_GetTick may not advance), per reference N6_EDGEAI_1. Poll ISR.TXFNF (bit 7), write TDR. */
// static void uart1_putc(char c)
// {
//     while ((USART1->ISR & (1u << 7)) == 0) { __NOP(); }
//     USART1->TDR = (uint32_t)(uint8_t)c;
// }
// static void uart_puts(const char *s)
// {
//     if (USART1->CR1 == 0U) return;   /* UART not initialised yet */
//     for (; *s; ++s) uart1_putc(*s);
// }

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* RAM-bank power-on for the M55 core — BEFORE any code touches AXISRAM3-6 / CACHEAXIRAM
   * (boot ROM parks them; first access faults). Ref project N6_EDGEAI_1 main(). */
  Enable_NPU_RAM_ForCore();
  Enable_AXICACHE_RAM_ForCore();

  /* Re-authorize Secure SWD core debug (runtime BSEC, no OTP) so the MCP GDB session can
   * attach to the running FSBL. Ref N6_EDGEAI_1. REMOVE for production. */
  OpenDebug();

  /* GDB catch loop (debugFlag technique — methodology-debug-fsbl-direct + ref N6_EDGEAI_1):
   * CPU spins here after OpenDebug. MCP flow: gdb_server_start(attach) -> connect -> CPU is
   * in this loop -> release via write M[&debugFlag]=0 + DCIMVAC + resume -> step. */
  while (debugFlag) { __NOP(); }

  /* Force-disable MPU first: BootROM/leftover FSBL state may leave XN regions that fault
   * instruction fetch in our load range. Ref N6_EDGEAI_1. */
  HAL_MPU_Disable();

  /* Suppress the BootROM "unsigned-boot" RED LED (LD2 = PG10, active-low): under SWD
   * load_and_run the ROM drives PG10 LOW (LED ON) as a debug signal; a BSRR set-bit flips
   * it HIGH (OFF). Ref N6_EDGEAI_1. */
  __HAL_RCC_GPIOG_CLK_ENABLE();
  GPIOG->BSRR = (1U << 10);

  /* Clear stale pending IRQs the boot ROM / prior boot leave — they re-fire right after
   * load_and_run and fault into uninitialised HAL state. Ref N6_EDGEAI_1. */
  HAL_NVIC_DisableIRQ(TIM2_IRQn);
  HAL_NVIC_ClearPendingIRQ(TIM2_IRQn);
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* NPU RAMs were powered in USER CODE 1; enable the M55 caches. */
  // SystemInit_POST();
  g_boot_stage = 3;
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_XSPI1_Init();
  MX_XSPI2_Init();
  MX_BSEC_Init();
  MX_USART1_UART_Init();
  MX_RAMCFG_Init();
  MX_ICACHE_Init();
  MX_CACHEAXI_Init();
  SystemIsolation_Config();
  MX_EXTMEM_MANAGER_Init();
  /* USER CODE BEGIN 2 */
  // NPU_Config();
  g_boot_stage = 4;

  /* AI weights — dev=0/dev=1 UNIFIED load from flash to SRAM (not baked, not XIP). The HW network
   * reads its weights from AXISRAM1 @0x34064000; they are flashed ONCE to XSPI2 NOR @0x70200000
   * (separate from the FSBL image, so they are present at dev=0 — where the baked .ai_weights blob
   * is NOT carried by the signed image). MX_EXTMEM_MANAGER_Init (above) set up the NOR; ensure it is
   * memory-mapped, copy the weights into AXISRAM, then clean the M55 D-cache so the Neural-ART reads
   * fresh weights over its AXI master. SIZE = network_weights.bin (526496 B) — re-flash 0x70200000
   * + update this size if the model changes. */
  (void)EXTMEM_MemoryMappedMode(EXTMEMORY_1, EXTMEM_ENABLE);            /* NOR memory-mapped for XIP */
  /* OPTION B (FB_IN_SRAM): the regenerated network reads its weights + epoch-controller blobs XIP
   * from NOR @0x70000000 (ecloader) — NO copy to AXISRAM1, which is now the FB front bank. The
   * legacy `memcpy(0x34064000 <- 0x70200000, 526496)` + D-cache clean are removed. Flash the
   * regenerated NOR blob (network_regen/generated/atonbuf_xSPI2.bin) to 0x70000000. */
  g_boot_stage = 5;

  // RISAF_Config();
  g_boot_stage = 6;

    /* App logic: bring up the NPU calculator model + run one inference. Traced via MCP
   * (g_boot_stage + step). The REPL (UART I/O) is deferred per focus. */
  /* LLM_FSBL_Init();  ...one inference... */
  g_boot_stage = 7;

  /* LEDS */
  {
    (void)BSP_LED_Init(0U);   /* LED_GREEN = LED1 */
    BSP_LED_Off(0U);
    BSP_LED_On (0U);

    (void)BSP_LED_Init(1U);   /* LED_RED = LED2 — error indicator: OFF at init, ON only on error */
    BSP_LED_Off(1U);
  }

  /* COM TERMINAL */
  {
    /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
    BspCOMInit.BaudRate   = 115200;
    BspCOMInit.WordLength = COM_WORDLENGTH_8B;
    BspCOMInit.StopBits   = COM_STOPBITS_1;
    BspCOMInit.Parity     = COM_PARITY_NONE;
    BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
    if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
    {
      Error_Handler();
    }

    printf( "\x1B[2J" );
    printf("%c[0;0H", 0x1b);
    printf("MAIN APP ON\n");
  }

  /* NPU */
  {
    /* NPU peripheral bring-up: clock + reset. Banks were powered in USER CODE 1;
     * RIF master/slave granted in SystemIsolation_Config; CACHEAXI by MX_CACHEAXI_Init.
     * No NPU register access in code yet (smoke-read via MCP). Ref N6_EDGEAI_1. */
    __HAL_RCC_NPU_CLK_ENABLE();
    __HAL_RCC_NPU_FORCE_RESET();
    __HAL_RCC_NPU_RELEASE_RESET();
    /* The RCC/RIF macros are void (no status). Verify the result instead: if the NPU
     * clock did not engage, printf the detail + trigger the error, per the rule. */
    if (__HAL_RCC_NPU_IS_CLK_ENABLED() == 0U)
    {
      printf("NPU clock enable FAILED (RCC AHB5ENR) - boot_stage=%lu\r\n",
             (unsigned long)g_boot_stage);
      Error_Handler();
    }
    g_boot_stage = 8;

    /* NPU performance: keep the RAM/NPU/cache clocks alive in low-power/sleep (the NPU
     * runs async to the M55) + enable the AXI cache so the NPU's weight/activation
     * accesses are cached — without this the NPU runs at software-fallback speed.
     * Mirrors NPU_Validation set_clk_sleep_mode() + npu_cache_enable(). */
    __HAL_RCC_AXISRAM1_MEM_CLK_SLEEP_ENABLE();   __HAL_RCC_AXISRAM2_MEM_CLK_SLEEP_ENABLE();
    __HAL_RCC_AXISRAM3_MEM_CLK_SLEEP_ENABLE();   __HAL_RCC_AXISRAM4_MEM_CLK_SLEEP_ENABLE();
    __HAL_RCC_AXISRAM5_MEM_CLK_SLEEP_ENABLE();   __HAL_RCC_AXISRAM6_MEM_CLK_SLEEP_ENABLE();
    __HAL_RCC_CACHEAXIRAM_MEM_CLK_SLEEP_ENABLE();
    __HAL_RCC_CACHEAXI_CLK_SLEEP_ENABLE();        __HAL_RCC_NPU_CLK_SLEEP_ENABLE();
    { extern void npu_cache_enable(void); npu_cache_enable(); }  /* HAL_CACHEAXI_Enable */

    /* Grant all CIDs RISAF R/W on the NPU memory (AXISRAM weights @0x34064000 +
     * activations) + the NPU masters + NPU cache — without this the NPU epoch stalls
     * reading the weights. Mirrors NPU_Validation set_risaf_default. */
    {
      RISAF_TypeDef *const npu_risafs[] = { RISAF2_S, RISAF4_S, RISAF5_S, RISAF6_S, RISAF8_S };
      RISAF_BaseRegionConfig_t rc;
      memset(&rc, 0, sizeof(rc));
      rc.StartAddress   = 0x0;
      rc.EndAddress     = 0x0FFFFFFF;
      rc.Filtering      = RISAF_FILTER_ENABLE;
      rc.PrivWhitelist  = RIF_CID_NONE;
      rc.ReadWhitelist  = RIF_CID_MASK;
      rc.WriteWhitelist = RIF_CID_MASK;
      for (unsigned i = 0; i < sizeof(npu_risafs) / sizeof(npu_risafs[0]); i++)
      {
        rc.Secure = RIF_ATTRIBUTE_SEC;
        HAL_RIF_RISAF_ConfigBaseRegion(npu_risafs[i], RISAF_REGION_1, &rc);
        rc.Secure = RIF_ATTRIBUTE_NSEC;
        HAL_RIF_RISAF_ConfigBaseRegion(npu_risafs[i], RISAF_REGION_2, &rc);
      }
    }

    printf("NPU clocked + reset OK + AXI cache + RISAF mem (boot_stage=%lu)\r\n", (unsigned long)g_boot_stage);
  }

  /* NPU RUNTIME (global ATON runtime init — MUST run before any network init/run).
   * The --st-neural-art network runs as epoch *blobs* on the NPU epoch controller,
   * whose host sync is interrupt-based (RM0486 §20.3.15-16) and supported ONLY in
   * LL_ATON_RT_ASYNC mode (polling asserts out — ll_aton_runtime.c:132). The runtime
   * init is gated behind stai_runtime_init() (NOT stai_network_init): it enables the
   * ATON interrupt controller and INSTALL+NVIC_EnableIRQ(NPU0_IRQn) so the epoch-end
   * IRQ wakes the runtime's __WFE(). Skipping it = silent WFE stall (init OK, run never
   * returns) — the earlier polling-mode bug. Build sets -DLL_ATON_RT_MODE=ASYNC. */
  {
    stai_return_code rrc = stai_runtime_init();
    if (rrc != STAI_SUCCESS)
    {
      printf("NPU runtime init FAILED (stai rc=%d)\r\n", (int)rrc);
      Error_Handler();
    }
    printf("NPU runtime init OK (ATON INTCTRL + NPU0 IRQ enabled, ISER1=0x%08lX)\r\n",
           (unsigned long)NVIC->ISER[1]);
  }

  /* NPU MODEL */
  {
    /* Initialize the TCN model context (STAI). Inputs/outputs/weights are preallocated
     * in network.c / network_data.c and placed by the custom linker. */
    g_network = (stai_network *)s_network_ctx;
    stai_return_code rc = stai_network_init(g_network);
    if (rc != STAI_SUCCESS)
    {
      printf("NPU model init FAILED (stai rc=%d) - boot_stage=%lu\r\n",
             (int)rc, (unsigned long)g_boot_stage);
      Error_Handler();
    }
    g_boot_stage = 9;
    printf("NPU HW model init OK (ctx=%u B, weights @0x34064000)\r\n",
           (unsigned)STAI_NETWORK_CONTEXT_SIZE);
  }

  /* NPU INFERENCE — boot self-test: prove the Neural-ART path end-to-end on-device. */
  {
    stai_ptr in_buf = NULL, out_buf = NULL;
    stai_size n_in = 0, n_out = 0;
    stai_network_get_inputs(g_network, &in_buf, &n_in);
    stai_network_get_outputs(g_network, &out_buf, &n_out);
    if (in_buf == NULL || out_buf == NULL)
    {
      printf("NPU get in/out FAILED\r\n");
      Error_Handler();
    }

    g_boot_stage = 10;
    {
      const char *expr = "3 + 4";
      int ok = 0;
      long res = Grammar_Calc(g_network, (int8_t *)in_buf, (const int8_t *)out_buf, expr, &ok);
      if (ok) printf("\r\nNPU self-test: %s = %ld\r\n", expr, res);
      else    printf("\r\nNPU self-test: %s -> parse failed\r\n", expr);
    }
  }

  /* PSRAM status — mapped early in MX_XSPI1_Init (before the NOR/EXTMEM setup, so the NOR survives
   * the XSPI-manager reset). The NPU demo below is the stability gate: if it still computes (no red
   * LED), the early PSRAM map left XSPI2/NOR + the weights intact. */
  printf("\r\nPSRAM @0x90000000: %s (rc=%d)\r\n",
         (g_psram_rc == 0) ? "mapped + CPU-writable" : "FAILED", g_psram_rc);
  /* PSRAM_Mpu() — mark 0x90000000 Normal Non-Cacheable: the documented fix for the cacheable-FB
   * corruption (the framebuffer goes to garbage/all-red as the CPU + printf churn the D-cache; the
   * LTDC reads PSRAM directly and sees the cache-stale content). The earlier red-LED was from
   * flashing onto a wedged device (MCP reset is ineffective); retried here from a clean programmer
   * -hardRst. */
  if (g_psram_rc == 0) {
    PSRAM_Mpu();
  }

  /* Reference low-bus-traffic mechanical (flicker fix — stage 1): now that PSRAM_Mpu() has marked the
   * framebuffer window Normal-Non-Cacheable in an ENABLED MPU, turn the M55 L1 caches ON. This is the
   * lever that lets the reference (N6_EDGEAI_1) run the NPU concurrently glitch-free on the same
   * silicon: with D-cache on, the CPU working set (embedding Gather, token loop, LVGL heap in
   * cacheable AXISRAM1) is absorbed by cache instead of flooding the AXI interconnect that the LTDC
   * scans the framebuffer from — the root cause of the inference-time scanout starvation. Ordering is
   * critical: the FB MUST already be non-cacheable (above) so LVGL's pixel writes never sit in cache
   * for the LTDC to read stale; flush_cb's __DSB drains them to PSRAM before each swap. The NPU
   * activation arena stays write-back, and the STAI runtime's mcu_cache_* hooks (self-gated on
   * SCB->CCR.DC) now auto-invalidate the network output each epoch; the CPU-filled input is cleaned
   * explicitly in npu_query.c. SCB_EnableDCache invalidates the whole D-cache before enabling, so
   * prior uncached writes (weights memcpy, BSS) are already in memory — no stale lines. */
  SCB_EnableICache();
  SCB_EnableDCache();

  /* Reference layout (FB_IN_SRAM): mark the on-chip FB banks (AXISRAM1 front + AXISRAM5-6 back)
   * MPU Normal-Non-Cacheable so the LTDC scans fresh pixels with D-cache on. No-op for the legacy
   * PSRAM layout (PSRAM_Mpu already covers 0x90000000). Must run before the FB is first written. */
  fb_sram_enable();

  /* LCD: LTDC + Layer-1 framebuffer up (paints a brief R/G/B pattern). */
  LCD_Init();

  /* LVGL (DIRECT mode) renders straight into the same PSRAM framebuffer the LTDC scans — it repaints
   * the whole frame continuously, so it overwrites/absorbs the static-FB corruption. */
  if (g_psram_rc == 0) {
    /* DWT cycle counter = lvgl_port_n6's ms tick source. */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    lvgl_port_n6_cfg_t lv_cfg = {
      .fb_addr         = (void *)FB_FRONT_ADDR,   /* front — PSRAM (legacy) or AXISRAM1 (FB_IN_SRAM) */
      .fb_back_addr    = (void *)FB_BACK_ADDR,    /* back  — PSRAM+1MB (legacy) or AXISRAM5-6 (FB_IN_SRAM) */
      .fb_width        = FB_WIDTH,
      .fb_height       = FB_HEIGHT,
      .fb_bytes_per_px = FB_BPP,
      .hclk_hz         = 800000000U,   /* run-23 HCLK (DWT->ms); tweak if clock differs */
      .build_scene_cb  = lvgl_scene_build,
    };
    g_lvgl_ok = (lvgl_port_n6_init(&lv_cfg) == LVGL_PORT_N6_OK);
    printf("LVGL: init %s (DIRECT @0x90000000)\r\n", g_lvgl_ok ? "ok" : "FAILED");
  }

  /* Autonomous on-chip calculator over the ST-Link VCP (USART1 / huart1). The device does
   * ALL the work — this proves edge autonomy. The host unified runner in `--mode device` is a
   * thin terminal: it pushes an expression and collects the result; the tokenize, parse, evaluate
   * and NPU grammar recall all run HERE on the Cortex-M55 + Neural-ART. Connect directly with
   *   minicom -D /dev/ttyACM0 -b 115200
   * or via the runner (scripts/classes/class_model_runner.py --mode device). */
  {
    stai_ptr rin = NULL, rout = NULL; stai_size rn = 0;
    stai_network_get_inputs(g_network, &rin, &rn);
    stai_network_get_outputs(g_network, &rout, &rn);
    NPU_SetVerbose(0);   /* concise: per-rule [model #N] dialog, no per-token spam */

    /* Per-grammar demo set: the CALCULATOR grammar's test expressions. Each grammar carries its own
     * demo/test inputs; the `/demo` runner command cycles the active grammar's set. (run-23 carries
     * one grammar today; a future grammar would ship its own set.) */
    static const char *const calc_demo[] = { "3 + 4", "12 - 5", "6 * 7", "8 / 2", "9 * 9", "2 * (3 + 4)" };
    const int ndemo = (int)(sizeof(calc_demo) / sizeof(calc_demo[0]));

    /* Interactive CLI over the VCP (minicom -D /dev/ttyACM0 -b 115200), line-buffered, with a built-in
     * `/demo` runner command (orchestration — NOT a nocode grammar; the carried *logic* is the
     * dispatch). Accepts any phrasing: "/demo", "/demo off", "/demo calculator on", "/calculator demo
     * off" — a line containing "demo" toggles the loop (off when it also contains "off"). Anything
     * else is evaluated as an expression on-chip via the nocode dispatch + NPU. Heartbeat (GREEN LED
     * + g_heartbeat) ticks throughout; RED lights only on an error. Demo runs by default at boot. */
    uint32_t hb_last = HAL_GetTick(), demo_last = 0U, fb_last = 0U;
    int di = 0, demo_on = 1;
    char cmd[48]; int cmdn = 0;
    printf("\r\n=== NPU calculator — demo running.  /demo off  ·  /demo on  ·  or type an expression ===\r\n");
    for (;;)
    {
      uint32_t now = HAL_GetTick();
      if (now - hb_last >= 500U) { hb_last = now; BSP_LED_Toggle(LED_GREEN); g_heartbeat++; }  /* heartbeat */
      /* Cap LVGL rendering to ~30 FPS (reference cadence). Rendering the super-loop uncapped repaints far
       * faster than the 60 Hz LTDC can swap, so the LTDC scans a buffer mid-render -> tearing/distortion
       * ("stretched"). 33 ms matches the reference's ThreadX 30 FPS render task. */
      if (g_lvgl_ok && (now - fb_last >= 33U)) { fb_last = now; lvgl_scene_tick(lvgl_port_n6_loop_count, g_heartbeat); lvgl_port_n6_run_once(); }  /* LVGL frame @30 FPS */

      uint8_t ch;                              /* line-buffered command/expression input (non-blocking) */
      if (huart1.Instance->ISR & USART_ISR_ORE) { huart1.Instance->ICR = USART_ICR_ORECF; }  /* clear overrun: the super-loop stalls ~ms during NPU inference so a byte can be missed; without this the ORE flag wedges HAL RX and the UART goes dead ("cannot write to it") */
      if (huart1.Instance->ISR & USART_ISR_RXNE_RXFNE) {
        ch = (uint8_t)(huart1.Instance->RDR & 0xFFU);   /* read straight from the UART; reading RDR clears the RX flag */
        if (ch == '\r' || ch == '\n') {
          cmd[cmdn] = '\0';
          if (cmdn > 0) {
            if (strstr(cmd, "demo") != NULL) {   /* /demo [grammar] [on|off] — orchestration command */
              demo_on  = (strstr(cmd, "off") == NULL);
              demo_last = 0U;
              printf("\r\n[demo %s]\r\n", demo_on ? "on" : "off");
            } else {                             /* evaluate as an expression (the calculator grammar) */
              int ok = 0;
              if (g_lvgl_ok) { lvgl_scene_set_prompt(cmd); lvgl_port_n6_run_once(); }   /* show the prompt */
              long res = Grammar_Calc(g_network, (int8_t *)rin, (const int8_t *)rout, cmd, &ok);   /* gates LTDC fetch per NPU epoch internally */
              if (g_lvgl_ok) { char ab[40]; if (ok) snprintf(ab, sizeof ab, "= %ld", res); else snprintf(ab, sizeof ab, "parse error"); lvgl_scene_set_answer(ab); lvgl_port_n6_run_once(); }   /* show the answer */
              if (ok) printf("\r\n= %ld\r\n", res);
              else  { printf("\r\nparse failed for \"%s\"\r\n", cmd); BSP_LED_On(LED_RED); }
            }
          }
          cmdn = 0;
        } else if (ch == 0x08 || ch == 0x7F) { if (cmdn > 0) { --cmdn; printf("\b \b"); } }
        else if (cmdn < (int)sizeof(cmd) - 1) { cmd[cmdn++] = (char)ch; HAL_UART_Transmit(&huart1, &ch, 1, 100); }
      }

      if (demo_on && (now - demo_last >= 3000U)) {   /* demo cadence: next expression every 3 s (circular) — leaves ~2 s to read the answer before the next parse */
        demo_last = now;
        const char *expr = calc_demo[di]; di = (di + 1) % ndemo;
        int ok = 0;
        /* Option-B display path: the LTDC double-buffer lives in NPU-free AXISRAM (front AXISRAM1, back
         * AXISRAM5-6), so the LTDC scans cleanly while the NPU runs — no per-epoch LTDC gate needed
         * (g_npu_gate defaults 0). set_prompt shows the new expression and clears the answer to "= ?";
         * after inference set_answer paints the result. Both are dirty-region updates (no full-screen
         * invalidate), so only the changed labels repaint. */
        /* Render TWICE after each label change: the mark_dirty full-redraw propagates over 2 render
         * passes, and the ~1 s inference blocks the super-loop right after — so drain both passes now
         * to land "expr = ?" in BOTH DIRECT buffers before the block, else the buffer the LTDC lands on
         * during inference is stale (blank). */
        if (g_lvgl_ok) { lvgl_scene_set_prompt(expr); lvgl_port_n6_run_once(); lvgl_port_n6_run_once(); }   /* show the prompt + "= ?" (no NPU yet) */
        long res = Grammar_Calc(g_network, (int8_t *)rin, (const int8_t *)rout, expr, &ok);   /* gates the LTDC fetch per NPU epoch internally */
        if (g_lvgl_ok) { char ab[40]; if (ok) snprintf(ab, sizeof ab, "= %ld", res); else snprintf(ab, sizeof ab, "ERROR"); lvgl_scene_set_answer(ab); lvgl_port_n6_run_once(); lvgl_port_n6_run_once(); }   /* show the answer in both buffers */
        if (ok) printf("[hb %lu] %s = %ld\r\n", (unsigned long)g_heartbeat, expr, res);
        else  { printf("[hb %lu] %s -> ERROR\r\n", (unsigned long)g_heartbeat, expr); BSP_LED_On(LED_RED); }
      }
    }
  }
  /* USER CODE END 2 */

  /* Launch the application */
  if (BOOT_OK != BOOT_Application())
  {
    Error_Handler();
  }
  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}
/* USER CODE BEGIN CLK 1 */
/* USER CODE END CLK 1 */

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the System Power Supply
  */
  if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE0) != HAL_OK)
  {
    Error_Handler();
  }

  /* Enable HSI */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Get current CPU/System buses clocks configuration and if necessary switch
 to intermediate HSI clock to ensure target clock can be set
  */
  HAL_RCC_GetClockConfig(&RCC_ClkInitStruct);
  if ((RCC_ClkInitStruct.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
     (RCC_ClkInitStruct.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11))
  {
    RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK);
    RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
    {
      /* Initialization Error */
      Error_Handler();
    }
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL1.PLLM = 2;
  RCC_OscInitStruct.PLL1.PLLN = 75;
  RCC_OscInitStruct.PLL1.PLLFractional = 0;
  RCC_OscInitStruct.PLL1.PLLP1 = 1;
  RCC_OscInitStruct.PLL1.PLLP2 = 1;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL2.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL2.PLLM = 8;
  RCC_OscInitStruct.PLL2.PLLN = 125;
  RCC_OscInitStruct.PLL2.PLLFractional = 0;
  RCC_OscInitStruct.PLL2.PLLP1 = 1;
  RCC_OscInitStruct.PLL2.PLLP2 = 1;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL3.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL3.PLLM = 8;
  RCC_OscInitStruct.PLL3.PLLN = 225;
  RCC_OscInitStruct.PLL3.PLLFractional = 0;
  RCC_OscInitStruct.PLL3.PLLP1 = 2;
  RCC_OscInitStruct.PLL3.PLLP2 = 1;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_HCLK
                              |RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1
                              |RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_PCLK5
                              |RCC_CLOCKTYPE_PCLK4;
  RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;
  RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC1Selection.ClockDivider = 3;
  RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC2Selection.ClockDivider = 6;
  RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL2;
  RCC_ClkInitStruct.IC6Selection.ClockDivider = 1;
  RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL3;
  RCC_ClkInitStruct.IC11Selection.ClockDivider = 1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief BSEC Initialization Function
  * @param None
  * @retval None
  */
static void MX_BSEC_Init(void)
{

  /* USER CODE BEGIN BSEC_Init 0 */

  /* USER CODE END BSEC_Init 0 */

  /* USER CODE BEGIN BSEC_Init 1 */

  /* USER CODE END BSEC_Init 1 */
  /* USER CODE BEGIN BSEC_Init 2 */

  /* USER CODE END BSEC_Init 2 */

}

/**
  * @brief CACHEAXI Initialization Function
  * @param None
  * @retval None
  */
static void MX_CACHEAXI_Init(void)
{

  /* USER CODE BEGIN CACHEAXI_Init 0 */

  /* USER CODE END CACHEAXI_Init 0 */

  /* USER CODE BEGIN CACHEAXI_Init 1 */

  /* USER CODE END CACHEAXI_Init 1 */
  hcacheaxi.Instance = CACHEAXI;
  if (HAL_CACHEAXI_Init(&hcacheaxi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CACHEAXI_Init 2 */

  /* USER CODE END CACHEAXI_Init 2 */

}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief RAMCFG Initialization Function
  * @param None
  * @retval None
  */
static void MX_RAMCFG_Init(void)
{

  /* USER CODE BEGIN RAMCFG_Init 0 */

  /* USER CODE END RAMCFG_Init 0 */

  /* USER CODE BEGIN RAMCFG_Init 1 */

  /* USER CODE END RAMCFG_Init 1 */

  /** Initialize RAMCFG SRAM1
  */
  hramcfg_SRAM1.Instance = RAMCFG_SRAM1_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM2
  */
  hramcfg_SRAM2.Instance = RAMCFG_SRAM2_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM3
  */
  hramcfg_SRAM3.Instance = RAMCFG_SRAM3_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM4
  */
  hramcfg_SRAM4.Instance = RAMCFG_SRAM4_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM5
  */
  hramcfg_SRAM5.Instance = RAMCFG_SRAM5_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMCFG SRAM6
  */
  hramcfg_SRAM6.Instance = RAMCFG_SRAM6_AXI;
  if (HAL_RAMCFG_Init(&hramcfg_SRAM6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RAMCFG_Init 2 */

  /* USER CODE END RAMCFG_Init 2 */

}

/**
  * @brief RIF Initialization Function
  * @param None
  * @retval None
  */
  static void SystemIsolation_Config(void)
{

  /* USER CODE BEGIN RIF_Init 0 */

  /* USER CODE END RIF_Init 0 */

  /* set all required IPs as secure privileged */
  __HAL_RCC_RIFSC_CLK_ENABLE();

  /*RIMC configuration*/
  RIMC_MasterConfig_t RIMC_master = {0};
  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_ETH1, &RIMC_master);

  /* LTDC masters (display) — MUST be SEC+PRIV to read the PSRAM framebuffer. The shared RIMC_master
   * above is SEC|NPRIV; an NPRIV LTDC master's DMA reads return ZEROS -> the layer renders BLACK
   * (reference, verified). Use a dedicated SEC|PRIV master config for LTDC1/LTDC2. */
  RIMC_MasterConfig_t ltdc_master = {0};
  ltdc_master.MasterCID = RIF_CID_1;
  ltdc_master.SecPriv   = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC1, &ltdc_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC2, &ltdc_master);
  /* LTDC slaves (RISC) — SEC+PRIV (matches the master). With SEC master / NSEC slave the RISAF
   * blocks the LTDC DMA reads of the PSRAM framebuffer and the panel stays black. */
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDC,   RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL1, RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL2, RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);

  RIMC_master.MasterCID = RIF_CID_0;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_SDMMC2, &RIMC_master);

  /*RISUP configuration*/
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);

  /* RIF-Aware IPs Config */

  /* set up PWR configuration */
  HAL_PWR_ConfigAttributes(PWR_ITEM_WKUP1,PWR_SEC_NPRIV);

  /* set up GPIO configuration */
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOF,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);

  /* USER CODE BEGIN RIF_Init 1 */
  /* USART1 (BSP COM1 VCP) — complete the peripheral security like the rest of the FSBL:
   * mark it secure-accessible. NPRIV (privileged FSBL access still allowed), consistent
   * with its PE5/PE6 pins (SEC|NPRIV). Gentle — does not restrict the working access. */
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_USART1, RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV);
  /* NPU master — grant the NPU's AXI master secure+privileged access so its weight/
   * activation transactions are not RIF-blocked (the NPU slave RISC is granted by MX
   * above). Ref N6_EDGEAI_1. RIMC_master is in scope (from the generated config). */
  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv   = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &RIMC_master);
  /* USER CODE END RIF_Init 1 */
  /* USER CODE BEGIN RIF_Init 2 */

  /* USER CODE END RIF_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief XSPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_XSPI1_Init(void)
{

  /* USER CODE BEGIN XSPI1_Init 0 */

  /* USER CODE END XSPI1_Init 0 */

  XSPIM_CfgTypeDef sXspiManagerCfg = {0};

  /* USER CODE BEGIN XSPI1_Init 1 */

  /* USER CODE END XSPI1_Init 1 */
  /* XSPI1 parameter configuration*/
  hxspi1.Instance = XSPI1;
  hxspi1.Init.FifoThresholdByte = 1;
  hxspi1.Init.MemoryMode = HAL_XSPI_SINGLE_MEM;
  hxspi1.Init.MemoryType = HAL_XSPI_MEMTYPE_MICRON;
  hxspi1.Init.MemorySize = HAL_XSPI_SIZE_16B;
  hxspi1.Init.ChipSelectHighTimeCycle = 1;
  hxspi1.Init.FreeRunningClock = HAL_XSPI_FREERUNCLK_DISABLE;
  hxspi1.Init.ClockMode = HAL_XSPI_CLOCK_MODE_0;
  hxspi1.Init.WrapSize = HAL_XSPI_WRAP_NOT_SUPPORTED;
  hxspi1.Init.ClockPrescaler = 0;
  hxspi1.Init.SampleShifting = HAL_XSPI_SAMPLE_SHIFT_NONE;
  hxspi1.Init.DelayHoldQuarterCycle = HAL_XSPI_DHQC_DISABLE;
  hxspi1.Init.ChipSelectBoundary = HAL_XSPI_BONDARYOF_NONE;
  hxspi1.Init.MaxTran = 0;
  hxspi1.Init.Refresh = 0;
  hxspi1.Init.MemorySelect = HAL_XSPI_CSSEL_NCS1;
  if (HAL_XSPI_Init(&hxspi1) != HAL_OK)
  {
    Error_Handler();
  }
  sXspiManagerCfg.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
  sXspiManagerCfg.IOPort = HAL_XSPIM_IOPORT_1;
  sXspiManagerCfg.Req2AckTime = 1;
  if (HAL_XSPIM_Config(&hxspi1, &sXspiManagerCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN XSPI1_Init 2 */
  /* Map the external APS256 PSRAM @0x90000000 (framebuffer home) via the board BSP. Done HERE,
   * inside MX_XSPI1_Init, so BSP_XSPI_RAM_Init's __HAL_RCC_XSPIM_FORCE_RESET (which wipes the shared
   * XSPI manager) happens BEFORE MX_XSPI2_Init + the NOR/EXTMEM setup run — the NOR is configured
   * afterward and survives, the NPU weights copy from the NOR stays intact (no XSPI2 re-init dance).
   * Silent here (UART/COM not up yet); result in g_psram_rc, printed later in USER CODE 2. */
  g_psram_rc = PSRAM_Init();
  /* USER CODE END XSPI1_Init 2 */

}

/**
  * @brief XSPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_XSPI2_Init(void)
{

  /* USER CODE BEGIN XSPI2_Init 0 */

  /* USER CODE END XSPI2_Init 0 */

  XSPIM_CfgTypeDef sXspiManagerCfg = {0};

  /* USER CODE BEGIN XSPI2_Init 1 */

  /* USER CODE END XSPI2_Init 1 */
  /* XSPI2 parameter configuration*/
  hxspi2.Instance = XSPI2;
  hxspi2.Init.FifoThresholdByte = 4;
  hxspi2.Init.MemoryMode = HAL_XSPI_SINGLE_MEM;
  hxspi2.Init.MemoryType = HAL_XSPI_MEMTYPE_MACRONIX;
  hxspi2.Init.MemorySize = HAL_XSPI_SIZE_1GB;
  hxspi2.Init.ChipSelectHighTimeCycle = 1;
  hxspi2.Init.FreeRunningClock = HAL_XSPI_FREERUNCLK_DISABLE;
  hxspi2.Init.ClockMode = HAL_XSPI_CLOCK_MODE_0;
  hxspi2.Init.WrapSize = HAL_XSPI_WRAP_NOT_SUPPORTED;
  hxspi2.Init.ClockPrescaler = 0;
  hxspi2.Init.SampleShifting = HAL_XSPI_SAMPLE_SHIFT_NONE;
  hxspi2.Init.DelayHoldQuarterCycle = HAL_XSPI_DHQC_ENABLE;
  hxspi2.Init.ChipSelectBoundary = HAL_XSPI_BONDARYOF_NONE;
  hxspi2.Init.MaxTran = 0;
  hxspi2.Init.Refresh = 0;
  hxspi2.Init.MemorySelect = HAL_XSPI_CSSEL_NCS1;
  if (HAL_XSPI_Init(&hxspi2) != HAL_OK)
  {
    Error_Handler();
  }
  sXspiManagerCfg.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
  sXspiManagerCfg.IOPort = HAL_XSPIM_IOPORT_2;
  sXspiManagerCfg.Req2AckTime = 1;
  if (HAL_XSPIM_Config(&hxspi2, &sXspiManagerCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN XSPI2_Init 2 */

  /* USER CODE END XSPI2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOP_CLK_ENABLE();
  __HAL_RCC_GPIOO_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPION_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SMPS_OVD_GPIO_Port, SMPS_OVD_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : I2C1_SDA_Pin */
  GPIO_InitStruct.Pin = I2C1_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(I2C1_SDA_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : SD_D0_Pin SD_D1_Pin SD_D2_Pin SD_CMD_Pin */
  GPIO_InitStruct.Pin = SD_D0_Pin|SD_D1_Pin|SD_D2_Pin|SD_CMD_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_MEDIUM;
  GPIO_InitStruct.Alternate = GPIO_AF11_SDMMC2;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : MIC_D1_Pin PE7 MIC_CK_Pin */
  GPIO_InitStruct.Pin = MIC_D1_Pin|GPIO_PIN_7|MIC_CK_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_MDF1;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : LCD_B4_Pin LCD_B5_Pin LCD_R4_Pin */
  GPIO_InitStruct.Pin = LCD_B4_Pin|LCD_B5_Pin|LCD_R4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

  /*Configure GPIO pins : LCD_R2_Pin LCD_R7_Pin LCD_R1_Pin */
  GPIO_InitStruct.Pin = LCD_R2_Pin|LCD_R7_Pin|LCD_R1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : I2C2_SDA_Pin I2C2_SCL_Pin */
  GPIO_InitStruct.Pin = I2C2_SDA_Pin|I2C2_SCL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C2;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : LCD_HSYNC_Pin LCD_B2_Pin LCD_G4_Pin LCD_G6_Pin
                           LCD_G5_Pin LCD_R3_Pin */
  GPIO_InitStruct.Pin = LCD_HSYNC_Pin|LCD_B2_Pin|LCD_G4_Pin|LCD_G6_Pin
                          |LCD_G5_Pin|LCD_R3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : I2C1_SCL_Pin */
  GPIO_InitStruct.Pin = I2C1_SCL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(I2C1_SCL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SD_CK_Pin */
  GPIO_InitStruct.Pin = SD_CK_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_SDMMC2;
  HAL_GPIO_Init(SD_CK_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SD_D3_Pin */
  GPIO_InitStruct.Pin = SD_D3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_MEDIUM;
  GPIO_InitStruct.Alternate = GPIO_AF11_SDMMC2;
  HAL_GPIO_Init(SD_D3_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LCD_VSYNC_Pin */
  GPIO_InitStruct.Pin = LCD_VSYNC_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(LCD_VSYNC_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : SAI1_FS_A_Pin SAI1_SD_A_Pin SAI1_CLK_A_Pin */
  GPIO_InitStruct.Pin = SAI1_FS_A_Pin|SAI1_SD_A_Pin|SAI1_CLK_A_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF6_SAI1;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : User_Pin */
  GPIO_InitStruct.Pin = User_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(User_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SAI1_SD_B_Pin */
  GPIO_InitStruct.Pin = SAI1_SD_B_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF6_SAI1;
  HAL_GPIO_Init(SAI1_SD_B_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : HEXASPI_IO_15_Pin HEXASPI_IO_12_Pin HEXASPI_IO_13_Pin HEXASPI_IO_11_Pin
                           HEXASPI_IO_8_Pin HEXASPI_IO_14_Pin HEXASPI_IO_9_Pin HEXASPI_IO_10_Pin */
  GPIO_InitStruct.Pin = HEXASPI_IO_15_Pin|HEXASPI_IO_12_Pin|HEXASPI_IO_13_Pin|HEXASPI_IO_11_Pin
                          |HEXASPI_IO_8_Pin|HEXASPI_IO_14_Pin|HEXASPI_IO_9_Pin|HEXASPI_IO_10_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF9_XSPIM_P1;
  HAL_GPIO_Init(GPIOP, &GPIO_InitStruct);

  /*Configure GPIO pin : SAI1_MCLK_A_Pin */
  GPIO_InitStruct.Pin = SAI1_MCLK_A_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF6_SAI1;
  HAL_GPIO_Init(SAI1_MCLK_A_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SMPS_OVD_Pin */
  GPIO_InitStruct.Pin = SMPS_OVD_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(SMPS_OVD_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LCD_B3_Pin LCD_B0_Pin LCD_G1_Pin LCD_R0_Pin
                           LCD_G0_Pin LCd_G7_Pin LCD_DE_Pin LCD_R6_Pin */
  GPIO_InitStruct.Pin = LCD_B3_Pin|LCD_B0_Pin|LCD_G1_Pin|LCD_R0_Pin
                          |LCD_G0_Pin|LCd_G7_Pin|LCD_DE_Pin|LCD_R6_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pin : HEXASPI_DQS1_Pin */
  GPIO_InitStruct.Pin = HEXASPI_DQS1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF9_XSPIM_P1;
  HAL_GPIO_Init(HEXASPI_DQS1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LCD_G2_Pin LCD_R5_Pin LCD_B1_Pin LCD_B7_Pin
                           LCD_B6_Pin LCD_G3_Pin */
  GPIO_InitStruct.Pin = LCD_G2_Pin|LCD_R5_Pin|LCD_B1_Pin|LCD_B7_Pin
                          |LCD_B6_Pin|LCD_G3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : UCPD1_VSENSE_Pin */
  GPIO_InitStruct.Pin = UCPD1_VSENSE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(UCPD1_VSENSE_GPIO_Port, &GPIO_InitStruct);

  /*Configure the EXTI line attribute */
  HAL_EXTI_ConfigLineAttributes(EXTI_LINE_13, EXTI_LINE_SEC);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* Power on AXISRAM3-6 (NPU activations, 4×448 KB) for the M55 core.
 * Boot ROM parks these banks (SRAMSD=1); first access faults silently.
 * Must run BEFORE HAL_Init, BSS zeroing, or .data copy touches these ranges. */
static void Enable_NPU_RAM_ForCore(void)
{
    RAMCFG_HandleTypeDef hramcfg = {0};

    /* AXISRAM1 (0x34000000) — holds the NPU weights (.npu_weights section,
     * loaded via load-and-run). Boot ROM parks it; power it before use. */
    __HAL_RCC_AXISRAM1_MEM_CLK_ENABLE();
    hramcfg.Instance = RAMCFG_SRAM1_AXI; HAL_RAMCFG_EnableAXISRAM(&hramcfg);

    __HAL_RCC_AXISRAM3_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM4_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM5_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM6_MEM_CLK_ENABLE();

    hramcfg.Instance = RAMCFG_SRAM3_AXI; HAL_RAMCFG_EnableAXISRAM(&hramcfg);
    hramcfg.Instance = RAMCFG_SRAM4_AXI; HAL_RAMCFG_EnableAXISRAM(&hramcfg);
    hramcfg.Instance = RAMCFG_SRAM5_AXI; HAL_RAMCFG_EnableAXISRAM(&hramcfg);
    hramcfg.Instance = RAMCFG_SRAM6_AXI; HAL_RAMCFG_EnableAXISRAM(&hramcfg);
}

/* Power on CACHEAXIRAM (0x343C0000-0x343FFFFF) and enable CACHEAXI peripheral clock.
 * Required before any NPU AXI transaction that might route through CACHEAXI. */
static void Enable_AXICACHE_RAM_ForCore(void)
{
    __HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
}

/* Re-authorize SWD core debug + secure debug via HAL_BSEC — matches reference
 * project N6_EDGEAI_1 exactly. Runtime BSEC registers only, no OTP writes. */
static void OpenDebug(void)
{
    BSEC_HandleTypeDef hbsec;
    hbsec.Instance = BSEC;
    BSEC_DebugCfgTypeDef config_debug;
    config_debug.HDPL_Open_Dbg   = HAL_BSEC_OPEN_DBG_LEVEL_0;
    config_debug.NonSec_Dbg_Auth = HAL_BSEC_NONSEC_DBG_AUTH;
    config_debug.Sec_Dbg_Auth    = HAL_BSEC_SEC_DBG_AUTH;
    if (HAL_BSEC_ConfigDebug(&hbsec, &config_debug) != HAL_OK)
        Error_Handler();
    if (HAL_BSEC_UnlockDebug(&hbsec) != HAL_OK)
        Error_Handler();
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  /* Signal presence + detail on the VCP before halting. g_boot_stage tells how far the
   * boot got, so we know which init step failed. The RED LED also fast-blinks below.
   * (If the UART is not up yet, the printf is a no-op; the LED still signals.) */
  printf("\r\n!!! Error_Handler reached - boot_stage=%lu - halting !!!\r\n",
         (unsigned long)g_boot_stage);
  // __disable_irq();
  while (1)
  {
    HAL_Delay(200);
    BSP_LED_Off(LED_GREEN);
    BSP_LED_Toggle(LED_RED);
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
