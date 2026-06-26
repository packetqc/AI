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
#include "stm32n6570_discovery_xspi.h"
#include "stm32n6570_discovery_errno.h"
#include "stm32n6570_discovery.h"    /* BSP: COM1 VCP UART + LEDs (board-canonical) */
#include "stm32n6xx_hal_bsec.h"
#include "stm32n6xx_hal_ramcfg.h"
#include "llm_fsbl.h"
#include "llm_test_fsbl.h"
#include "npu_init.h"
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

XSPI_HandleTypeDef hxspi1;
XSPI_HandleTypeDef hxspi2;

/* USER CODE BEGIN PV */
UART_HandleTypeDef huart1;
volatile int debugFlag = 0;  /* 0 = run free (Programmer/flash boot). Set 1 to spin for GDB attach. */
volatile uint32_t g_boot_stage = 0;  /* init progress marker — read via GDB to localize a stall/error */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_XSPI1_Init(void);
static void MX_XSPI2_Init(void);
/* USER CODE BEGIN PFP */
void MX_USART1_UART_Init(void);
static void Enable_NPU_RAM_ForCore(void);
static void Enable_AXICACHE_RAM_ForCore(void);
static void OpenDebug(void);
void LLM_Repl_Run(void);   /* interactive NPU grammar REPL (llm_repl.cpp) */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Direct-register USART1 TX — HAL_UART_Transmit fails silently in the FSBL boot context
 * (HAL_GetTick may not advance), per reference N6_EDGEAI_1. Poll ISR.TXFNF (bit 7), write TDR. */
static void uart1_putc(char c)
{
    while ((USART1->ISR & (1u << 7)) == 0) { __NOP(); }
    USART1->TDR = (uint32_t)(uint8_t)c;
}
static void uart_puts(const char *s)
{
    if (USART1->CR1 == 0U) return;   /* UART not initialised yet */
    for (; *s; ++s) uart1_putc(*s);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* 1. Power up NPU RAMs and CACHEAXIRAM before anything can touch them.
   *    Boot ROM parks AXISRAM3-6 and CACHEAXIRAM in shutdown (SRAMSD=1).
   *    Must run before HAL_Init, BSS zeroing, .data copy, or any code that
   *    dereferences pointers into 0x34200000-0x343FFFFF. */
  g_boot_stage = 98;  /* main entered — CPU resumed from load_and_run */
  Enable_NPU_RAM_ForCore();
  g_boot_stage = 99;  /* past Enable_NPU_RAM_ForCore */
  Enable_AXICACHE_RAM_ForCore();

  /* 2. Re-authorize SWD debug for this entire FSBL session.
   *    Runtime BSEC register writes only — no OTP/fuse programming.
   *    NOTE: This resets the current debug AP session. The GDB server will
   *    get a broken pipe — this is EXPECTED. Restart GDB server and reconnect;
   *    the CPU will be spinning at while(debugFlag) below with full Secure auth. */
  /* OpenDebug();  — DISABLED for load-and-run debug. Under load_and_run the ST-Link
   * already controls the CPU and SWD is authorized, so OpenDebug's BSEC AP-reset just
   * reboots the device back to ROM (the SRAM image is lost → LEDs/UART never come up).
   * Re-enable only for the flash-boot-then-attach flow. */

  /* 3. GDB catch point — spin here so GDB can attach/observe, then write debugFlag=0
   *    via SWD to continue. Position is movable once the trace localizes the stall. */
  while (debugFlag) { __NOP(); }
  g_boot_stage = 10;  /* past debugFlag spin */

  /* 4. Disable stale MPU regions left by Boot ROM.
   *    Boot ROM may mark parts of AXISRAM as XN; first branch into such a
   *    region fires IACCVIOL → HardFault before UART is up. */
  HAL_MPU_Disable();

  /* 5. Clear stack-pointer limits and FPCA (see original comments). */
  __set_MSPLIM(0U);
  __set_PSPLIM(0U);
#if defined(__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  __TZ_set_MSPLIM_NS(0U);
  __TZ_set_PSPLIM_NS(0U);
#endif
  __set_CONTROL(__get_CONTROL() & ~CONTROL_FPCA_Msk);
  __ISB();
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();
  g_boot_stage = 11;  /* past HAL_Init */

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();
  g_boot_stage = 12;  /* past SystemClock_Config */

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();

  /* BSP LEDs early (right after GPIO) to signal init-phase errors, forced to a known OFF
   * state first — the N6570-DK LEDs are active-inverse, so the post-init/default level
   * can read as ON; BSP_LED_Off normalizes it. RED is lit by Error_Handler (an init
   * failed); GREEN (below) = reached the REPL. (HW breakpoints don't catch under
   * load-and-run, so the LEDs + g_boot_stage are our init trace.) */
  BSP_LED_Init(LED_GREEN);
  BSP_LED_Init(LED_RED);
  BSP_LED_Off(LED_GREEN);
  BSP_LED_Off(LED_RED);
  g_boot_stage = 13;  /* past MX_GPIO_Init + BSP LED init */
  /* MX_XSPI1_Init() — skipped: PSRAM not used, hxspi1 never referenced */
  /* MX_XSPI2_Init() — skipped: load_and_run leaves XSPI2 mid-transaction (BUSY=1);
   * HAL_XSPIM_Config polls BUSY forever. BSP_XSPI_NOR_Init handles XSPI2 from
   * scratch and works correctly without the HAL mux pre-config. */

  /* MX_EXTMEM_MANAGER_Init() — skip: BOOT_Application() path not used */
  /* USER CODE BEGIN 2 */

  /* UART up FIRST in USER CODE 2 — direct-register USART1 (uart_puts polls ISR.TXFNF, no
   * HAL_UART: per reference N6_EDGEAI_1, HAL_UART_* fails silently in the FSBL boot context
   * because HAL_GetTick doesn't advance). Brought up early so each init step is traced. */
  MX_USART1_UART_Init();
  uart_puts("\r\n[run-23] FSBL alive — boot trace:\r\n");

  /* NOR flash memory-mapped init FIRST — BEFORE NPU_Config / CACHEAXI enable (CACHEAXI on
   * the AXI path would intercept XSPI2 command transactions during NOR init). */
  BSP_XSPI_NOR_Init_t xspiInit;
  xspiInit.InterfaceMode = MX66UW1G45G_OPI_MODE;
  xspiInit.TransferRate  = MX66UW1G45G_DTR_TRANSFER;
  g_boot_stage = 1;  uart_puts("[boot] 1 XSPI NOR\r\n");
  BSP_XSPI_NOR_Init(0, &xspiInit);
  BSP_XSPI_NOR_EnableMemoryMappedMode(0);
  MODIFY_REG(XSPI2->CR, XSPI_CR_NOPREF, HAL_XSPI_AUTOMATIC_PREFETCH_DISABLE);

  /* NPU + security init — AFTER NOR flash is mapped (reference project order) */
  g_boot_stage = 2;  uart_puts("[boot] 2 SystemInit_POST + cache\r\n");
  SystemInit_POST();
  SCB_EnableICache();
  SCB_EnableDCache();

  g_boot_stage = 3;  uart_puts("[boot] 3 NPU_Config\r\n");
  NPU_Config();
  g_boot_stage = 4;  uart_puts("[boot] 4 RISAF_Config\r\n");
  RISAF_Config();

  /* nocode grammar REPL over USART1 — the host solution on the N6 (CM55 CPU; the embedded
   * grammar playbook is the oracle; the NPU model dialog is opt-in via '/model on'). */
  g_boot_stage = 6;  uart_puts("[boot] 6 REPL\r\n");
  BSP_LED_On(LED_GREEN);             /* proof of life: reached the REPL, all init done */
  LLM_Repl_Run();
  while (1) {}  /* prevent BOOT_Application() — remove to restore normal boot */

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

/**
 * @brief USART1 initialization — TX=PE5, RX=PE6, AF7 (BSP COM1 on STM32N6570-DK VCP).
 */
void MX_USART1_UART_Init(void)
{
  /* BSP COM1 = USART1 on PE5/PE6 AF7 (STM32N6570-DK ST-Link VCP) — board-canonical
   * init: BSP_COM_Init configures the GPIO pins + the USART together (the documented
   * way), avoiding any pin/MSP ordering pitfalls. The REPL and uart_puts use the global
   * huart1, so mirror the BSP-initialised handle (hcom_uart[COM1]) into it. */
  /* Direct-register USART1 init — no HAL_UART_Init (its TEACK/REACK poll uses HAL_GetTick,
   * which may not advance in the FSBL boot context). PE5=TX, PE6=RX, AF7; 115200 8N1, over16. */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_USART1_CLK_ENABLE();

  GPIO_InitTypeDef gpio = {0};
  gpio.Pin       = GPIO_PIN_5 | GPIO_PIN_6;
  gpio.Mode      = GPIO_MODE_AF_PP;
  gpio.Pull      = GPIO_NOPULL;
  gpio.Speed     = GPIO_SPEED_FREQ_HIGH;
  gpio.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOE, &gpio);

  uint32_t uart_clk = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_USART1);
  if (uart_clk == 0U) uart_clk = 64000000U;            /* fallback: HSI/default */
  USART1->CR1 = 0U;                                    /* disable while configuring */
  USART1->BRR = (uart_clk + (115200U / 2U)) / 115200U; /* oversampling by 16 */
  USART1->CR2 = 0U;
  USART1->CR3 = 0U;
  USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
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
  __disable_irq();
  while (1)
  {
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
