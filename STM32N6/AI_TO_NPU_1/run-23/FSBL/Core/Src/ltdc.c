/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    ltdc.c
  * @brief   This file provides code for the configuration
  *          of the LTDC instances.
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
#include "ltdc.h"

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

LTDC_HandleTypeDef hltdc;

/* LTDC init function */
void MX_LTDC_Init(void)
{

  /* USER CODE BEGIN LTDC_Init 0 */
  /* LTDC peripheral state + NVIC pending persist across GDB-driven SRAM
   * reloads (same class as XSPI bug). Without this, a stale LTDC IRQ from
   * a prior iteration fires as soon as HAL_LTDC_MspInit enables NVIC, BEFORE
   * hltdc.Instance is populated → HAL_LTDC_IRQHandler dereferences NULL at
   * hltdc->Instance->ISR2 → BFAR=0x68 → HardFault.
   * Force-reset LTDC + clear NVIC pending → clean init every time. */
  HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
  HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
  HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
  HAL_NVIC_ClearPendingIRQ(LTDC_UP_ERR_IRQn);
  __HAL_RCC_LTDC_FORCE_RESET();
  __HAL_RCC_LTDC_RELEASE_RESET();
  /* USER CODE END LTDC_Init 0 */

  LTDC_LayerCfgTypeDef pLayerCfg = {0};
  LTDC_LayerCfgTypeDef pLayerCfg1 = {0};

  /* USER CODE BEGIN LTDC_Init 1 */

  /* USER CODE END LTDC_Init 1 */
  hltdc.Instance = LTDC;
  hltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;
  hltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
  hltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
  hltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;
  hltdc.Init.HorizontalSync = 3;
  hltdc.Init.VerticalSync = 3;
  hltdc.Init.AccumulatedHBP = 11;
  hltdc.Init.AccumulatedVBP = 11;
  hltdc.Init.AccumulatedActiveW = 811;
  hltdc.Init.AccumulatedActiveH = 491;
  hltdc.Init.TotalWidth = 819;
  hltdc.Init.TotalHeigh = 499;
  hltdc.Init.Backcolor.Blue = 0;
  hltdc.Init.Backcolor.Green = 0;
  hltdc.Init.Backcolor.Red = 0;
  if (HAL_LTDC_Init(&hltdc) != HAL_OK)
  {
    Error_Handler();
  }
  pLayerCfg.WindowX0 = 0;
  pLayerCfg.WindowX1 = 800;
  pLayerCfg.WindowY0 = 0;
  pLayerCfg.WindowY1 = 480;
  pLayerCfg.PixelFormat = LTDC_PIXEL_FORMAT_RGB888;
  pLayerCfg.Alpha = 255;
  pLayerCfg.Alpha0 = 0;
  pLayerCfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_CA;
  pLayerCfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_CA;
  pLayerCfg.FBStartAdress = 0;
  pLayerCfg.ImageWidth = 800;
  pLayerCfg.ImageHeight = 480;
  pLayerCfg.Backcolor.Blue = 0;
  pLayerCfg.Backcolor.Green = 0;
  pLayerCfg.Backcolor.Red = 0;
  if (HAL_LTDC_ConfigLayer(&hltdc, &pLayerCfg, 0) != HAL_OK)
  {
    Error_Handler();
  }
  pLayerCfg1.WindowX0 = 0;
  pLayerCfg1.WindowX1 = 800;
  pLayerCfg1.WindowY0 = 0;
  pLayerCfg1.WindowY1 = 480;
  pLayerCfg1.PixelFormat = LTDC_PIXEL_FORMAT_ARGB4444;
  pLayerCfg1.Alpha = 255;
  pLayerCfg1.Alpha0 = 0;
  pLayerCfg1.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
  pLayerCfg1.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
  pLayerCfg1.FBStartAdress = 0;
  pLayerCfg1.ImageWidth = 800;
  pLayerCfg1.ImageHeight = 480;
  pLayerCfg1.Backcolor.Blue = 0;
  pLayerCfg1.Backcolor.Green = 0;
  pLayerCfg1.Backcolor.Red = 0;
  if (HAL_LTDC_ConfigLayer(&hltdc, &pLayerCfg1, 1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LTDC_Init 2 */

  /* USER CODE END LTDC_Init 2 */

}

void HAL_LTDC_MspInit(LTDC_HandleTypeDef* ltdcHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(ltdcHandle->Instance==LTDC)
  {
  /* USER CODE BEGIN LTDC_MspInit 0 */

  /* USER CODE END LTDC_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_LTDC;
    PeriphClkInitStruct.LtdcClockSelection = RCC_LTDCCLKSOURCE_IC16;
    PeriphClkInitStruct.ICSelection[RCC_IC16].ClockSelection = RCC_ICCLKSOURCE_PLL3;
    PeriphClkInitStruct.ICSelection[RCC_IC16].ClockDivider = 36;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* LTDC clock enable */
    __HAL_RCC_LTDC_CLK_ENABLE();

    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**LTDC GPIO Configuration
    PH3     ------> LTDC_B4
    PH6     ------> LTDC_B5
    PD15     ------> LTDC_R2
    PB14     ------> LTDC_HSYNC
    PB13     ------> LTDC_CLK
    PB2     ------> LTDC_B2
    PB15     ------> LTDC_G4
    PE11     ------> LTDC_VSYNC
    PD8     ------> LTDC_R7
    PH4     ------> LTDC_R4
    PD9     ------> LTDC_R1
    PG6     ------> LTDC_B3
    PA1     ------> LTDC_G2
    PB11     ------> LTDC_G6
    PA15(JTDI)     ------> LTDC_R5
    PG15     ------> LTDC_B0
    PG1     ------> LTDC_G1
    PB12     ------> LTDC_G5
    PA7     ------> LTDC_B1
    PG0     ------> LTDC_R0
    PA2     ------> LTDC_B7
    PG12     ------> LTDC_G0
    PB4(NJTRST)     ------> LTDC_R3
    PG8     ------> LTDC_G7
    PA8     ------> LTDC_B6
    PG13     ------> LTDC_DE
    PA0     ------> LTDC_G3
    PG11     ------> LTDC_R6
    */
    GPIO_InitStruct.Pin = LCD_B4_Pin|LCD_B5_Pin|LCD_R4_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_R2_Pin|LCD_R7_Pin|LCD_R1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_HSYNC_Pin|GPIO_PIN_13|LCD_B2_Pin|LCD_G4_Pin
                          |LCD_G6_Pin|LCD_G5_Pin|LCD_R3_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_VSYNC_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(LCD_VSYNC_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_B3_Pin|LCD_B0_Pin|LCD_G1_Pin|LCD_R0_Pin
                          |LCD_G0_Pin|LCd_G7_Pin|LCD_DE_Pin|LCD_R6_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = LCD_G2_Pin|LCD_R5_Pin|LCD_B1_Pin|LCD_B7_Pin
                          |LCD_B6_Pin|LCD_G3_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF14_LCD;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* LTDC interrupt Init — run-23 has NO LTDC IRQ handler and does not use LTDC interrupts. During
     * bring-up the LTDC raises a FIFO-underrun error (no valid framebuffer yet); if the NVIC line is
     * enabled the IRQ vectors to a NULL/Default handler -> HardFault (IBUSERR, stacked PC=0) or an
     * infinite Default_Handler loop. Traced via GDB: VECTACTIVE=194 (LTDC_UP_ERR_IRQn). So we keep
     * the priorities but do NOT enable the NVIC lines. (Re-enable + add handlers when LVGL needs the
     * line-event IRQ.) */
    HAL_NVIC_SetPriority(LTDC_UP_IRQn, 10, 0);
    /* HAL_NVIC_EnableIRQ(LTDC_UP_IRQn); */
    HAL_NVIC_SetPriority(LTDC_UP_ERR_IRQn, 10, 0);
    /* HAL_NVIC_EnableIRQ(LTDC_UP_ERR_IRQn); */
  /* USER CODE BEGIN LTDC_MspInit 1 */

  /* USER CODE END LTDC_MspInit 1 */
  }
}

void HAL_LTDC_MspDeInit(LTDC_HandleTypeDef* ltdcHandle)
{

  if(ltdcHandle->Instance==LTDC)
  {
  /* USER CODE BEGIN LTDC_MspDeInit 0 */

  /* USER CODE END LTDC_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_LTDC_CLK_DISABLE();

    /**LTDC GPIO Configuration
    PH3     ------> LTDC_B4
    PH6     ------> LTDC_B5
    PD15     ------> LTDC_R2
    PB14     ------> LTDC_HSYNC
    PB13     ------> LTDC_CLK
    PB2     ------> LTDC_B2
    PB15     ------> LTDC_G4
    PE11     ------> LTDC_VSYNC
    PD8     ------> LTDC_R7
    PH4     ------> LTDC_R4
    PD9     ------> LTDC_R1
    PG6     ------> LTDC_B3
    PA1     ------> LTDC_G2
    PB11     ------> LTDC_G6
    PA15(JTDI)     ------> LTDC_R5
    PG15     ------> LTDC_B0
    PG1     ------> LTDC_G1
    PB12     ------> LTDC_G5
    PA7     ------> LTDC_B1
    PG0     ------> LTDC_R0
    PA2     ------> LTDC_B7
    PG12     ------> LTDC_G0
    PB4(NJTRST)     ------> LTDC_R3
    PG8     ------> LTDC_G7
    PA8     ------> LTDC_B6
    PG13     ------> LTDC_DE
    PA0     ------> LTDC_G3
    PG11     ------> LTDC_R6
    */
    HAL_GPIO_DeInit(GPIOH, LCD_B4_Pin|LCD_B5_Pin|LCD_R4_Pin);

    HAL_GPIO_DeInit(GPIOD, LCD_R2_Pin|LCD_R7_Pin|LCD_R1_Pin);

    HAL_GPIO_DeInit(GPIOB, LCD_HSYNC_Pin|GPIO_PIN_13|LCD_B2_Pin|LCD_G4_Pin
                          |LCD_G6_Pin|LCD_G5_Pin|LCD_R3_Pin);

    HAL_GPIO_DeInit(LCD_VSYNC_GPIO_Port, LCD_VSYNC_Pin);

    HAL_GPIO_DeInit(GPIOG, LCD_B3_Pin|LCD_B0_Pin|LCD_G1_Pin|LCD_R0_Pin
                          |LCD_G0_Pin|LCd_G7_Pin|LCD_DE_Pin|LCD_R6_Pin);

    HAL_GPIO_DeInit(GPIOA, LCD_G2_Pin|LCD_R5_Pin|LCD_B1_Pin|LCD_B7_Pin
                          |LCD_B6_Pin|LCD_G3_Pin);

    /* LTDC interrupt Deinit */
    HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
    HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
  /* USER CODE BEGIN LTDC_MspDeInit 1 */

  /* USER CODE END LTDC_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

