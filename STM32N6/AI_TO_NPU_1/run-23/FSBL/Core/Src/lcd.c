/**
 * lcd.c — LTDC + panel bring-up for run-23 (transposed from N6_EDGEAI_1/FSBL/Display).
 *
 * Stage B2: Layer-1 (LayerIdx 0) RGB565 framebuffer in the mapped APS256 PSRAM @0x90000000, painted
 * with a static R/G/B test pattern. Requires: PSRAM mapped (PSRAM_Init, g_psram_rc==0) + LTDC RIF
 * master+slaves granted (main.c RIF_Init). The §4 pre-init guard + source-level LTDC interrupt mask
 * (no LTDC IRQ handler in run-23) carry over from Stage A. If PSRAM is unavailable, falls back to a
 * solid background colour (layers disabled) so the device stays stable.
 */
#include "lcd.h"
#include "ltdc.h"
#include "main.h"
#include <stdio.h>

#define LCD_W    800U
#define LCD_H    480U
#define LCD_FB0  0x90000000U          /* RGB565 framebuffer in the mapped APS256 PSRAM */

extern volatile int g_psram_rc;       /* 0 = PSRAM mapped + CPU-writable (set in MX_XSPI1_Init) */

/* Panel power / reset / backlight control pins (RK050 on the N6570-DK): drive HIGH. */
static void lcd_panel_gpio_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOQ_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    HAL_GPIO_WritePin(GPIOQ, GPIO_PIN_3 | GPIO_PIN_6, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_1,              GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOG, GPIO_PIN_13,             GPIO_PIN_SET);
    g.Mode = GPIO_MODE_OUTPUT_PP; g.Pull = GPIO_NOPULL; g.Speed = GPIO_SPEED_FREQ_LOW;
    g.Pin = GPIO_PIN_3 | GPIO_PIN_6; HAL_GPIO_Init(GPIOQ, &g);
    g.Pin = GPIO_PIN_1;              HAL_GPIO_Init(GPIOE, &g);
    g.Pin = GPIO_PIN_13;             HAL_GPIO_Init(GPIOG, &g);   /* PG13 static HIGH (panel) */
}

/* R/G/B horizontal bands — proves the LTDC scans the PSRAM framebuffer (vs the BCCR background). */
static void lcd_fill_test_pattern(void)
{
    uint16_t *fb = (uint16_t *)LCD_FB0;
    for (uint32_t y = 0; y < LCD_H; ++y) {
        uint16_t c = (y < LCD_H / 3U) ? 0xF800u : (y < 2U * LCD_H / 3U) ? 0x07E0u : 0x001Fu;
        for (uint32_t x = 0; x < LCD_W; ++x) fb[y * LCD_W + x] = c;
    }
}

void LCD_Init(void)
{
    printf("LCD: enter (g_psram_rc=%d)\r\n", g_psram_rc);
    lcd_panel_gpio_init();

    /* §4 pre-init guard (methodology-ltdc-n6) — clear stale LTDC IRQ/NVIC before init. */
    HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
    HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_ERR_IRQn);
    __HAL_RCC_LTDC_FORCE_RESET();
    __HAL_RCC_LTDC_RELEASE_RESET();

    MX_LTDC_Init();                          /* 800x480 timing + IC16 (PLL3) pixel clock */

    /* Mask LTDC interrupts at the SOURCE (run-23 has no LTDC handler — a fired IRQ traps/hangs). */
    hltdc.Instance->IER = 0U;
    hltdc.Instance->ICR = 0x3FU;
    HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
    HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_ERR_IRQn);

    hltdc.Instance->BCCR = 0x000000U;        /* black background (shows where no layer covers) */

    if (g_psram_rc == 0) {
        /* Paint the pattern, push it out of D-cache so the LTDC AXI master reads fresh pixels, then
         * point Layer-1 (idx 0) at it in RGB565 via direct register writes (HAL config zeros CFBAR). */
        lcd_fill_test_pattern();
        SCB_CleanDCache_by_Addr((uint32_t *)LCD_FB0, LCD_W * LCD_H * 2U);
        LTDC_Layer1->CFBAR  = LCD_FB0;
        LTDC_Layer1->CFBLR  = ((LCD_W * 2U) << 16) | ((LCD_W * 2U) + 3U);   /* RGB565 = 2 B/pixel */
        LTDC_Layer1->CFBLNR = LCD_H;
        LTDC_Layer1->PFCR   = 4U;            /* STM32N6: 100 = RGB565 (RM0486 LTDC_LxPFCR) */
        __HAL_LTDC_LAYER_ENABLE(&hltdc, 0);
        __HAL_LTDC_LAYER_DISABLE(&hltdc, 1); /* L2 (ARGB4444 overlay) off until LVGL */
        printf("LCD: Layer-1 FB @0x%08lX RGB565 800x480 (R/G/B bands)\r\n", (unsigned long)LCD_FB0);
    } else {
        /* No PSRAM — keep the panel stable on the background colour only. */
        __HAL_LTDC_LAYER_DISABLE(&hltdc, 0);
        __HAL_LTDC_LAYER_DISABLE(&hltdc, 1);
        printf("LCD: no framebuffer (g_psram_rc=%d) — background only\r\n", g_psram_rc);
    }

    hltdc.Instance->SRCR = LTDC_SRCR_IMR;    /* immediate reload, direct (no IER re-enable) */
    printf("LCD: reload done\r\n");
}
