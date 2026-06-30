/**
 * lcd.c — minimal LTDC + panel bring-up for run-23 (ported from N6_EDGEAI_1/FSBL/Display).
 *
 * Stage A (this file): prove the LTDC -> panel path with NO framebuffer — disable the layers and
 * output a solid background colour (LTDC BCCR). This needs only the pixel clock (PLL3 -> IC16, set
 * by MX_LTDC_Init's MspInit) + the panel control GPIOs — no external PSRAM. A framebuffer (for the
 * LVGL content) is added in a later stage once a RAM home is settled.
 */
#include "lcd.h"
#include "ltdc.h"
#include "main.h"
#include <stdio.h>

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

void LCD_Init(void)
{
    printf("LCD: enter\r\n");
    lcd_panel_gpio_init();
    printf("LCD: gpio ok\r\n");

    /* Pre-init guard (methodology-ltdc-n6 §4) — CRITICAL for iterative GDB reload. A stale LTDC IRQ
     * + NVIC pending from the prior build fires the instant MspInit calls HAL_NVIC_EnableIRQ, before
     * hltdc.Instance is assigned -> HAL_LTDC_IRQHandler derefs a NULL handle (BFAR=0x68). Disable +
     * clear pending + force-reset the peripheral BEFORE init. Do NOT remove. */
    HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
    HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_ERR_IRQn);
    __HAL_RCC_LTDC_FORCE_RESET();
    __HAL_RCC_LTDC_RELEASE_RESET();
    printf("LCD: guard ok\r\n");

    MX_LTDC_Init();                          /* timing + IC16 (PLL3) pixel clock */
    printf("LCD: ltdc init ok\r\n");

    /* ROOT-CAUSE FIX (traced via GDB: VECTACTIVE=194 = LTDC_UP_ERR_IRQn, EXC_RETURN lr, stuck in
     * Default_Handler): with no valid framebuffer the LTDC raises a continuous FIFO-underrun. The MX
     * MspInit enables the LTDC NVIC line and run-23 has NO LTDC IRQ handler, so the IRQ traps in the
     * Default_Handler infinite loop = hang. Mask ALL LTDC interrupts at the SOURCE (IER=0), clear
     * latched flags + NVIC pending, and reload via a DIRECT SRCR write — HAL_LTDC_Reload would
     * re-enable LTDC_IT_RR in IER and re-arm the trap. */
    hltdc.Instance->IER = 0U;                /* no LTDC interrupt sources */
    hltdc.Instance->ICR = 0x3FU;             /* clear latched LI/FU/TE/RR error flags */
    HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
    HAL_NVIC_DisableIRQ(LTDC_UP_ERR_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
    HAL_NVIC_ClearPendingIRQ(LTDC_UP_ERR_IRQn);

    /* No framebuffer yet — MX enabled the layers pointing at address 0 (garbage). Disable both and
     * output a solid background colour to prove the LTDC -> panel signal path. */
    __HAL_LTDC_LAYER_DISABLE(&hltdc, 0);
    __HAL_LTDC_LAYER_DISABLE(&hltdc, 1);
    hltdc.Instance->BCCR = (0x00U << 16) | (0xFFU << 8) | (0x00U);   /* background GREEN (R,G,B) */
    hltdc.Instance->SRCR = LTDC_SRCR_IMR;    /* immediate reload, DIRECT (no IER re-enable) */
    printf("LCD: bccr+reload ok\r\n");
}
