/**
 * fsbl_main_patch.c
 *
 * Copy these USER CODE blocks into FSBL/Core/Src/main.c.
 * They replace the normal boot flow with the LLM test loop.
 *
 * The FSBL will:
 *   1. Init clocks, GPIO, XSPI1 (PSRAM), XSPI2 (NOR flash) — as usual
 *   2. Map external memories — as usual (USER CODE BEGIN 2, already present)
 *   3. Init UART
 *   4. Init AI network
 *   5. Run calculator tests
 *   6. Spin in while(1) — NEVER jumps to BOOT_Application()
 *
 * ── In main.c: USER CODE BEGIN Includes ────────────────────────────────
 */
#include "stm32n6570_discovery_xspi.h"      /* already present */
#include "stm32n6570_discovery_errno.h"     /* already present */
#include "llm_fsbl.h"                       /* ADD */
#include "llm_test_fsbl.h"                  /* ADD */
/* ── END USER CODE BEGIN Includes ────────────────────────────────────── */


/**
 * ── In main.c: USER CODE BEGIN PV (private variables) ──────────────────
 * Add the UART handle if not already present:
 */
UART_HandleTypeDef huart1;
/* ── END USER CODE BEGIN PV ─────────────────────────────────────────── */


/**
 * ── In main.c: USER CODE BEGIN PFP (private function prototypes) ────────
 */
void MX_USART1_UART_Init(void);
/* ── END USER CODE BEGIN PFP ─────────────────────────────────────────── */


/**
 * ── In main.c: USER CODE BEGIN 2 ────────────────────────────────────────
 *
 * This block already has the XSPI / external memory init (keep it).
 * Add the UART + AI block AFTER the existing code in USER CODE BEGIN 2:
 *
 * (existing) __HAL_RCC_AXISRAM2_MEM_CLK_ENABLE(); ... etc.
 * (existing) BSP_XSPI_NOR_Init / EnableMemoryMappedMode
 * (existing) BSP_XSPI_RAM_Init / EnableMemoryMappedMode
 *
 * ADD after the existing memory init:
 */
void fsbl_user_code_2(void)
{
    /* ── UART init ── */
    MX_USART1_UART_Init();

    /* ── AI inference init ── */
    if (LLM_FSBL_Init() != 0) {
        /* If init fails, blink an LED or hang — adjust to your board */
        while (1) {}
    }

    /* ── Run calculator tests ── */
    LLM_TestCalcFSBL();

    /* ── Halt here — do NOT boot Appli ── */
    while (1) {}
}
/* ── END USER CODE BEGIN 2 ──────────────────────────────────────────── */


/**
 * ── MX_USART1_UART_Init — add as a new function in main.c ──────────────
 *
 * Paste this function body into FSBL/Core/Src/main.c (after Error_Handler).
 * STM32N6570-DK: USART1 TX=PA9, RX=PA10, VCP at 115200 baud.
 */
void MX_USART1_UART_Init(void)
{
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitTypeDef gpio = {0};
    gpio.Pin       = GPIO_PIN_9 | GPIO_PIN_10;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_LOW;
    gpio.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOA, &gpio);

    huart1.Instance          = USART1;
    huart1.Init.BaudRate     = 115200;
    huart1.Init.WordLength   = UART_WORDLENGTH_8B;
    huart1.Init.StopBits     = UART_STOPBITS_1;
    huart1.Init.Parity       = UART_PARITY_NONE;
    huart1.Init.Mode         = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart1);
}
