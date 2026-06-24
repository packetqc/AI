/**
 * npu_selftest.c — minimal in-FSBL NPU confirmation for VANILLA run-23.
 *
 * Runs ONE inference of the Studio-generated CNN on the Neural-ART NPU using the
 * vanilla run-23 config: weights in flash @ 0x71000000 (already flashed), RISAF8
 * enabled, ASYNC epoch controller.
 *
 * Readout is over UART (USART1, TX=PE5/RX=PE6, 115200, ST-Link VCP) because the
 * vanilla RISAF config isolates AXISRAM from the debug AP (so GDB can't read the
 * result). UART is independent of RISAF/debug. The board prints:
 *
 *   "NPU init rc=%d"            after stai_network_init
 *   "NPU RUN... "               before stai_network_run  (if this is the last line
 *                                seen, the epoch STALLED — run never returned)
 *   "DONE rc=%d ms=%u out0=%d"  after stai_network_run RETURNS (epoch completed)
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "stm32n6xx_hal.h"
#include "stai.h"
#include "stai_network.h"

UART_HandleTypeDef huart1;

STAI_NETWORK_CONTEXT_DECLARE(g_npu_ctx, STAI_NETWORK_CONTEXT_SIZE)

volatile int      g_npu_init_rc = -99;
volatile int      g_npu_run_rc  = -99;
volatile uint32_t g_npu_run_ms  = 0xFFFFFFFFu;
volatile int      g_npu_done    = 0;
volatile int8_t   g_npu_out0    = 0;

void NPU_UART_Init(void)
{
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin       = GPIO_PIN_5 | GPIO_PIN_6;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_LOW;
    gpio.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOE, &gpio);
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

static void up(const char *s)
{
    HAL_UART_Transmit(&huart1, (const uint8_t *)s, (uint16_t)strlen(s), 1000);
}

void NPU_SelfTest(void)
{
    char m[96];
    stai_ptr  in[STAI_NETWORK_IN_NUM];
    stai_ptr  out[STAI_NETWORK_OUT_NUM];
    stai_size n;

    up("\r\n==== run-23 VANILLA NPU self-test (Neural-ART) ====\r\n");

    g_npu_init_rc = (int)stai_network_init(g_npu_ctx);
    snprintf(m, sizeof(m), "NPU init rc=%d\r\n", g_npu_init_rc); up(m);
    if (g_npu_init_rc != STAI_SUCCESS) { up("init FAILED — stop\r\n"); g_npu_done = 1; return; }

    stai_network_get_inputs(g_npu_ctx, in, &n);
    stai_network_get_outputs(g_npu_ctx, out, &n);

    up("NPU RUN... (if this is the last line, the epoch STALLED)\r\n");
    uint32_t t0 = HAL_GetTick();
    g_npu_run_rc = (int)stai_network_run(g_npu_ctx, STAI_MODE_SYNC);   /* <-- does it RETURN? */
    g_npu_run_ms = HAL_GetTick() - t0;

    g_npu_out0 = ((const int8_t *)out[0])[0];
    g_npu_done = 1;

    snprintf(m, sizeof(m), "DONE rc=%d  ms=%lu  out0=%d  >>> NPU EPOCH COMPLETED <<<\r\n",
             g_npu_run_rc, (unsigned long)g_npu_run_ms, (int)g_npu_out0); up(m);
}
