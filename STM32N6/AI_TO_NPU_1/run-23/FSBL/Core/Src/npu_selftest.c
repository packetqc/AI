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
#include "stm32n6570_discovery.h"   /* BSP_LED — RISAF-immune, physically visible readout */
#include "stai.h"
#include "stai_network.h"

UART_HandleTypeDef huart1;

/* LED readout (works even when RISAF masks memory + UART):
 *   GREEN on = stai_network_run RETURNED  -> NPU epoch COMPLETED
 *   RED   on = stai_network_init FAILED   -> NPU init problem
 *   neither  = stuck inside stai_network_run -> epoch STALLED (the earlier polling-mode behaviour) */
void NPU_LED_Init(void)
{
    BSP_LED_Init(LED_GREEN);
    BSP_LED_Init(LED_RED);
    BSP_LED_Off(LED_GREEN);
    BSP_LED_Off(LED_RED);
}

/* Power the NPU RAM banks BEFORE NPU_Config / the inference. Verbatim from the
 * operational reference project N6_EDGEAI_1 (FSBL/Core/Src/main.c §
 * Enable_NPU_RAM_ForCore / Enable_AXICACHE_RAM_ForCore). The chip default in
 * run-mode boot leaves these power domains OFF; with RISAF8/15 (NPU_CACHE)
 * enabled, the Neural-ART streaming engine uses CACHEAXIRAM — if that RAM is
 * unpowered the engine reads dead memory and the epoch never completes (the
 * LL_ATON_RT_WFE stall). Powering CACHEAXIRAM is the missing piece. */
void NPU_PowerRAM(void)
{
    RAMCFG_HandleTypeDef h = {0};

    /* AXISRAM3-6 — AI activations (0x34200000 - 0x343BFFFF) */
    __HAL_RCC_AXISRAM3_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM4_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM5_MEM_CLK_ENABLE();
    __HAL_RCC_AXISRAM6_MEM_CLK_ENABLE();
    h.Instance = RAMCFG_SRAM3_AXI; HAL_RAMCFG_EnableAXISRAM(&h);
    h.Instance = RAMCFG_SRAM4_AXI; HAL_RAMCFG_EnableAXISRAM(&h);
    h.Instance = RAMCFG_SRAM5_AXI; HAL_RAMCFG_EnableAXISRAM(&h);
    h.Instance = RAMCFG_SRAM6_AXI; HAL_RAMCFG_EnableAXISRAM(&h);

    /* CACHEAXIRAM — the NPU cache RAM (0x343C0000 - 0x343FFFFF). THE FIX. */
    __HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
}

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
    if (g_npu_init_rc != STAI_SUCCESS) { up("init FAILED — stop\r\n"); BSP_LED_On(LED_RED); g_npu_done = 1; return; }

    stai_network_get_inputs(g_npu_ctx, in, &n);
    stai_network_get_outputs(g_npu_ctx, out, &n);

    up("NPU RUN... (if this is the last line, the epoch STALLED)\r\n");
    uint32_t t0 = HAL_GetTick();
    g_npu_run_rc = (int)stai_network_run(g_npu_ctx, STAI_MODE_SYNC);   /* <-- does it RETURN? */
    g_npu_run_ms = HAL_GetTick() - t0;

    g_npu_out0 = ((const int8_t *)out[0])[0];
    g_npu_done = 1;
    BSP_LED_On(LED_GREEN);   /* epoch completed — visible even through the RISAF mask */

    snprintf(m, sizeof(m), "DONE rc=%d  ms=%lu  out0=%d  >>> NPU EPOCH COMPLETED <<<\r\n",
             g_npu_run_rc, (unsigned long)g_npu_run_ms, (int)g_npu_out0); up(m);
}
