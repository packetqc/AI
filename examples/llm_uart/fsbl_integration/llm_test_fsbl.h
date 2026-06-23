/**
 * llm_test_fsbl.h — FSBL calculator test harness
 */
#ifndef LLM_TEST_FSBL_H
#define LLM_TEST_FSBL_H

/**
 * Run 10 calculator expression tests via NPU.
 * Prints results over USART1 (huart1, 115200 baud).
 * Call after MX_USART1_UART_Init() and LLM_FSBL_Init().
 */
void LLM_TestCalcFSBL(void);

#endif /* LLM_TEST_FSBL_H */
