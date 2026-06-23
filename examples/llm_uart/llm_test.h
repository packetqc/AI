/**
 * llm_test.h
 *
 * Calculator expression test harness for the on-device Qwen2 model.
 *
 * Usage in main.c:
 *
 *   #include "llm_test.h"
 *
 *   // After MX_* init and STM32CubeAI_Studio_AI_Init():
 *   LLM_TestCalc();
 *
 *   // Or replace the while(1) loop content:
 *   while (1) {
 *       LLM_TestCalc();
 *       HAL_Delay(5000);
 *   }
 */
#ifndef LLM_TEST_H
#define LLM_TEST_H

/**
 * Run a fixed set of calculator-expression prompts through the NPU.
 * For each expression: tokenize → one inference → decode predicted next token.
 * Results are printed over UART (USART1 at 115200).
 *
 * Initialises the NPU on first call; subsequent calls reuse the same handle.
 */
void LLM_TestCalc(void);

#endif /* LLM_TEST_H */
