/**
 * llm_test_fsbl.c
 *
 * Calculator expression test harness for FSBL build.
 *
 * Differences from llm_test.c (Appli build):
 *   - UART handle is huart1 from FSBL main.c (extern, no bsp_ai.h)
 *   - Uses LLM_FSBL_* API instead of LLM_NPU_*
 *
 * Add this call to FSBL main.c USER CODE before BOOT_Application():
 *
 *   MX_USART1_UART_Init();     // UART for results
 *   LLM_FSBL_Init();           // AI init
 *   LLM_TestCalcFSBL();        // run tests
 *   while (1) {}               // halt — do NOT boot Appli
 */

#include "llm_test_fsbl.h"
#include "llm_fsbl.h"
#include "llm_tokenizer.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "stm32n6xx_hal.h"

extern UART_HandleTypeDef huart1;   /* declared in FSBL/Core/Src/main.c */

/* ── UART helpers ─────────────────────────────────────────────────────── */

static void uart_puts(const char *s)
{
    HAL_UART_Transmit(&huart1, (const uint8_t *)s, (uint16_t)strlen(s), 2000);
}

#define uart_printf(fmt, ...) \
    do { char _b[160]; snprintf(_b, sizeof(_b), fmt, ##__VA_ARGS__); uart_puts(_b); } while(0)

/* ── Test table ───────────────────────────────────────────────────────── */

typedef struct { const char *prompt; const char *note; } test_t;

static const test_t k_tests[] = {
    { "1+1",                    "simple addition"        },
    { "2*3",                    "multiplication"         },
    { "10-4",                   "subtraction"            },
    { "8/2",                    "division"               },
    { "(2+3)*4",                "parentheses"            },
    { "1+2+3",                  "chained addition"       },
    { "100/10",                 "two-digit operands"     },
    { "calculator expr",        "grammar rule query"     },
    { "calculator digit",       "digit rule query"       },
    { "The grammar defines",    "sentence continuation"  },
};
#define N_TESTS (sizeof(k_tests)/sizeof(k_tests[0]))

/* ── Main test entry point ────────────────────────────────────────────── */

void LLM_TestCalcFSBL(void)
{
    uart_puts("\r\n");
    uart_puts("================================================\r\n");
    uart_puts("  Qwen2 NPU Calc Test  [FSBL]  STM32N6570-DK\r\n");
    uart_puts("================================================\r\n");

    int pass = 0;
    for (int t = 0; t < (int)N_TESTS; t++) {
        const test_t *tc = &k_tests[t];

        int32_t ids[LLM_SEQ_LEN]  = {0};
        int32_t mask[LLM_SEQ_LEN] = {0};
        int n_tok = LLM_Tokenize(tc->prompt, ids, mask, LLM_SEQ_LEN);

        uart_printf("[%02d/%02d] \"%s\"  (%d tokens)\r\n",
                    t + 1, (int)N_TESTS, tc->prompt, n_tok);

        int32_t next_tok = -1;
        int ret = LLM_FSBL_RunOnce(ids, mask, n_tok, &next_tok);
        if (ret != 0) {
            uart_printf("        [INFERENCE ERROR %d]\r\n\r\n", ret);
            continue;
        }

        uart_printf("        next: %d → \"%s\"\r\n",
                    (int)next_tok, LLM_TokenStr(next_tok));

        char cont[128] = {0};
        int n_gen = LLM_FSBL_Generate(tc->prompt, cont, sizeof(cont), 8);
        if (n_gen > 0)
            uart_printf("        continuation (%d): \"%s\"\r\n", n_gen, cont);

        uart_printf("        [%s]\r\n\r\n", tc->note);
        pass++;
    }

    uart_puts("------------------------------------------------\r\n");
    uart_printf("Done: %d/%d\r\n", pass, (int)N_TESTS);
    uart_puts("================================================\r\n\r\n");
}
