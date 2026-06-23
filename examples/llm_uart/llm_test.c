/**
 * llm_test.c
 *
 * Calculator expression tests for the on-device Qwen2 LLM.
 *
 * Each test feeds a calculator expression as a prompt and prints the model's
 * predicted next token.  This validates the complete NPU pipeline:
 *   UART rx → BPE tokenize → STAI inference → int8 argmax → BPE decode → UART tx
 *
 * The model was trained on a calculator BNF grammar, so its predictions are
 * grammar continuations, not arithmetic results — the goal is confirming the
 * NPU path works end-to-end, not verifying arithmetic correctness.
 */

#include "llm_test.h"
#include "llm_npu.h"
#include "llm_tokenizer.h"
#include "bsp_ai.h"   /* UartHandle */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ── UART helper ─────────────────────────────────────────────────────── */

static void uart_puts(const char *s)
{
    HAL_UART_Transmit(&UartHandle, (const uint8_t *)s, (uint16_t)strlen(s), 2000);
}

/* snprintf into a fixed stack buffer then send */
#define uart_printf(fmt, ...) \
    do { char _b[128]; snprintf(_b, sizeof(_b), fmt, ##__VA_ARGS__); uart_puts(_b); } while(0)

/* ── Test table ───────────────────────────────────────────────────────── */

typedef struct {
    const char *prompt;          /* input expression */
    const char *description;     /* human-readable note */
} test_case_t;

static const test_case_t k_tests[] = {
    { "1+1",              "simple addition"           },
    { "2*3",              "multiplication"            },
    { "10-4",             "subtraction"               },
    { "8/2",              "division"                  },
    { "(2+3)*4",          "parentheses with ops"      },
    { "1+2+3",            "chained addition"          },
    { "100/10",           "two-digit operands"        },
    { "calculator expr",  "grammar rule query"        },
    { "calculator digit", "digit rule query"          },
    { "The grammar defines rules", "full sentence"    },
};

#define NUM_TESTS  (sizeof(k_tests) / sizeof(k_tests[0]))

/* ── Test runner ──────────────────────────────────────────────────────── */

void LLM_TestCalc(void)
{
    static int s_initialised = 0;

    uart_puts("\r\n");
    uart_puts("============================================\r\n");
    uart_puts("  Qwen2 NPU Calculator Test — STM32N6570-DK\r\n");
    uart_puts("============================================\r\n");

    if (!s_initialised) {
        uart_puts("Initialising NPU...\r\n");
        if (LLM_NPU_Init() != 0) {
            uart_puts("[ERROR] LLM_NPU_Init failed\r\n");
            return;
        }
        s_initialised = 1;
        uart_puts("NPU ready.\r\n\r\n");
    }

    int pass = 0;

    for (int t = 0; t < (int)NUM_TESTS; t++) {
        const test_case_t *tc = &k_tests[t];

        /* ── Tokenize ── */
        int32_t ids[LLM_SEQ_LEN]  = {0};
        int32_t mask[LLM_SEQ_LEN] = {0};
        int n_tok = LLM_Tokenize(tc->prompt, ids, mask, LLM_SEQ_LEN);

        uart_printf("[%02d/%02d] prompt: \"%s\"\r\n", t + 1, (int)NUM_TESTS, tc->prompt);
        uart_printf("        tokens (%d): ", n_tok);
        for (int i = 0; i < n_tok; i++) {
            uart_printf("%d(\"%s\") ", (int)ids[i], LLM_TokenStr(ids[i]));
        }
        uart_puts("\r\n");

        /* ── One inference step → next token ── */
        int32_t next_tok = -1;
        int ret = LLM_NPU_RunOnce(ids, mask, n_tok, &next_tok);
        if (ret != 0) {
            uart_printf("        [INFERENCE ERROR %d]\r\n\r\n", ret);
            continue;
        }

        uart_printf("        next token: %d → \"%s\"\r\n", (int)next_tok,
                    LLM_TokenStr(next_tok));

        /* ── Multi-token continuation (up to 8 new tokens) ── */
        char continuation[128] = {0};
        int n_gen = LLM_Generate(tc->prompt, continuation, sizeof(continuation), 8);
        if (n_gen > 0)
            uart_printf("        continuation (%d tok): \"%s\"\r\n", n_gen, continuation);

        uart_printf("        note: %s\r\n\r\n", tc->description);
        pass++;
    }

    uart_puts("--------------------------------------------\r\n");
    uart_printf("Result: %d/%d tests completed\r\n", pass, (int)NUM_TESTS);
    uart_puts("============================================\r\n\r\n");
}
