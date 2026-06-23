/**
 * llm_repl.cpp
 *
 * Interactive on-device REPL: the STM32N6 queries the NPU the same way the
 * host queries the Ollama model. Reads a line over USART1, auto-detects whether
 * it matches a loaded grammar (probe), and either:
 *   - runs the grammar engine (parse → evaluate → result, with the NPU queried
 *     once per rule as the grammar oracle — visible multi-step dialog), or
 *   - falls back to a free NPU generation (chat) for non-grammar input.
 *
 * C++ application layer; the C boot/HAL/NPU layer is reached via extern "C".
 * Embedded subset: no heap, bounded buffers.
 */
#include "terminal_logger.hpp"
#include "grammar_runner.hpp"

#include <string.h>
#include <stdio.h>

extern "C" {
#include "stm32n6xx_hal.h"
#include "llm_fsbl.h"          /* LLM_FSBL_Init / Generate */
extern UART_HandleTypeDef huart1;   /* defined in FSBL main.c */
}

namespace {

/* ── embedded calculator playbook (stored format: bare NTs, quoted terms) ── */
const llm::Rule kCalcPlaybook[] = {
    { "expr",   "expr \"+\" term | expr \"-\" term | term"        },
    { "term",   "term \"*\" factor | term \"/\" factor | factor"  },
    { "factor", "\"(\" expr \")\" | number"                        },
    { "number", "digit number | digit"                             },
    { "digit",  "\"0\" | \"1\" | \"2\" | \"3\" | \"4\" | \"5\" | "
                "\"6\" | \"7\" | \"8\" | \"9\""                     },
};
const int kCalcRules = (int)(sizeof(kCalcPlaybook) / sizeof(kCalcPlaybook[0]));

/* ── board I/O glue ─────────────────────────────────────────────────────── */

void uart_sink(const char* s, int n)
{
    HAL_UART_Transmit(&huart1, (const uint8_t*)s, (uint16_t)n, 2000);
}

uint32_t tick_ms(void) { return HAL_GetTick(); }

void uart_puts(const char* s) { uart_sink(s, (int)strlen(s)); }

/* Read one line (until CR/LF) into buf; echoes chars. Returns length. */
int uart_readline(char* buf, int max)
{
    int n = 0;
    for (;;) {
        uint8_t c;
        if (HAL_UART_Receive(&huart1, &c, 1, HAL_MAX_DELAY) != HAL_OK) continue;
        if (c == '\r' || c == '\n') { uart_puts("\r\n"); break; }
        if ((c == 0x7f || c == 0x08) && n > 0) {       /* backspace */
            n--; uart_puts("\b \b"); continue;
        }
        if (c >= 0x20 && c < 0x7f && n < max - 1) {
            buf[n++] = (char)c;
            HAL_UART_Transmit(&huart1, &c, 1, 100);     /* echo */
        }
    }
    buf[n] = '\0';
    return n;
}

/*
 * NPU oracle — the device's equivalent of get_ollama_answer(prompt).
 * Runs the trained model on `prompt`, returns generated text length.
 */
int npu_query(const char* prompt, char* out, int out_max, void* /*user*/)
{
    /* One forward pass per rule is enough for the visible NPU interaction —
     * the playbook is authoritative for parsing, so we don't need a long
     * generation here. Keeps the interactive dialog responsive. */
    int n = LLM_FSBL_Generate(prompt, out, out_max, 1);
    if (n < 0) { out[0] = '\0'; return -1; }
    return (int)strlen(out);
}

} /* anonymous namespace */

/* ── public entry (called from FSBL main USER CODE) ─────────────────────── */

extern "C" void LLM_Repl_Run(void)
{
    static llm::TerminalLogger logger(uart_sink, tick_ms, /*color=*/true);
    static llm::GrammarRunner  calc("calculator", kCalcPlaybook, kCalcRules,
                                    &logger, npu_query, nullptr);

    uart_puts("\r\n");
    uart_puts("================================================\r\n");
    uart_puts("  NPU Grammar REPL  [FSBL]  STM32N6570-DK\r\n");
    uart_puts("  type an expression (e.g. 3 + 4), or 'quit'\r\n");
    uart_puts("================================================\r\n");

    char line[128];
    for (;;) {
        uart_puts("\r\ncalc> ");
        int n = uart_readline(line, sizeof(line));
        if (n == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            uart_puts("bye.\r\n");
            break;
        }

        /* auto-detect: does it match the calculator grammar? (no NPU calls) */
        if (calc.probe(line, "expr")) {
            long result = 0;
            if (calc.run(line, "expr", &result)) {
                char msg[64];
                snprintf(msg, sizeof(msg), "\r\nResult: %ld\r\n", result);
                uart_puts(msg);
            } else {
                uart_puts("\r\n(parsed, non-numeric result)\r\n");
            }
        } else {
            /* not a grammar match → free NPU generation (chat) */
            logger.log(llm::Severity::Notice, "REPL", "not a calc expr — asking NPU...");
            char gen[160] = {0};
            int g = LLM_FSBL_Generate(line, gen, (int)sizeof(gen), 12);
            if (g > 0) { uart_puts("npu> "); uart_puts(gen); uart_puts("\r\n"); }
            else        uart_puts("npu> (no output)\r\n");
        }
    }
}
