/**
 * llm_repl.cpp
 *
 * Interactive nocode grammar REPL on the STM32N6570-DK — the host GrammarRunner
 * solution converted to the N6. Pioneer exploration of grammar-driven inference
 * at the edge, in two modes:
 *
 *   CPU mode (default, minimal):  the embedded grammar (playbook) is the
 *     authoritative oracle — parse + evaluate run on the Cortex-M55, instant,
 *     no NPU needed. Type "3 + 4" → "Result: 7" immediately.
 *
 *   Model dialog (opt-in, '/model on'):  also query the NPU model once per
 *     grammar rule (the visible "[model #N]" dialog) — the experimental frontier
 *     (slow today: the transformer falls back to SW + reads weights from flash).
 *
 * C++ application layer; the C boot/HAL/NPU layer is reached via extern "C".
 */
#include "terminal_logger.hpp"
#include "grammar_runner.hpp"

#include <string.h>
#include <stdio.h>

extern "C" {
#include "stm32n6xx_hal.h"
#include "llm_fsbl.h"          /* LLM_FSBL_Init / Generate — lazy, only for model dialog */
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
        if ((c == 0x7f || c == 0x08) && n > 0) { n--; uart_puts("\b \b"); continue; }
        if (c >= 0x20 && c < 0x7f && n < max - 1) {
            buf[n++] = (char)c;
            HAL_UART_Transmit(&huart1, &c, 1, 100);     /* echo */
        }
    }
    buf[n] = '\0';
    return n;
}

/* NPU oracle (the device's "Ollama") — one forward pass per rule (responsive). */
int npu_query(const char* prompt, char* out, int out_max, void* /*user*/)
{
    int n = LLM_FSBL_Generate(prompt, out, out_max, 1);
    if (n < 0) { out[0] = '\0'; return -1; }
    return (int)strlen(out);
}

bool g_npu_ready = false;   /* LLM_FSBL_Init done? (lazy) */
bool g_model_on  = false;   /* model dialog enabled?      */

} /* anonymous namespace */

/* ── public entry (called from FSBL main USER CODE) ─────────────────────── */

extern "C" void LLM_Repl_Run(void)
{
    static llm::TerminalLogger logger(uart_sink, tick_ms, /*color=*/true);
    /* query_fn = nullptr → CPU/playbook mode (minimal, instant). */
    static llm::GrammarRunner calc("calculator", kCalcPlaybook, kCalcRules,
                                   &logger, nullptr, nullptr);

    uart_puts("\r\n");
    uart_puts("================================================\r\n");
    uart_puts("  nocode Grammar REPL  [FSBL]  STM32N6570-DK\r\n");
    uart_puts("  CPU mode (instant): type an expr, e.g. 3 + 4\r\n");
    uart_puts("  '/model on' = NPU model dialog (experimental)\r\n");
    uart_puts("  '/model off' = back to CPU   ·   'quit'\r\n");
    uart_puts("================================================\r\n");

    /* Boot self-test: prove the model runs on the Neural-ART NPU and time it. */
    uart_puts("\r\nInitialising NPU model (Neural-ART)...\r\n");
    if (LLM_FSBL_Init() == 0) {
        g_npu_ready = true;
        calc.set_query(npu_query, nullptr); g_model_on = true;
        uart_puts("NPU OK. Self-test (model running on the NPU):\r\n");
        static const char* const k_probe[] = { "calculator expr", "calculator digit" };
        for (int i = 0; i < 2; i++) {
            char b[160] = {0};
            uint32_t t0 = HAL_GetTick();
            int g = LLM_FSBL_Generate(k_probe[i], b, (int)sizeof(b), 16);
            uint32_t dt = HAL_GetTick() - t0;
            char m[256];
            snprintf(m, sizeof(m), "  %s -> \"%s\"  (%lu ms)\r\n",
                     k_probe[i], g > 0 ? b : "(none)", (unsigned long)dt);
            uart_puts(m);
        }
    } else {
        uart_puts("NPU init FAILED — CPU grammar mode only.\r\n");
    }

    char line[128];
    for (;;) {
        uart_puts("\r\ncalc> ");
        int n = uart_readline(line, sizeof(line));
        if (n == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) { uart_puts("bye.\r\n"); break; }

        /* ── mode toggles ── */
        if (strcmp(line, "/model on") == 0) {
            if (!g_npu_ready) {
                uart_puts("initialising NPU model (one-time)...\r\n");
                g_npu_ready = (LLM_FSBL_Init() == 0);
            }
            if (g_npu_ready) {
                calc.set_query(npu_query, nullptr); g_model_on = true;
                uart_puts("model dialog ON  (slow: NPU queried per rule)\r\n");
            } else {
                uart_puts("NPU init FAILED — staying in CPU mode\r\n");
            }
            continue;
        }
        if (strcmp(line, "/model off") == 0) {
            calc.set_query(nullptr, nullptr); g_model_on = false;
            uart_puts("model dialog OFF  (CPU only, instant)\r\n");
            continue;
        }

        /* ── grammar: probe (CPU, instant) then run ── */
        if (calc.probe(line, "expr")) {
            long result = 0;
            if (calc.run(line, "expr", &result)) {
                char msg[64];
                snprintf(msg, sizeof(msg), "\r\nResult: %ld\r\n", result);
                uart_puts(msg);
            } else {
                uart_puts("\r\n(parsed, non-numeric result)\r\n");
            }
        } else if (g_model_on) {
            /* not a calc expr → free NPU generation (chat) */
            logger.log(llm::Severity::Notice, "REPL", "not a calc expr — asking NPU...");
            char gen[160] = {0};
            int g = LLM_FSBL_Generate(line, gen, (int)sizeof(gen), 12);
            if (g > 0) { uart_puts("npu> "); uart_puts(gen); uart_puts("\r\n"); }
            else        uart_puts("npu> (no output)\r\n");
        } else {
            uart_puts("not a calculator expression  (try '3 + 4', or '/model on')\r\n");
        }
    }
}
