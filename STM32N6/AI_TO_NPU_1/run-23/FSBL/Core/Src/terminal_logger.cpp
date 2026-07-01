/**
 * terminal_logger.cpp — see terminal_logger.hpp.
 * Faithful device port of TerminalLogger from class_terminal_logs.py.
 */
#include "terminal_logger.hpp"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

namespace llm {

/* ── palette (identical escapes to the Python COLORS dict) ──────────────── */

static const char* const RESET = "\033[0m";

const char* severity_name(Severity s)
{
    switch (s) {
        case Severity::Debug:    return "DEBUG";
        case Severity::Ok:       return "OK";
        case Severity::Warning:  return "WARNING";
        case Severity::Error:    return "ERROR";
        case Severity::Info:     return "INFO";
        case Severity::Critical: return "CRITICAL";
        case Severity::Notice:   return "NOTICE";
    }
    return "INFO";
}

const char* severity_color(Severity s)
{
    switch (s) {
        /* 16-color codes only — 256-color "\033[38;5;Nm" sequences contain ";5;" which minicom's
         * limited parser misreads as the BLINK attribute (\033[5m). Use basic/bright colors instead. */
        case Severity::Debug:    return "\033[34m";   /* Blue           */
        case Severity::Ok:       return "\033[32m";   /* Green          */
        case Severity::Warning:  return "\033[93m";   /* Bright Yellow  */
        case Severity::Error:    return "\033[31m";   /* Red            */
        case Severity::Info:     return "\033[37m";   /* Light Grey     */
        case Severity::Critical: return "\033[91m";   /* Bright Red     */
        case Severity::Notice:   return "\033[35m";   /* Magenta        */
    }
    return "";
}

/* ── helpers ────────────────────────────────────────────────────────────── */

void TerminalLogger::emit(const char* s) const
{
    if (!s) return;
    if (sink_) sink_(s, (int)strlen(s));   /* injected sink (host tests / custom TX) */
    else       fputs(s, stdout);           /* native printf path: device __io_putchar -> COM1 */
}

/* Device timestamp: uptime "HH:MM:SS.mmm" (no RTC at FSBL stage).
 * Mirrors the fixed-width time column of the Python "%Y-%m-%d %H:%M:%S.%f"[:‑3]. */
void TerminalLogger::format_time(char* buf, int n) const
{
    uint32_t ms = tick_ ? tick_() : 0u;
    uint32_t msec =  ms % 1000u;
    uint32_t sec  = (ms / 1000u) % 60u;
    uint32_t min  = (ms / 60000u) % 60u;
    uint32_t hour = (ms / 3600000u) % 100u;
    snprintf(buf, n, "%02u:%02u:%02u.%03u",
             (unsigned)hour, (unsigned)min, (unsigned)sec, (unsigned)msec);
}

/* ── core log: {color}{time}  {SEV:<10}{CAT:<16}{message}{RESET}\r\n ─────── */

void TerminalLogger::log(Severity sev, const char* category, const char* message) const
{
    char time_str[16];
    format_time(time_str, sizeof(time_str));

    /* Sanitize the message: blank out embedded control chars (esp. a bare '\r'). A mid-line carriage
     * return makes the next log line overwrite this one from column 0; when the newer line is shorter
     * it leaves this line's tail visible — the "blinking NOTICE" effect. The NPU grammar body can
     * carry a "\r" token (network_tokens.c). Keep printable ASCII + UTF-8 bytes (>=0x20). */
    char safe[256];
    if (message) {
        size_t j = 0;
        for (size_t i = 0; message[i] != '\0' && j < sizeof(safe) - 1U; i++) {
            unsigned char c = (unsigned char)message[i];
            safe[j++] = (c >= 0x20U) ? (char)c : ' ';
        }
        safe[j] = '\0';
    } else {
        safe[0] = '\0';
    }

    char line[320];
    const char* col   = color_ ? severity_color(sev) : "";
    const char* reset = color_ ? RESET : "";

    /* "%-10s%-16s" reproduces Python's {sev:<10}{cat:<16} column padding. */
    snprintf(line, sizeof(line), "%s%s  %-10s%-16s%s%s\r\n",
             col, time_str,
             severity_name(sev),
             category ? category : "",
             safe,
             reset);
    emit(line);
}

void TerminalLogger::logf(Severity sev, const char* category, const char* fmt, ...) const
{
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    log(sev, category, msg);
}

} /* namespace llm */

/* ── C -> C++ transition smoke test (called from the C FSBL main via extern "C") ──
 * Builds a printf-native TerminalLogger (no sink -> fputs(stdout) -> COM1) and emits
 * a couple of colored, time-stamped lines. Proves g++ compile, C/C++ link, extern "C"
 * interop, and the native-printf logger path all work inside the C FSBL. */
extern "C" uint32_t HAL_GetTick(void);

extern "C" void TerminalLogger_SmokeTest(void)
{
    llm::TerminalLogger log(HAL_GetTick, /*color=*/true);
    log.log (llm::Severity::Ok,     "C++TEST", "TerminalLogger live (printf-native)");
    log.log (llm::Severity::Info,   "C++TEST", "C -> C++ transition OK");
    log.logf(llm::Severity::Notice, "C++TEST", "tick=%lu ms", (unsigned long)HAL_GetTick());
}
