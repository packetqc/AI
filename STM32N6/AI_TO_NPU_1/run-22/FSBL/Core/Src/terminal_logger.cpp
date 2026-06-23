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
        case Severity::Debug:    return "\033[34m";       /* Blue        */
        case Severity::Ok:       return "\033[32m";       /* Green       */
        case Severity::Warning:  return "\033[38;5;226m"; /* True Yellow */
        case Severity::Error:    return "\033[31m";       /* Red         */
        case Severity::Info:     return "\033[37m";       /* Light Grey  */
        case Severity::Critical: return "\033[38;5;208m"; /* Orange      */
        case Severity::Notice:   return "\033[38;5;129m"; /* Purple      */
    }
    return "";
}

/* ── helpers ────────────────────────────────────────────────────────────── */

void TerminalLogger::emit(const char* s) const
{
    if (sink_ && s) sink_(s, (int)strlen(s));
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
    if (!sink_) return;
    char time_str[16];
    format_time(time_str, sizeof(time_str));

    char line[320];
    const char* col   = color_ ? severity_color(sev) : "";
    const char* reset = color_ ? RESET : "";

    /* "%-10s%-16s" reproduces Python's {sev:<10}{cat:<16} column padding. */
    snprintf(line, sizeof(line), "%s%s  %-10s%-16s%s%s\r\n",
             col, time_str,
             severity_name(sev),
             category ? category : "",
             message ? message : "",
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
