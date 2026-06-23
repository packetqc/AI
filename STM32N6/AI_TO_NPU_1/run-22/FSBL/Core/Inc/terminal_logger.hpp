/**
 * terminal_logger.hpp
 *
 * C++ device port of classes/class_terminal_logs.py (TerminalLogger).
 * Reproduces the exact colored, column-aligned log format on the STM32N6
 * over UART. Embedded subset: no exceptions, no RTTI, no heap, no std::string.
 *
 * Python original line layout:
 *   {color}{time}  {SEV:<10}{CAT:<16}{message}{RESET}
 *   time = "HH:MM:SS.mmm"   (device: uptime; host: wall clock via injected tick)
 *
 * Output and time source are injected so the class is host-unit-testable and
 * board-independent (sink → UART on device, stdout on host).
 */
#ifndef TERMINAL_LOGGER_HPP
#define TERMINAL_LOGGER_HPP

#include <stdint.h>

namespace llm {

/* Severity levels — mirror the Python COLORS palette keys, same order/colors. */
enum class Severity {
    Debug,     /* blue   \033[34m       */
    Ok,        /* green  \033[32m       */
    Warning,   /* yellow \033[38;5;226m */
    Error,     /* red    \033[31m       */
    Info,      /* grey   \033[37m       */
    Critical,  /* orange \033[38;5;208m */
    Notice     /* purple \033[38;5;129m */
};

/* Byte sink: write exactly n bytes of s (e.g. UART TX). */
typedef void (*LogSink)(const char* s, int n);

/* Monotonic milliseconds (e.g. HAL_GetTick on device). May be null → 0. */
typedef uint32_t (*TickMs)(void);

const char* severity_name(Severity s);   /* "DEBUG", "OK", ... */
const char* severity_color(Severity s);  /* ANSI escape, "" if unknown */

class TerminalLogger {
public:
    TerminalLogger(LogSink sink, TickMs tick, bool color = true)
        : sink_(sink), tick_(tick), color_(color) {}

    /* Core: format and emit one aligned, colored log line (CRLF-terminated). */
    void log(Severity sev, const char* category, const char* message) const;

    /* printf-style convenience — formats into a bounded stack buffer. */
    void logf(Severity sev, const char* category, const char* fmt, ...) const
        __attribute__((format(printf, 4, 5)));

    void set_color(bool on) { color_ = on; }

private:
    void emit(const char* s) const;        /* strlen + sink */
    void format_time(char* buf, int n) const;  /* "HH:MM:SS.mmm" from tick */

    LogSink sink_;
    TickMs  tick_;
    bool    color_;
};

} /* namespace llm */

#endif /* TERMINAL_LOGGER_HPP */
