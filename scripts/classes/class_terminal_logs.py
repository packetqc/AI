import datetime

class TerminalLogger:
    # Color palette — BASIC 16-color SGR only. 256-color codes ("\033[38;5;NNN m") are NOT used:
    # on a client without 256-color support (e.g. minicom), "38;5;NNN" is parsed as separate SGR
    # params 38 / 5 / NNN, and param 5 is the ANSI BLINK attribute — so 256-color text blinks and
    # loses its color. Basic codes (30-37, plus a bold variant) render correctly everywhere, no blink.
    RESET = "\033[0m"
    COLORS = {
        "DEBUG": "\033[34m",      # Blue
        "OK": "\033[32m",         # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "INFO": "\033[37m",       # Light Grey
        "CRITICAL": "\033[1;31m", # Bold Red (distinct from ERROR; was 256-color orange)
        "NOTICE": "\033[35m",     # Magenta/purple (was 256-color purple)
    }

    def _get_timestamp(self) -> str:
        """Generates the YYYY-MM-DD HH:MM:SS.ms string."""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def log(self, severity: str, category: str, message: str, *args) -> None:
        """Prints a perfectly aligned log line using your custom color palette."""
        time_str = self._get_timestamp()
        sev_upper = severity.upper()
        cat_upper = category.upper()
        color = self.COLORS.get(sev_upper, self.RESET)

        # Format extra arguments if they exist
        extra = f" | Args: {list(args)}" if args else ""
        full_msg = f"{message}{extra}"

        # Standard Python formatting layout for column structure
        print(
            f"{color}{time_str}  "
            f"{sev_upper:<10}"
            f"{cat_upper:<16}"
            f"{full_msg}{self.RESET}"
        )


# --- Quick Testing Suite ---
if __name__ == "__main__":
    logger = TerminalLogger()
    
    logger.log("info", "SYSTEM", "Worker thread initialized smoothly.")
    logger.log("debug", "ENGINE", "Garbage collection pass complete.", "freed_mb", 42)
    logger.log("ok", "DATABASE", "Connected to replica node successfully.")
    logger.log("notice", "SECURITY", "User updated their recovery profile.")
    logger.log("warning", "NETWORK", "Packet latency spikes detected over WAN.", "ping_ms", 340)
    logger.log("critical", "HARDWARE", "Core temperature approaching threshold!", "92C")
    logger.log("error", "API", "Webhook delivery failed completely.", "HTTP_503")
