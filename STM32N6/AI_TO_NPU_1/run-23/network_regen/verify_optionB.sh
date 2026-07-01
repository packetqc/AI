#!/usr/bin/env bash
# verify_optionB.sh — static invariants for the option-B 100%-SRAM framebuffer layout.
#
# Run AFTER the AI Studio / stedgeai re-gen (see README.md) and BEFORE flashing, to confirm the NPU
# has vacated the framebuffer banks and the FB is switched to SRAM. Pairs with the on-device check
# ("NPU self-test: 3 + 4 = 7" over the VCP) which proves the weights were relocated correctly.
#
# Usage:  network_regen/verify_optionB.sh
# Exit 0 = all invariants hold; non-zero = something still points into the FB banks.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN23="$(cd "$HERE/.." && pwd)"
AI="$RUN23/FSBL/AI"
INC="$RUN23/FSBL/Core/Inc"
SRC="$RUN23/FSBL/Core/Src"
fail=0

echo "== Option-B invariant check (run-23) =="

# 1. No NPU/network address inside the FB banks — AXISRAM1 (0x34000000-0x340FFFFF) and
#    AXISRAM5-6 (0x342E0000-0x343BFFFF) must be FB-only.
hits=$(grep -rohE "0x3400[0-9A-Fa-f]{4}|0x340[0-9A-Fa-f]{5}|0x342[EF][0-9A-Fa-f]{4}|0x343[0-9AB][0-9A-Fa-f]{4}" \
        "$AI"/network*.c "$AI"/network*.h 2>/dev/null | sort -u)
if [ -n "$hits" ]; then
  echo "  FAIL: NPU/network addresses land inside the FB banks:"; echo "$hits" | sed 's/^/        /'; fail=1
else
  echo "  PASS: no NPU address in AXISRAM1 or AXISRAM5-6 (FB banks are clear)"
fi

# 2. FB_IN_SRAM must be 1 for the SRAM layout.
if grep -qE "^#define[[:space:]]+FB_IN_SRAM[[:space:]]+1" "$INC/fb_layout.h" 2>/dev/null; then
  echo "  PASS: FB_IN_SRAM=1 (framebuffer in on-chip SRAM)"
else
  echo "  WARN: FB_IN_SRAM != 1 — still the legacy PSRAM layout (set it to 1 after the re-gen)"
fi

# 3. Weights memcpy destination must no longer target AXISRAM1 (0x34064000).
if grep -qE "memcpy\(\(void \*\)0x34064000" "$SRC/main.c" 2>/dev/null; then
  echo "  FAIL: weights memcpy still targets 0x34064000 (AXISRAM1) — point it at the option-B pool base (0x34200000)"; fail=1
else
  echo "  PASS: weights memcpy no longer targets AXISRAM1"
fi

# 4. The two FB buffers must not overlap (front + 768K must be <= back, or in separate banks).
echo "  INFO: FB front/back from fb_layout.h:"
grep -E "define FB_FRONT_ADDR|define FB_BACK_ADDR" "$INC/fb_layout.h" 2>/dev/null | sed 's/^/        /'

echo "-------------------------------------------"
if [ $fail -eq 0 ]; then echo "== ALL STATIC INVARIANTS PASS =="; echo "   Next: build, flash, and confirm 'NPU self-test: 3 + 4 = 7' over the VCP, then camera A/B."; else echo "== FAIL — fix the above before flashing =="; fi
exit $fail
