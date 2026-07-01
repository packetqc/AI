# run-23 — Autonomous on-chip NPU grammar calculator (STM32N6570-DK · Neural-ART)

Pure-FSBL firmware (no Appli jump) for the STM32N6570-DK. A C++ grammar-runner evaluates
arithmetic expressions (e.g. `3 + 4`, `6 * 7`) by recalling each grammar rule from a TCN model
that runs **on the Neural-ART NPU** (hardware epochs) — the NPU is the grammar *oracle*. I/O is
over the ST-Link VCP UART (`calc>` prompt); the whole pipeline runs on-chip with no host.

## Architecture

| Piece | Detail |
|-------|--------|
| Core | Cortex-M55 FSBL only (boot ROM → FSBL → app; no second-stage Appli) |
| NPU | Neural-ART (ATON), LL_ATON runtime — `LL_ATON_RT_ASYNC` + bare-metal OSAL; epoch-done IRQ on `NPU0_IRQn` wakes the `__WFE()` (handler provided by the runtime) |
| Model | TCN, int8, **HW** export (`--st-neural-art`, 5 pure-HW epochs) — see [`models/npu_export/NPU_HW_GENERATE.md`](../../../models/npu_export/NPU_HW_GENERATE.md) |
| Weights | Flashed to **XSPI2 NOR @0x70200000**, copied to **AXISRAM1 @0x34064000** at boot (one path for dev=0 + dev=1). `.ai_weights` is NOLOAD; FSBL image ≈ 244 KB |
| Runner | `FSBL/AI/grammar_runner.cpp` + `npu_query.c` (autoregressive rule recall) → `FSBL/Core/Src/main.c` self-test + interactive REPL |

**Why flash-copy, not XIP or baked:** baking weights into the FSBL image leaves the SRAM-VMA blob
out of the signed dev=0 image ("assets not on the system"); XIP read-in-place from XSPI **stalls
the Neural-ART epoch**. Copying flash→AXISRAM at boot fixes both and unifies dev=0/dev=1.

## Build

```bash
cd Makefile/FSBL && make -f Makefile.local all      # -> build/run-23_FSBL.bin (~244 KB)
```

## Flash the weights (once, persists across reflashes)

```bash
STM32_Programmer_CLI -c port=SWD mode=UR \
  -el MX66UW1G45G_STM32N6570-DK.stldr -w FSBL/AI/network_weights.bin 0x70200000 -v
```

## Run — dev=1 (debugger / development)

`load_and_run` the ELF, clear the real `debugFlag` (read its address from the ELF — it moves with
the build), drop the `load_and_run` breakpoint at `main` (else the resume re-traps it), then
release. The boot copies the weights from the flashed NOR (the image carries no weights). UART:

```
MAIN APP ON ... "3 + 4" = 7 (5 rule-queries via NPU oracle) ... calc>
```

## Run — dev=0 (autonomous, boot from flash, no probe)

```bash
# sign + flash the FSBL to 0x70000000
STM32_SigningTool_CLI -s -bin build/run-23_FSBL.bin -nk -of 0x80000000 -t fsbl \
  -hv 2.3 -align -la 0x34180000 -o build/run-23_FSBL-trusted.bin -dump build/run-23_FSBL-trusted.bin
STM32_Programmer_CLI -c port=SWD mode=UR \
  -el MX66UW1G45G_STM32N6570-DK.stldr -d build/run-23_FSBL-trusted.bin 0x70000000
```

Set the boot switch to dev=0 (boot-from-flash) and power-cycle → boot-ROM loads the FSBL → it
copies the weights → the NPU calculator runs autonomously on UART.

## Status — device-validated

✅ `3 + 4 = 7` and `6 * 7 = 42` (Neural-ART grammar oracle, 5 rule-queries/eval) on UART, in
**both dev=1 and dev=0**.

## Display — LVGL calculator UI + NPU coexistence

`run-23` drives the 800×480 RGB565 panel over LTDC with an LVGL 9.2.3 scene (top app-tab bar,
centred 60 px inference/response text, bottom SW-LED + version bar). The display shares the FSBL
super-loop with the NPU, so the interesting engineering is the **LTDC ↔ Neural-ART coexistence**.

### Memory map — who lives where

| Master | Memory | Address |
|--------|--------|---------|
| NPU weights | AXISRAM1 | `0x34064000` (copied flash→SRAM at boot) |
| NPU activations (`.AI_RAM`) | AXISRAM3 (+ HW pool in 4-6) | `0x34200000` |
| LVGL heap | AXISRAM1 gap | `0x34100000` (256 KB) |
| **LTDC framebuffer(s)** | **PSRAM** | **`0x90000000` / `0x90100000`** |

**The NPU never touches PSRAM.** Its weights and activations live entirely in on-chip AXISRAM — there
is no `0x9…` address anywhere in the `ll_aton` network config. The *only* master in external PSRAM is
the LTDC. So the inference/display interference is **not** two masters fighting over PSRAM; it is the
NPU's two 64-bit AXI masters saturating the shared AXI interconnect while the LTDC pulls each scanline
from *slow external PSRAM* across that same bus.

### The inference glitch — what it is and isn't

Characterised live over SWD:
- **Not an LTDC FIFO underrun** — `LTDC_S->ISR` (`0x58001038`) reads `0` throughout; FUWIF/FUIF are sticky and never set.
- **Not framebuffer-memory corruption** — a vblank trace ring showed the FB bytes, `CFBAR`, and layer regs all intact during inference.
- **It is live-scanout disturbance** — the LTDC's real-time PSRAM fetch is starved by the NPU's AXI flood, so the *fetched* pixels are transiently wrong even though the memory is correct. LTDC AXI QoS is already maxed (no register lever — confirmed vs AN4861 / RM0486).

### Coexistence design — the reference low-bus-traffic mechanical (implemented)

Rather than fight the contention with a gate, run-23 now matches how the reference (N6_EDGEAI_1) stays
glitch-free on the same silicon: keep the AXI bus quiet during inference.

1. **Double-buffer + line-event vsync swap** — LVGL renders the back buffer; the LTDC line-event ISR swaps `CFBAR` at vblank ([lvgl_port_n6.c](FSBL/Display/lvgl_port_n6.c)). Removes render-vs-read tearing.
2. **M55 D-cache ON + FB Non-Cacheable** — `SCB_EnableI/DCache()` right after `PSRAM_Mpu()` marks the FB `0x44` Normal-Non-Cacheable ([main.c](FSBL/Core/Src/main.c), [psram.c](FSBL/Core/Src/psram.c)). The CPU working set (embedding Gather, token loop, LVGL heap in cacheable AXISRAM1) is absorbed by cache instead of flooding the interconnect the LTDC scans from — the actual root-cause lever. FB stays non-cacheable so the LTDC never reads stale lines; `flush_cb`'s `__DSB` drains pixel writes before each swap.
3. **NPU I/O coherency** — with D-cache on, `mcu_cache_clean_range(in_buf)` after the embedding Gather cleans the CPU-filled input before the NPU reads it ([npu_query.c](FSBL/AI/npu_query.c)); the STAI runtime invalidates the output. Self-gating on `SCB->CCR.DC` (no-op if D-cache off).
4. **Per-epoch gate dropped** — the Layer-1 fetch gate around each `stai_network_run` is now off by default (`g_npu_gate=0`). Retained as a runtime A/B switch: set `g_npu_gate=1` via GDB `write_memory` to restore the old gate on-device without reflashing.

### Why the reference (N6_EDGEAI_1) doesn't show this

The reference **also** scans LTDC directly from PSRAM (`0x90000000` / `0x900C0000`) — it has **no**
AXISRAM scanout buffer — yet runs the NPU concurrently glitch-free with **no gate**. The difference was
bus pressure, not architecture: it keeps D-cache on and updates only a small dynamic region per frame,
whereas run-23 previously ran with D-cache **off**. Stage 1 closes that gap.

### Status — on-device verified 2026-07-01 (branch `lvgl`, commits `6ff1ea7` + `ae28c2d`)

Flashed via `load_and_run`, gate off. Verified over SWD + VCP:
- **Correctness (D-cache coherency holds):** boot self-test `3 + 4 = 7`, then the demo loop computes
  `12 - 5 = 7`, `6 * 7 = 42`, `8 / 2 = 4` — every multi-epoch NPU inference correct with D-cache ON.
- **Gate genuinely dropped:** `g_npu_gate=0`; the LTDC vblank trace ring shows `LayerCR.LEN=1` on
  every traced vblank (never gated off), `ISR=0` (no underrun ever), `CFBAR` valid with live
  double-buffer swaps. Display loop alive through inference (108 M loops, 226 flushes, 4378 vblanks).
- **Pending:** the final *visual* confirmation of scanout cleanliness during inference — by the
  glitch's nature (FB bytes stay correct, the LTDC *fetch* is transiently starved) it is invisible to
  SWD and requires a camera on the panel.

**Ruled out (2026-06-30):** silencing the per-token UART dialog (`g_npu_quiet=1`) did **not** fix the
glitch — confirming it is intrinsic per-*epoch* NPU↔LTDC AXI contention, which is why the fix targets
bus traffic (D-cache) rather than CPU churn duration.

## Debug notes (hard-won)

- **SWD desync wall:** after `SystemClock_Config`'s supply/clock switch the ST-Link↔target link
  desyncs — `continue`/`halt` across it fail. Single-step survives; FPB breakpoints still halt in HW.
- **debugFlag address moves** with the build — always read it from the ELF (`nm`), never hardcode.
- **`load_and_run` plants a BP at `main`** — remove it (or step past) before resume or it re-traps.
- **Display glitch ≠ underrun ≠ FB corruption** — read `LTDC_S->ISR` and the FB bytes live before
  theorising; the residual inference glitch is AXI-interconnect scanout starvation (see Display above).
