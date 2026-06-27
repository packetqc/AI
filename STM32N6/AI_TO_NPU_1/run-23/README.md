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

## Debug notes (hard-won)

- **SWD desync wall:** after `SystemClock_Config`'s supply/clock switch the ST-Link↔target link
  desyncs — `continue`/`halt` across it fail. Single-step survives; FPB breakpoints still halt in HW.
- **debugFlag address moves** with the build — always read it from the ELF (`nm`), never hardcode.
- **`load_and_run` plants a BP at `main`** — remove it (or step past) before resume or it re-traps.
