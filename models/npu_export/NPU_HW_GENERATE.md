# Generating the NPU-**hardware** model (Neural-ART), not the SW lite runtime

## The problem (perf)

`stedgeai generate --target stm32n6 --c-api st-ai` (our first run) silently used
the **lite runtime** (`use-lite-runtime`): the conv layers are emitted as **software
int8 kernels** (`smul_s8_s8`) that run on the Cortex-M55, **not** on the Neural-ART
NPU. On device a single 15.6 M-MACC forward pass took **~200 ms** (software-fallback
speed). The generated `network.c` is a plain operator-descriptor graph — **no
`pure_hw` ATON epoch blocks**.

## The fix — `--st-neural-art` (launch the Neural-Art / atonn compiler)

```bash
stedgeai generate \
  --model  models/generated/convolutional/model_calculator_tcn_version_1/model_npu_int8.onnx \
  --target stm32n6 \
  --st-neural-art "sram_weights@STM32N6/AI_TO_NPU_1/run-23/Makefile/FSBL/user_neuralart_sram.json" \
  --name   network \
  --output models/npu_export/model_calculator_tcn_version_1/generated_hw
```

Result: **5 epochs, all `pure HW`** (0 SW, 0 hybrid). The conv body runs on the
Neural-ART. (No atonn segfault hit on this model — the documented caveat did not
trigger with this stedgeai 4.0 build.)

## Weights placement — keep them in **GDB-loadable SRAM** (`--memory-pool` via a profile)

By default the atonn puts the weights in **XSPI NOR flash @ 0x71000000** (`octoFlash`,
`use4initializers`), which `load_and_run` cannot program. To keep the dev flow
(GDB load to RAM), override the memory pool **through an `--st-neural-art` profile**
(the bare `--memory-pool` flag is ignored — it must come from the profile JSON):

- `user_neuralart_sram.json` — profile `sram_weights` → `memory_pool: stm32n6_sram_weights.mpool`.
- `stm32n6_sram_weights.mpool` — only `cpuRAM2` (AXISRAM2 @ 0x34100000, 512 KB,
  **boot-ROM-active = GDB-loadable**) + a small `npuRAM3` (AXISRAM3, for activations,
  un-parked by the FSBL at runtime). No `octoFlash` → the 483 KB of weights are
  forced into `cpuRAM2 @ 0x34100000`.

`network.c` then has `.addr_base = 0x34100000` for the conv weights — the same
GDB-loadable SRAM the software model already used. (Tune `npuRAM3` size so the whole
weights blob lands in `cpuRAM2`; a few KB can otherwise spill into the parked NPU RAM.)

## Deployment — flash-copy (dev=0 + dev=1 unified)  [FINAL, device-validated]

The GDB-loadable-SRAM placement above is **dev=1 only**: the weights ride inside the FSBL
image, so a dev=0 device (boot-from-flash, no probe) has **no weights on the system**. The
shipped solution keeps the weights in **persistent flash, copied to SRAM at boot** — ONE
path for both dev modes. (Do **not** try XIP read-in-place from XSPI: the Neural-ART epoch
stalls reading XSPI-resident weights — confirmed dead end. Copy to AXISRAM and run from there.)

1. **Weights blob** (`network_atonbuf.AXISRAM1.raw` ≙ `network_weights.bin`, 526 496 B) is
   flashed ONCE to **XSPI2 NOR @0x70200000** (separate from the FSBL image):
   ```bash
   STM32_Programmer_CLI -c port=SWD mode=UR \
     -el MX66UW1G45G_STM32N6570-DK.stldr -w network_weights.bin 0x70200000 -v
   ```
2. **At boot** the FSBL (after `MX_EXTMEM_MANAGER_Init` sets up the NOR) maps it, copies the
   blob into **AXISRAM1 @0x34064000** (the `.addr_base` the HW `network.c` reads), and cleans
   the M55 D-cache so the Neural-ART sees fresh weights:
   ```c
   (void)EXTMEM_MemoryMappedMode(EXTMEMORY_1, EXTMEM_ENABLE);
   memcpy((void *)0x34064000, (const void *)0x70200000, 526496);
   SCB_CleanDCache_by_Addr((uint32_t *)0x34064000, 526496);
   ```
3. `.ai_weights` is **NOLOAD** (linker) and the `network_weights.o` bake is removed (Makefile)
   → FSBL image **244 KB** (vs 1.4 MB baked), so dev=0 boots from a small signed image.

- **dev=1**: `load_and_run` loads only the 244 KB code (weights are NOLOAD); the boot copy feeds
  the NPU from the flashed NOR. Clear the real `debugFlag` (read its address from the ELF — it
  moves with the build) before release, and drop the `load_and_run` BP at `main` or the resume
  re-traps it.
- **dev=0**: sign the FSBL, then flash it to **0x70000000**; boot-ROM loads it → copy → NPU,
  fully autonomous (no probe):
  ```bash
  STM32_SigningTool_CLI -s -bin run-23_FSBL.bin -nk -of 0x80000000 -t fsbl \
    -hv 2.3 -align -la 0x34180000 -o run-23_FSBL-trusted.bin -dump run-23_FSBL-trusted.bin
  STM32_Programmer_CLI -c port=SWD mode=UR \
    -el MX66UW1G45G_STM32N6570-DK.stldr -d run-23_FSBL-trusted.bin 0x70000000
  ```

## Status

- ✅ HW generation: 5 pure-HW epochs (`--st-neural-art`), conv body on the Neural-ART.
- ✅ Weights deployment: flash-copy **XSPI2 NOR 0x70200000 → AXISRAM1 0x34064000** (dev=0 + dev=1).
- ✅ Device-validated: autonomous on-chip NPU grammar calculator over UART — `3 + 4 = 7` and
  `6 * 7 = 42` (5 rule-queries/eval via the Neural-ART oracle), in **both dev=1 and dev=0**.
