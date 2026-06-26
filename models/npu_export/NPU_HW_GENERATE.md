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

## Status

- ✅ HW generation conclusive: 5 pure-HW epochs, weights in GDB-loadable AXISRAM2.
- ▢ On-device timing confirmation: deploy this `generated_hw`/`generated_sram`
  `network.c` (+ the `network_atonbuf.*.raw` weight blob at 0x34100000) into the
  run-23 FSBL, rebuild, and time one forward pass (expect sub-ms vs the 200 ms SW).
