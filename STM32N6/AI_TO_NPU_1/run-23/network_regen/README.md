# Host model → NPU path — memory-layout re-generation (option B)

Persisted method for regenerating the run-23 Neural-ART network so its memory lands **off AXISRAM1**,
freeing that bank (and AXISRAM5-6) for the 100%-SRAM LTDC double-buffer. This is the "host model →
NPU path" for the reference LVGL-bare-metal memory architecture. See [`../MEMORY_MAP.md`](../MEMORY_MAP.md).

## The pipeline

```
host model (ONNX)  ──►  ST Edge AI / atonn  ──►  network.c + network_data.c + network_weights.bin
   model_cnn_calc.onnx      (--st-neural-art)         (baked HW epochs + weight blob)
                                │
                          --load-mpool-file  ◄── THIS is the memory-layout knob
                                │
                                ▼
   FSBL: memcpy(weights_dst ⟵ NOR 0x70200000)  +  stai_network_run()  ──►  NPU
```

The **mempool file** given to atonn (`--load-mpool-file`) decides which AXISRAM bank(s) the network's
weights + activations occupy. run-23 shipped with [`../Makefile/FSBL/stm32n6_sram_weights.mpool`](../Makefile/FSBL/stm32n6_sram_weights.mpool)
= a single pool at **AXISRAM1 `0x34064000` (624 K)** → that is why AXISRAM1 is unavailable for the FB.

## Current vs option-B layout

| | Legacy (`stm32n6_sram_weights.mpool`) | **Option B (`stm32n6_optionB_axisram34.mpool`)** |
|---|---|---|
| NPU pool | AXISRAM1 `0x34064000`, 624 K | AXISRAM3 `0x34200000` + AXISRAM4 `0x34270000`, 896 K |
| AXISRAM1 | NPU (blocked) | **FREE → FB front** |
| AXISRAM5-6 | free | FB back |
| Weights `memcpy` dst | `0x34064000` | `0x34200000` (new pool base) |

## Re-generation steps (user runs ST Edge AI / AI Studio)

1. Regenerate the network against the option-B pool:
   ```
   stedgeai generate --model model_cnn_calc.onnx --target stm32n6 --st-neural-art default \
     --load-mpool-file network_regen/stm32n6_optionB_axisram34.mpool --enable-virtual-mem-pools
   ```
   (Or set the mem-pool file in AI Studio's project to `stm32n6_optionB_axisram34.mpool` and regenerate.)
2. Copy the regenerated `network.c`, `network_data.c`, `network_details.h`, `network_weights.bin`,
   `network_atonbuf_xSPI2.bin` into `FSBL/AI/`.
3. Run **`/regen-fix`** (mandatory after any CubeMX/AI-Studio regen — restores sourceEntries etc.).
4. **Update the weights `memcpy` destination** in `FSBL/Core/Src/main.c` from `0x34064000` to the new
   pool base (`0x34200000`); update the `SCB_CleanDCache_by_Addr` range to match.
5. **Re-flash the weight blob** to NOR `0x70200000` if `network_weights.bin` changed size/content.

## Then flip the framebuffer to SRAM

Once AXISRAM1 is free (steps above), enable the reference layout:

- `FSBL/Core/Inc/fb_layout.h`: set **`FB_IN_SRAM 1`** (front → AXISRAM1 `0x34000000`, back → AXISRAM5-6
  `0x342E0000`). `fb_sram_enable()` (already wired) marks those banks MPU Non-Cacheable.
- Build, flash, camera A/B with the gate off (`g_npu_gate=0`) → expect clean scanout through inference
  at full color. Keep the gate as the runtime fallback.

## Verification (invariants for the reference)

- Boot self-test + demo still compute correctly (weights relocated correctly): `3 + 4 = 7`, etc.
- No address in the regenerated `network*.c` falls inside AXISRAM1 (`0x34000000-0x340FFFFF`) or
  AXISRAM5-6 (`0x342E0000-0x343BFFFF`) — those are now FB-only. Quick check:
  ```
  grep -oE "0x3400[0-9A-Fa-f]{4}|0x340[0-9A-Fa-f]{5}|0x342[EF][0-9A-Fa-f]{4}|0x343[0-9AB][0-9A-Fa-f]{4}" FSBL/AI/network*.c
  ```
  (must be empty)
- On-camera: no whole-screen scanline corruption during inference.
