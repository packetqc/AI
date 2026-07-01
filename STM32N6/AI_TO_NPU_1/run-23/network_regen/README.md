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
| NPU pool | AXISRAM1 `0x34064000` (spills to AXISRAM5-6 via virtual pools) | AXISRAM3 + AXISRAM4 + AXISRAM7, ~1152 K |
| AXISRAM1 | NPU (blocked) | **FREE → FB front** |
| AXISRAM5-6 | **NPU (spill) — blocked** | **FREE → FB back** |
| AXISRAM7 (CACHEAXI) | NPU weight cache | NPU data (no cache needed — weights already on-chip) |
| Weights `memcpy` dst | `0x34064000` | `0x34200000` (new pool base) |

> **Why 3 banks:** the compiled network is ~1 MB (weights 526 K + activations 514 K) and the *current*
> build already spreads into AXISRAM5-6 (verify with `verify_optionB.sh` — it lists NPU addresses inside
> the FB banks today). AXISRAM3+4 (896 K) alone is too small, so the option-B pool adds AXISRAM7. Since
> the weights then live in fast on-chip SRAM, the CACHEAXI is unnecessary — **skip `npu_cache_enable()`**
> (the `{ extern void npu_cache_enable(void); npu_cache_enable(); }` line in main.c's NPU block) so SRAM7
> is free for network data instead of being claimed as the NPU cache.

## Re-generation — PROVEN command (run autonomously, 2026-07-01)

```bash
ONNX=/opt/DEV/AI/models/npu_export/model_calculator_tcn_version_1/generated_sram/model_npu_int8_OE_3_3_1.onnx
/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai generate --target stm32n6 --name network -m "$ONNX" \
  --st-neural-art "optionB_sram@network_regen/user_neuralart_optionB.json" \
  --c-api st-ai --enable-epoch-controller --output <outdir> --workspace <outdir>/ws --no-report
```

Result (verified): the network lands in **AXISRAM3-4 + npuCACHE(7) + vencRAM(8) + NOR**, with **zero
addresses in AXISRAM1 or AXISRAM5-6** (`verify_optionB.sh` passes on the generated files) and PSRAM
unused. Generated sources are staged in [`generated/`](generated/); the ~1.9 MB NOR blob
(`atonbuf.xSPI2` → `atonbuf_xSPI2.bin`) is gitignored (regenerate it).

## Integration steps (the destructive part — commit first)

1. Copy `generated/{network.c,network.h,stai_network.c,stai_network.h,network_ecblobs.h}` into
   `FSBL/AI/`. (Keep the current `network_embed.c/.h`, `network_tokens.c/.h`, `npu_query.*` — the I/O
   interface is unchanged: `IN=8192`, `OUT=11968`.)
2. Run **`/regen-fix`** (restores sourceEntries etc.).
3. **`FSBL/Core/Src/main.c`: DELETE the weights `memcpy` to `0x34064000`** (and its
   `SCB_CleanDCache_by_Addr`) — option B keeps the weights XIP in NOR (memory-mapped), no SRAM copy.
   Ensure `EXTMEM_MemoryMappedMode(EXTMEMORY_1, ENABLE)` stays (it does).
4. **Flash the NOR blob** `generated/atonbuf_xSPI2.bin` to its XSPI2 base (per the generated
   `network_ecblobs.h` / the `0x70……` base in `network.c`) using the run-23 flash path.
5. Linker: the activation pool now spans AXISRAM3-4-7-8 — confirm `AI_RAM`/`.AI_RAM` covers it (the
   stai context allocates there at runtime).
6. `FSBL/Core/Inc/fb_layout.h`: set **`FB_IN_SRAM 1`**. `fb_sram_enable()` handles the FB banks' MPU.
7. Build → `network_regen/verify_optionB.sh` (all pass) → flash → **camera A/B, gate off** → full-color
   double-buffer, glitch-free.

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
