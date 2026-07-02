---
title: "Methodology — STM32N6 Neural-ART NPU Memory Pools (weights/activations) & Large-Model PSRAM+AXISRAM"
title_fr: "[TODO FR] Methodology — STM32N6 Neural-ART NPU Memory Pools"
permalink: /methodologies/methodology-npu-memory-pools/
permalink_fr: /fr/methodologies/methodology-npu-memory-pools/
pub_id: "Guide — NPU Memory Pools"
version: v1
pub_date: "July 2026"
keywords: "STM32N6, Neural-ART, NPU, mpool, weights, activations, PSRAM, AXISRAM, stedgeai, atonn"
---
# STM32N6 Neural-ART — NPU Memory Pools & Large-Model PSRAM+AXISRAM Design

## Purpose

Authoritative reference for how `stedgeai`/atonn maps **weights** and **activations** to memory pools
(`.mpool`) on STM32N6, and how to design for **large models** that outgrow on-chip AXISRAM by combining
AXISRAM + external PSRAM. Captured during the run-23 nocode work when the single-pool SRAM network's
**weights and activations overlapped** (`Conv2D_6_weights` at offset 0 shared its pool with activations
→ the activation overwrote the weights mid-inference → NPU emitted garbage/empty grammar).

**Origin of the official section below:** answered by the ST provider AI sidekick against official ST
documentation (RM0486, DS14791, AN4861, the STM32N6-GettingStarted repos, ST Community). Sources listed
at the end. Recorded **integrally** for repo reference.

---

## Official ST guidance (integral)

> **Summary**
>
> Yes — the STM32N6 Neural-ART documentation supports using external PSRAM as a memory pool for
> activations when it is defined in the .mpool, while weights and activations remain separate memory
> classes managed by the memory-pool mapping.
>
> What I could not confirm from the available official material is that atonn will automatically spill
> activations from AXISRAM to PSRAM with a documented hot/cold policy, or that there is a documented
> `--mpool-params` rule for steering that placement. The official material does confirm the explicit
> pool-selection options `--weights-pool`, `--activations-pool`, and support for
> `--enable-virtual-mem-pools`.
>
> **1) AXISRAM + PSRAM tiering for activations**
> The .mpool file maps activations and weights; activations are read/write buffers for intermediate
> results, weights are read-only model data. External PSRAM can be used for activations if mapped in
> the .mpool. STM32N6 provides 4.2 MB of contiguous embedded RAM; AXISRAM3 to AXISRAM6 are NPURAM
> optimized for Neural-ART access. So defining external PSRAM in the .mpool is a documented way to
> extend activation storage beyond on-chip AXISRAM. Docs also say "hot tensors" should be prioritized
> for fast internal RAM and colder tensors can be placed in slower external memory (performance vs
> capacity trade-off). However: no official statement that atonn auto-spills AXISRAM→PSRAM once full,
> no official description of the hot/cold heuristic, and no `--mpool-params` tiering rule found.
>
> **2) Documented pool-selection options**
> Explicitly mentioned: `--weights-pool`, `--activations-pool`, `--enable-virtual-mem-pools`. The
> official profile example shows `memory_pool` plus options including `--enable-virtual-mem-pools`,
> `--cache-maintenance`, `--Ocache-opt`. Documented knobs: define regions in `.mpool`, select pools
> explicitly with `--weights-pool`/`--activations-pool`, optionally `--enable-virtual-mem-pools`.
> Not found: whether `.mpool` rights alone drive placement for every tensor class, or a per-pool
> usage/role field beyond the pool-selection options.
>
> **3) Cache and coherency for PSRAM activations**
> STM32N6 includes an AXI cache (CACHEAXI) to improve data traffic, especially for the NPU accessing
> external memories such as PSRAM and xSPI. When not used for caching, CACHEAXI can be configured as
> additional RAM, accessible by NPU, Cortex-M55, and other AXI masters. Cache maintenance operations
> are necessary for coherency and must be handled carefully; profile options include
> `--cache-maintenance` and `--Ocache-opt`. So NPU accesses to external PSRAM can benefit from
> CACHEAXI (external-memory activations are not inherently uncached), and cache maintenance matters.
> Not confirmed: the exact boundary between what `--cache-maintenance`/`--Ocache-opt` handles
> automatically vs what the application must clean/invalidate around `stai_network_run`.
>
> **4) PSRAM shared between LTDC framebuffer and NPU activations**
> The LTDC application note: display bandwidth must be budgeted; if framebuffer memory is accessed by
> other masters (CPU or NPU), graphics performance can be impacted and FIFO underrun can occur. On
> STM32N6xx, LTDC AXI QoS is set to maximum. The NPU uses two 64-bit AXI master ports; Neural-ART
> memory is software-configurable shared system memory. So: they CAN share PSRAM in the address-map
> sense as long as `.mpool`, linker, and framebuffer allocation do not overlap — BUT sharing
> framebuffer memory bandwidth with NPU traffic is a real contention risk and can cause display
> degradation or FIFO underrun. Docs do not say the framebuffer must move to AXISRAM, but clearly warn
> about contention when display fetches and other masters use the same memory path.
>
> **5) Weights for large models**
> Official getting-started material: weights are usually stored in external flash, activations in RAM.
> So for oversized weights: prefer external flash as the normal location; activations remain the RAM
> consumer. Not found: an official XIP-from-NOR vs memcpy-to-AXISRAM comparison, or a recommendation to
> put oversized weights in PSRAM instead of flash.
>
> **6) Reference examples**
> The STM32N6 getting-started projects provide example `.mpool` files for NUCLEO-N657X0-Q and
> STM32N6570-DK, covering object detection, face detection, pose estimation, image classification,
> ReID, instance segmentation, semantic segmentation. These can leverage external PSRAM for larger
> models; buffer sizes/lifetimes are in the generated `st_ai_output/<model>_c_info.json` (ST Edge AI
> Developer Cloud / custom Netron view). No single project named as THE best reference, but the
> STM32N6570-DK getting-started projects for the larger detection workloads are the documented family
> that include board-specific `.mpool` examples.
>
> **Bottom line**
> - Use `.mpool` to add PSRAM as activation capacity when AXISRAM is not enough.
> - Keep activations in AXISRAM3-6 where possible (optimized for Neural-ART).
> - Use explicit pool-selection options (`--weights-pool`, `--activations-pool`) and, if needed,
>   `--enable-virtual-mem-pools`.
> - Assume PSRAM shared with LTDC is a **contention risk**, not a free overflow area — LTDC scanout
>   can be disturbed by other masters using the same memory bandwidth.
> - Weights are normally placed in **external flash**, not in activation RAM.
>
> **Sources:** pres11_STM32N6_MCUs_Rev1.0.pdf p18 · DS14791_Rev9 p14,p23 · DB5274_Rev2 p2 ·
> AN4861_Rev7 p27,p32 · RM0486_Rev4 p321,p1028,p159,p160,p289,p468,p1029 · STCommunity "AI app from
> scratch on STM32N6570-DK" · STM32N6-GettingStarted-{FaceDetection,ImageClassification,
> InstanceSegmentation,ObjectDetection,PoseEstimation,ReID,SemanticSegmentation} Neural-ART docs ·
> STCommunity {very-bad-performances-n657, ltdc-glitches-when-npu, code-executing-in-external-flash}.

---

## run-23 adaptation — design decisions

Applying the guidance to the run-23 FSBL (nocode calculator + LTDC + PSRAM framebuffer):

### 1. Weights/activations split — the overlap fix (DONE)
- **Root cause:** single-pool `stm32n6_sram_weights.mpool` (one `cpuRAM1`/AXISRAM1 @0x34064000, 624K,
  `ACC_WRITE`) interleaved weights + activations → `Conv2D_6_weights` (offset 0) shared its bytes with
  activations → overwrite → garbage weights → empty grammar.
- **Fix:** `network_regen/stm32n6_twopool.mpool` — **weights → AXISRAM1 `cpuRAM1` `ACC_READ`**,
  **activations → AXISRAM3-6 (`npuRAM3..6`) `ACC_WRITE`**. atonn routes by **pool rights** in our
  ST Edge AI **4.0** (verified: weights land 0x34064000-0x340c…, activations 0x342…-0x343…).
- **Note on the CLI selectors:** `--weights-pool`/`--activations-pool` are documented, but **atonn 4.0
  here REJECTS them** (`arguments were not expected`). So on this toolchain the routing is driven by
  `.mpool` **rights** (`ACC_READ` = weights, `ACC_WRITE` = activations), matching the working optionB.
- **The split cannot reuse weight space** (the single-pool did) → the calc TCN needs **~1.7 MB** of
  activations across AXISRAM3-6. Sizing the AXISRAM `ACC_WRITE` pools is mandatory (added AXISRAM4,5,6).

### 2. Large-model activation overflow → PSRAM (design, per ST)
- Add `hyperRAM` (xSPI1 @0x90000000, `ACC_WRITE`, 32MB) as an `.mpool` pool → documented valid overflow
  for activations beyond AXISRAM3-6.
- **No auto AXISRAM→PSRAM spill heuristic is documented** → **order/size the pools explicitly**: AXISRAM
  first (fast, NPURAM), PSRAM last (overflow, slow). Keep hot tensors in AXISRAM.

### 3. ⚠ PSRAM ↔ LTDC framebuffer contention — the run-23 hard constraint
- run-23 today puts the **LTDC framebuffer in PSRAM** (0x90000000). ST confirms **sharing PSRAM
  bandwidth between LTDC scanout and NPU activations is a real contention risk → FIFO underrun /
  display glitches**. This is the SAME failure already seen in run-23 (`[[reference_stm32n6_ltdc_npu_psram_coexistence]]`:
  inference glitch = LTDC live-scanout starved by the NPU AXI flood).
- **Decision for large models:** do **NOT** let NPU activations and the FB share the same PSRAM path.
  Options, in order of preference:
  1. **Move the FB to on-chip SRAM** (the reference 100%-SRAM double-buffer) so PSRAM is free for NPU
     activations. (Frees the contention entirely.)
  2. If both must be in PSRAM: **partition by address range** in the `.mpool`/linker (non-overlapping)
     AND keep the per-epoch LTDC gate / dirty-region updates as the contention stopgap.
- For the **small calc** (fits AXISRAM3-6), PSRAM is NOT used by the NPU → no new contention; the FB
  stays in PSRAM as-is.

### 4. Weights placement for large models
- Small calc: **memcpy NOR (0x70200000) → AXISRAM1 (0x34064000)** at boot (fast on-chip reads; avoids
  the XIP-from-NOR "zero logits" issue seen earlier).
- Large models whose weights exceed AXISRAM1: put the **weights pool in external flash (xSPI2 NOR,
  `ACC_READ`, XIP)** — the documented normal location. (ST does not compare XIP vs memcpy; validate on
  HW — earlier run-23 XIP-from-NOR gave zero logits, so prefer on-chip where it fits.)

### 5. Cache / coherency
- Enable **CACHEAXI** for NPU external-memory (PSRAM/NOR) accesses (already done in run-23:
  `npu_cache_enable()`). Keep `--cache-maintenance --Ocache-opt` in the neural-art profile.
- The exact runtime-vs-app cache-maintenance boundary is **not documented** → keep run-23's explicit
  D-cache clean of app-filled input buffers (`mcu_cache_clean_range` in `npu_query.c`) and validate NPU
  output coherency on HW.

### 6. Inspect the layout
- After `stedgeai generate`, inspect `st_ai_output/<model>_c_info.json` (buffer sizes + lifetimes) and
  the generated `network.c` `addr_base`/`offset_start` to confirm weights ∈ AXISRAM1 and activations ∈
  AXISRAM3-6(+PSRAM), with **no overlap**.

---

## Related
- `network_regen/stm32n6_twopool.mpool` · `network_regen/user_neuralart_twopool.json` — the two-pool config.
- `[[reference_stm32n6_ltdc_npu_psram_coexistence]]` — the PSRAM/LTDC/NPU contention finding.
- `network_regen/README.md` — the host-model → NPU regeneration pipeline.
