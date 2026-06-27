# FSBL Integration — on-device LLM inference without booting Appli

> **⚠️ Legacy (superseded).** This guide documents the original Appli→FSBL integration of the Qwen2
> calculator test (`llm_fsbl.c` + `llm_test_fsbl.c`) used in the now-removed `run-22` project. The
> maintained deployment is **`STM32N6/AI_TO_NPU_1/run-23`**, where the device runs the grammar
> calculator **autonomously** on-chip (a host unified runner in device mode is a thin terminal) — see
> [`docs/STM32_NPU_DEPLOYMENT.md`](../../../docs/STM32_NPU_DEPLOYMENT.md). The integration *technique*
> below (linker/Makefile additions, FSBL `main.c` patch, SWD flash) still applies; only the project and
> the on-device application changed.

Run the Qwen2 calculator test entirely in the FSBL.  The FSBL halts before
`BOOT_Application()` — no Appli flash required.

**Dev mode advantage**: the FSBL binary is small (~200 KB internal SRAM), so
each iteration is a fast SWD flash.  The AI weights (1.32 MB in NOR flash)
only need to be flashed once per model change.

---

## Memory layout

| Region | Address | Size | Content |
|---|---|---|---|
| AXISRAM2 ROM | 0x34180400 | 255 KB | FSBL code + our C runtime |
| AXISRAM2 RAM | 0x341C0000 | 256 KB | FSBL data, context, states |
| AXISRAM3 | 0x34200000 | 512 KB | AI activation buffer (331 KB) |
| NOR flash (XSPI2) | 0x70000000 | 8 MB | AI weights (1.32 MB) |

AXISRAM3-6 are already clock-enabled by the existing FSBL USER CODE.
NOR flash is already memory-mapped by `BSP_XSPI_NOR_EnableMemoryMappedMode`.

---

## Step-by-step integration

### 1. Copy source files into the FSBL project

```
FSBL/Core/Src/   ← llm_fsbl.c  llm_test_fsbl.c
FSBL/Core/Inc/   ← llm_fsbl.h  llm_test_fsbl.h
```

Also ensure the Appli AI files are reachable (they're shared via relative paths):
```
Appli/AI/App/    ← network.c  network_data.c  network_weights.c
                    user_init.c  npu_init.c  npu_cache.c
                    network.h  network_data.h  user_init.h  npu_init.h
```

And our tokenizer (relative from FSBL):
```
Appli/Core/Src/  ← llm_tokenizer.c
Appli/Core/Inc/  ← llm_tokenizer.h
```

### 2. Modify the FSBL linker script

Edit `Makefile/FSBL/STM32N657XX_AXISRAM2_fsbl.ld`:

**In the MEMORY block**, add after `RAM`:
```ld
  AI_RAM    (xrw) : ORIGIN = 0x34200000, LENGTH = 512K
  NOR_FLASH  (rx) : ORIGIN = 0x70000000, LENGTH = 8M
```

**In the SECTIONS block**, add before `/DISCARD/`:
```ld
  .AI_RAM (NOLOAD) : {
    . = ALIGN(32);
    *(.AI_RAM)
    *(.AI_RAM*)
    . = ALIGN(32);
  } > AI_RAM

  .network_weights : {
    . = ALIGN(32);
    *network_data.o(.rodata .rodata*)
    *network_weights.o(.rodata .rodata*)
    . = ALIGN(32);
  } > NOR_FLASH
```

See `linker_additions.ld` for the complete annotated version.

### 3. Modify the FSBL Makefile

Edit `Makefile/FSBL/Makefile`.  Add the lines from `makefile_additions.mk`:
- C_SOURCES += (AI C files)
- C_INCLUDES += (AI headers)
- LIBDIR += (STAI lib path)
- LIBS += (STAI library)
- C_DEFS += -DLLM_FSBL_BUILD

### 4. Patch FSBL/Core/Src/main.c

Add the USER CODE sections from `fsbl_main_patch.c`:

**USER CODE BEGIN Includes** — add the two new includes.

**USER CODE BEGIN PV** — add `UART_HandleTypeDef huart1;`

**USER CODE BEGIN PFP** — add `void MX_USART1_UART_Init(void);`

**USER CODE BEGIN 2** — after the existing XSPI/PSRAM init, add:
```c
MX_USART1_UART_Init();
if (LLM_FSBL_Init() != 0) { while (1) {} }
LLM_TestCalcFSBL();
while (1) {}    /* ← prevents BOOT_Application() */
```

**After Error_Handler** — paste `MX_USART1_UART_Init()` function body.

### 5. Build

```bash
make -C Makefile/FSBL all
```

Expected: two output segments in the ELF:
- `0x34180400`: FSBL code (< 255 KB)
- `0x70000000`: AI weights (1.32 MB)

### 6. Flash (dev mode)

Flash both segments at once via STM32CubeProgrammer:

```bash
STM32_Programmer_CLI -c port=SWD freq=4000 -d Makefile/FSBL/build/run-23_FSBL.elf
```

The programmer reads the ELF load segments and flashes each to the correct address automatically.

For rapid iteration (weights unchanged):

```bash
# Re-flash only FSBL code (fast, ~200 KB SWD transfer)
STM32_Programmer_CLI -c port=SWD freq=4000 \
  -d Makefile/FSBL/build/run-23_FSBL.elf 0x34180400

# Weights stay in NOR flash from previous full flash
```

---

## Expected UART output (115200 baud, USART1)

```
================================================
  Qwen2 NPU Calc Test  [FSBL]  STM32N6570-DK
================================================
[01/10] "1+1"  (3 tokens)
        next: 11 → "+"
        continuation (8): "1+1..."
        [simple addition]

[02/10] "2*3"  (3 tokens)
        ...
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Linker overflow in ROM | Too much code in AXISRAM2 | Reduce OPT from -Og to -Os; exclude HAL modules not needed |
| Weights at wrong address | network_data.o not caught by `.network_weights` rule | Check `nm -n fsbl.elf` for rodata symbol addresses |
| UART silent | GPIO alt function wrong | Verify PA9/PA10 AF7 is correct for your PCB USART1 wiring |
| `user_stai_network_init` returns error | Weights not in NOR flash | Flash full ELF first so weights land at 0x70000000 |
| Activation buffer corrupt | AI_RAM region missing from linker | Verify AXISRAM3 section in .ld and ORIGIN correct |
