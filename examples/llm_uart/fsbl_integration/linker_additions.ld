/*
 * linker_additions.ld
 *
 * Add these two blocks to STM32N657XX_AXISRAM2_fsbl.ld to support on-device
 * AI inference from the FSBL:
 *
 *  1. Two new MEMORY regions  → add inside the MEMORY { ... } block
 *  2. Two new SECTIONS        → add inside the SECTIONS { ... } block
 *
 * ──────────────────────────────────────────────────────────────────────────
 * STEP 1 — add to MEMORY block (after the existing ROM / RAM entries)
 * ──────────────────────────────────────────────────────────────────────────
 *
 *   AI_RAM   (xrw) : ORIGIN = 0x34200000, LENGTH = 512K   <- AXISRAM3
 *   NOR_FLASH (rx) : ORIGIN = 0x70000000, LENGTH = 8M     <- external NOR (XSPI2)
 *
 * Origin notes:
 *   0x34200000 — AXISRAM3 start; already clock-enabled by FSBL USER CODE.
 *   0x70000000 — NOR flash start; already memory-mapped by BSP_XSPI_NOR_EnableMemoryMappedMode.
 *   LENGTH 512K is safe for the 331 KB activation buffer.
 *   LENGTH 8M covers the 1.32 MB weight blob with room to spare.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * STEP 2 — add to SECTIONS block (anywhere after .bss, before /DISCARD/)
 * ──────────────────────────────────────────────────────────────────────────
 */

  /* AI activation buffer — 331 KB, AXISRAM3 (no-load: zero-init at runtime) */
  .AI_RAM (NOLOAD) :
  {
    . = ALIGN(32);
    *(.AI_RAM)
    *(.AI_RAM*)
    . = ALIGN(32);
  } > AI_RAM

  /* Network weights — 1.32 MB INT8 blob, external NOR flash (XSPI2, memory-mapped) */
  .network_weights :
  {
    . = ALIGN(32);
    *network_data.o(.rodata .rodata*)
    *network_weights.o(.rodata .rodata*)
    . = ALIGN(32);
  } > NOR_FLASH

/*
 * ──────────────────────────────────────────────────────────────────────────
 * COMPLETE MEMORY block after modification:
 * ──────────────────────────────────────────────────────────────────────────
 *
 * MEMORY
 * {
 *   ROM      (xrw) : ORIGIN = 0x34180400, LENGTH = 255K   <- AXISRAM2: FSBL code
 *   RAM      (xrw) : ORIGIN = 0x341C0000, LENGTH = 256K   <- AXISRAM2: FSBL data
 *   AI_RAM   (xrw) : ORIGIN = 0x34200000, LENGTH = 512K   <- AXISRAM3: AI activations
 *   NOR_FLASH (rx) : ORIGIN = 0x70000000, LENGTH = 8M     <- XSPI2 NOR: AI weights
 * }
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Flash programming note:
 * ──────────────────────────────────────────────────────────────────────────
 *
 * The linker produces two load regions:
 *   - AXISRAM2 (0x34180400) : FSBL code + data  → flash to internal SRAM via SWD
 *   - NOR_FLASH (0x70000000): AI weights         → flash to XSPI2 NOR via STM32CubeProgrammer
 *
 * In dev mode:
 *   1. Flash weights once:  STM32_Programmer_CLI -c port=SWD -d fsbl.elf 0x70000000
 *   2. Flash FSBL each iteration: STM32_Programmer_CLI -c port=SWD -d fsbl.elf 0x34180400
 *   (Or use the generated .map file to get the exact weight section offset)
 *
 * The FSBL binary (AXISRAM2 region) is small (~200 KB) → fast to flash.
 * The weights (NOR flash) stay across FSBL re-flashes → flash once per model change.
 */
