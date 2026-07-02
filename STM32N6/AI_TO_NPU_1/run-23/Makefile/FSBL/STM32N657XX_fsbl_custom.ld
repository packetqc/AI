/*
******************************************************************************
**
** @file        : STM32N657XX_AXISRAM2_fsbl.ld
**
** @author      : GPM Application Team
**
** @brief       : Linker script for STM32N657XX Device from STM32N6 series
**                      512 KBytes RAM
**
**                Set heap size, stack size and stack location according
**                to application requirements.
**
**                Set memory bank area and size if external memory is used
**
**  Target      : STMicroelectronics STM32
**
**  Distribution: The file is distributed as is, without any warranty
**                of any kind.
**
******************************************************************************
** @attention
**
** Copyright (c) 2023 STMicroelectronics.
** All rights reserved.
**
** This software is licensed under terms that can be found in the LICENSE file
** in the root directory of this software component.
** If no LICENSE file comes with this software, it is provided AS-IS.
**
******************************************************************************
*/

/* Entry Point */
ENTRY(Reset_Handler)


/* Memories definition */
MEMORY
{
  ROM          (xrw) : ORIGIN = 0x34180400, LENGTH = 464K  /* AXISRAM2: FSBL code+rodata (+4K for the inline SRAM network.c) */
  RAM          (xrw) : ORIGIN = 0x341F4400, LENGTH = 50K   /* AXISRAM2: FSBL data+bss+stack (shifted +4K for the bigger ROM) */
  WEIGHTS_RAM  (xrw) : ORIGIN = 0x34064000, LENGTH = 624K  /* AXISRAM1 (cpuRAM1): boot-ROM-active (GDB-loadable) — atonn HW weights blob baked @0x34064000 */
  AI_RAM       (xrw) : ORIGIN = 0x34201000, LENGTH = 508K  /* AXISRAM3: AI activations (unused by the single-pool SRAM network; nudged +4K for RAM) */
  THREADX_RAM  (xrw) : ORIGIN = 0x34140000, LENGTH = 256K  /* low AXISRAM2 AFTER the LVGL heap (lv_conf LV_MEM_ADR=0x34100000, 256K → ends 0x34140000) and before the FSBL ROM @0x34180400: ThreadX byte pool + kernel .bss. MUST NOT overlap the LVGL heap (that corrupts s_led → get_local_style HardFault). */
  NOR_FLASH    (rx)  : ORIGIN = 0x70000000, LENGTH = 8M    /* XSPI2 NOR: AI weights (memory-mapped) */
  WEIGHTS_SRAM (rx)  : ORIGIN = 0x70000000, LENGTH = 8M    /* alias: atonn-generated section name */
}
/* Highest address of the user mode stack */
_estack = ORIGIN(RAM) + LENGTH(RAM); /* end of "RAM" Ram type memory */
_sstack = _estack - _Min_Stack_Size;

_Min_Heap_Size = 0x1000;      /* heap (STL containers: map/vector/string) */
_Min_Stack_Size = 0x8000; /* stack: C++ grammar-runner recursion + STL frames (was 0x800 -> STKOF) */

/* Sections */
SECTIONS
{
  /* The startup code into "RAM" Ram type memory */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >ROM

  /* The program code and other data into "RAM" Ram type memory */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)
    *(.RamFunc)        /* .RamFunc sections */
    *(.RamFunc*)       /* .RamFunc* sections */

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >ROM

  /* vanilla run-23: weights stay in flash @ 0x71000000 (already flashed); reference by symbol */
  PROVIDE(_mem_pool_xSPI2_network = 0x71000000);

  /* AI weights (~481 KB) → free lower AXISRAM2 @0x34100000: boot-ROM-active (so GDB load_and_run
   * can write them — the NPU RAMs AXISRAM3-6 are parked until the FSBL un-parks them, too late for
   * the loader), NPU-readable (AXI fabric), below the FSBL code. MUST precede .rodata so it captures
   * .rodata.g_network_weights_array before the generic *(.rodata*) rule. No RISAF/parking/boot-copy. */
  /* NOLOAD: weights are NOT baked into the FSBL image — flashed once to XSPI2 NOR @0x70200000 and
   * copied to this VMA (0x34064000) at boot by main.c (dev=0/dev=1 unified). Reserves the region;
   * keeps the FSBL image small so dev=0 boots cleanly from a signed image. */
  .ai_weights (NOLOAD) :
  {
    . = ALIGN(8);
    KEEP(*(.rodata.g_network_weights_array))
    KEEP(*(.ai_weights*))
    . = ALIGN(8);
  } > WEIGHTS_RAM

  /* Constant data into "RAM" Ram type memory */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >ROM

  .ARM.extab   (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
   {
    . = ALIGN(4);
    *(.ARM.extab* .gnu.linkonce.armextab.*)
    . = ALIGN(4);
  } >ROM

  .ARM (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
   {
    . = ALIGN(4);
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
    . = ALIGN(4);
  } >ROM

  .preinit_array     (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
  } >ROM

  .init_array (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
  } >ROM

  .fini_array (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
    . = ALIGN(4);
  } >ROM

  /* Used by the startup to initialize data */
  .gnu.sgstubs :
  {
    _sNSCVeneer = .;
    *(.gnu.sgstubs*)   /* Secure Gateway stubs */
    . = ALIGN(32);
    FILL(0x00);
  } >ROM
  _eNSCVeneer = .;
  /* Uninitialized data section into "RAM" Ram type memory */
  _sidata = LOADADDR(.data);

  /* Initialized data sections into "RAM" Ram type memory */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);

  } >RAM AT> ROM

  /* Initialized TLS data section */
  .tdata : ALIGN(4)
  {
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */
    PROVIDE(__data_end = .);
    PROVIDE(__tdata_end = .);
  } >RAM AT> ROM

  PROVIDE( __tdata_start = ADDR(.tdata) );
  PROVIDE( __tdata_size = __tdata_end - __tdata_start );

  PROVIDE( __data_start = ADDR(.data) );
  PROVIDE( __data_size = __data_end - __data_start );

  PROVIDE( __tdata_source = LOADADDR(.tdata) );
  PROVIDE( __tdata_source_end = LOADADDR(.tdata) + SIZEOF(.tdata) );
  PROVIDE( __tdata_source_size = __tdata_source_end - __tdata_source );

  PROVIDE( __data_source = LOADADDR(.data) );
  PROVIDE( __data_source_end = __tdata_source_end );
  PROVIDE( __data_source_size = __data_source_end - __data_source );


  .noncacheable :
  {
    . = ALIGN(8);
    __snoncacheable = .;/* create symbol for start of section */
    KEEP(*(.noncacheable))
    . = ALIGN(8);
    __enoncacheable = .;  /* create symbol for end of section */
  } > RAM


  /* Uninitialized TLS data section */
  .tbss (NOLOAD) : ALIGN(4)
  {
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.tbss .tbss.*)
    . = ALIGN(4);
    PROVIDE( __tbss_end = . );
  } >RAM

  PROVIDE( __tbss_start = ADDR(.tbss) );
  PROVIDE( __tbss_size = __tbss_end - __tbss_start );
  PROVIDE( __tbss_offset = ADDR(.tbss) - ADDR(.tdata) );

  PROVIDE( __tls_base = __tdata_start );
  PROVIDE( __tls_end = __tbss_end );
  PROVIDE( __tls_size = __tls_end - __tls_base );
  PROVIDE( __tls_align = MAX(ALIGNOF(.tdata), ALIGNOF(.tbss)) );
  PROVIDE( __tls_size_align = (__tls_size + __tls_align - 1) & ~(__tls_align - 1) );
  PROVIDE( __arm32_tls_tcb_offset = MAX(8, __tls_align) );
  PROVIDE( __arm64_tls_tcb_offset = MAX(16, __tls_align) );

  /* ThreadX (Azure RTOS) memory → free low AXISRAM2, off the tight FSBL RAM region. Placed BEFORE .bss
   * so the ThreadX kernel + app .bss/COMMON match HERE first and the generic .bss below takes the rest.
   * NOLOAD: zero-init by ThreadX/startup, nothing to load. Only present in the USE_THREADX=1 build; the
   * bare-metal build emits no .threadx_pool/tx_*.o so these sections are empty. */
  .threadx_pool (NOLOAD) : ALIGN(8)
  {
    *(.threadx_pool)
  } >THREADX_RAM

  .threadx_bss (NOLOAD) : ALIGN(8)
  {
    *tx_*.o(.bss .bss* COMMON)
    *app_threadx.o(.bss .bss* COMMON)
    *app_azure_rtos.o(.bss .bss* COMMON)
  } >THREADX_RAM

  .bss (NOLOAD) : ALIGN(4)
  {
    /* This is used by the startup in order to initialize the .bss section */
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
      PROVIDE( __bss_end = .);
  } >RAM
  PROVIDE( __non_tls_bss_start = ADDR(.bss) );

  PROVIDE( __bss_start = __tbss_start );
  PROVIDE( __bss_size = __bss_end - __bss_start );


  /* User_heap_stack section, used to check that there is enough "RAM" Ram  type memory left */
  ._user_heap_stack (NOLOAD) :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >RAM

  /* AI activation buffer — 331 KB, AXISRAM3 (NOLOAD: zero-init at runtime) */
  .AI_RAM (NOLOAD) :
  {
    . = ALIGN(32);
    *(.AI_RAM)
    *(.AI_RAM*)
    . = ALIGN(32);
  } > AI_RAM

  /* AI weights — 1.38 MB in NOR flash, memory-mapped at 0x70000000.
   * WEIGHTS_SOURCE_ADDRESS=0x70000000 tells user_stai_network_init where to find them. */
  .AI_EXTFLASH :
  {
    . = ALIGN(32);
    *(.AI_EXTFLASH)
    *(.AI_EXTFLASH*)
    . = ALIGN(32);
  } > WEIGHTS_SRAM

  /* Remove information from the compiler libraries */
  /DISCARD/ :
  {
    libc.a:* ( * )
    libm.a:* ( * )
    libgcc.a:* ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
