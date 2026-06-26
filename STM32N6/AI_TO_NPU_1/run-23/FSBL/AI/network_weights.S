/* network_weights.S — embed the atonn HW weights blob into the FSBL image.
 *
 * The Neural-ART HW network bakes its weights at absolute address 0x34064000
 * (AXISRAM1, cpuRAM1 pool — boot-ROM-active, so GDB load_and_run writes it directly;
 * no parked-NPU-RAM problem). The linker routes .ai_weights to WEIGHTS_RAM @0x34064000.
 * dev=1: GDB loads it with the code. dev=0 (prod): a boot-copy from XSPI would feed
 * the same VMA — same technique, different loader.
 */
  .section .ai_weights, "a", %progbits
  .global g_ai_weights_blob
  .align 8
g_ai_weights_blob:
  .incbin "../../FSBL/AI/network_weights.bin"
  .global g_ai_weights_blob_end
g_ai_weights_blob_end:
