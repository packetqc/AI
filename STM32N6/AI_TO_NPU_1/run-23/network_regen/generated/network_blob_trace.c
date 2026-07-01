#include "ll_aton_NN_interface.h"
#include "ll_aton.h"
#include "ll_aton_ec_trace.h"

#if 0
// Workaround: the tracer does not know the target at this moment
// and cannot call the functions since are used in static code
#define ATON_LIB_PHYSICAL_TO_VIRTUAL_ADDR(address) LL_Address_Physical2Virtual(address)
#define ATON_LIB_VIRTUAL_TO_PHYSICAL_ADDR(address) LL_Address_Virtual2Physical(address)
#else
#define ATON_LIB_PHYSICAL_TO_VIRTUAL_ADDR(address) (address)
#define ATON_LIB_VIRTUAL_TO_PHYSICAL_ADDR(address) (address)
#endif


// MCU cache line size: 32 (bytes)
// NPU cache line size: 64 (bytes)
// MCU+NPU cache line size equal to 64 bytes (power of 2 not less than 8)
unsigned int cache_line_size = 64;

mpool_reloc_info_t mpool_reloc_info[] = {
  {"AXISRAM4", "_mem_pool_AXISRAM4_network", 0x34270000, 1, 0},
  {"AXISRAM3", "_mem_pool_AXISRAM3_network", 0x34200000, 1, 0},
  {"vencRAM", "_mem_pool_vencRAM_network", 0x34400000, 1, 0},
  {"npuCACHE", "_mem_pool_npuCACHE_network", 0x343c0000, 1, 0},
  {"xSPI1", "_mem_pool_xSPI1_network", 0x90000000, 1, 0},
  {"xSPI2", "_mem_pool_xSPI2_network", 0x70000000, 1, 0},
  {"AXISRAM3_AXISRAM4", "_mem_pool_AXISRAM3_AXISRAM4_network", 0x34200000, 1, 0},
  {NULL, NULL, 0, 0, 0}
};


void trace_ec__ec_blob_network_1(void) {
  ec_trace_start_blob("_ec_blob_network_1");
  ec_trace_start_epoch(1);
  {
  }
  {
  }
  ec_trace_end_epoch(1);
  ec_trace_start_epoch(2);
  {
    /* Unit= 28 [NULL_UNIT 0] */
    /* kind=Identity node=Identity_inserted_id42 */
    /* node=Identity_inserted_id42 satisfies input and output adjacency (DMA->DMA) and can be omitted */

    /* Dma inputs units to cycle: */
    /* Unit= 4 [STREAM_ENG_V2 4] */
    /* Emit conf for STREAM_ENG_V2 node=Identity_inserted_id42 input ports=0 range=1[0,8192] */

    static const LL_Streng_TensorInitTypeDef Identity_inserted_id42_dma_init_in_0_2 = {
      /* from memory with batch=1 */
      .dir = 0,
      .raw = 1,
      .noblk = 0,
      .align_right = 0,
      .nbits_unsigned = 0,
      .addr_base = {(unsigned char *)(0x34200000UL) /* Equivalent hex address = 0x34200000UL */}, /* Reshape_2_out_0_inserted_in42 */
      .offset_start = 0,
      .offset_end = 32,
      .offset_limit = 8256,
      .frame_count = 0,
      .fwidth = 0,
      .fheight = 0,
      .batch_depth = 0,
      .batch_offset = 0,
      .frame_offset = 32,
      .line_offset = 0,
      .loop_offset = 0,
      .frame_loop_cnt = 0,
      .frame_tot_cnt = 256,
      .nbits_in = 8,
      .nbits_out = 8,
    };

    /* Unit=STREAM_ENG_V2 */
    LL_Streng_TensorInit(4, &Identity_inserted_id42_dma_init_in_0_2, 1);


    /* Dma input bandwidth from memory pools: */
    /* npuRAM3 -> 8192 */

    /* Dma output units from cycle: */
    /* Unit= 5 [STREAM_ENG_V2 5] */
    /* Emit conf for STREAM_ENG_V2 node=Identity_inserted_id42 output ports=0 range=1[32768,40960] */

    static const LL_Streng_TensorInitTypeDef Identity_inserted_id42_dma_init_out_0_2 = {
      /* to memory canonical from batch=1 */
      .dir = 1,
      .noblk = 0,
      .align_right = 0,
      .nbits_unsigned = 0,
      .addr_base = {(unsigned char *)(0x34200000UL) /* Equivalent hex address = 0x34200000UL */}, /* Reshape_2_out_0_inserted_out42 */
      .offset_start = 32768,
      .offset_limit = 41024,
      .frame_count = 0,
      .fwidth = 1,
      .fheight = 32,
      .batch_depth = 1,
      .batch_offset = 256,
      .frame_offset = 1,
      .line_offset = 0,
      .loop_offset = 8192,
      .frame_loop_cnt = 256,
      .frame_tot_cnt = 256,
      .nbits_in = 8,
      .nbits_out = 8,
    };

    /* Unit=STREAM_ENG_V2 */
    LL_Streng_TensorInit(5, &Identity_inserted_id42_dma_init_out_0_2, 1);


    /* Dma output bandwidth to memory pools: */
    /* npuRAM3 <- 8192 */

    static const LL_Switch_InitTypeDef STREAM_SWITCH_0_init_in_2[] = {
      { LL_Switch_Init_Dest() = ATONN_DSTPORT(STRSWITCH, 0, STRENG, 5, 0), LL_Switch_Init_Source(0) = ATONN_SRCPORT(STRSWITCH, 0, STRENG, 4, 0), LL_Switch_Init_Context(0) = 1, LL_Switch_Init_Frames(0) = 0, }, /* Identity_inserted_id42 OUT: in unit=STREAM_ENG_V2 5 in port=0 out unit=STREAM_ENG_V2 4 out port=0 */
    };


    /* epoch=2 */
    LL_Switch_Init(STREAM_SWITCH_0_init_in_2, 1);

    static const LL_ATON_EnableUnits_InitTypeDef Enable_epoch_2_all_units[] = {
      { {STRENG, 5} }, /* STREAM_ENG_V2 */
      { {STRENG, 4} }, /* STREAM_ENG_V2 */
    };


    LL_ATON_EnableUnits_Init(Enable_epoch_2_all_units, 2);

  }

  ec_trace_wait_epoch_end(0x20);

  {
    static const LL_Switch_DeinitTypeDef STREAM_SWITCH_0_deinit_in_2[] = {
      { LL_Switch_Init_Dest() = ATONN_DSTPORT(STRSWITCH, 0, STRENG, 5, 0), LL_Switch_Init_Source(0) = ATONN_SRCPORT(STRSWITCH, 0, STRENG, 4, 0), LL_Switch_Init_Context(0) = 1, LL_Switch_Init_Frames(0) = 0, }, /* Identity_inserted_id42 OUT: in unit=STREAM_ENG_V2 5 in port=0 out unit=STREAM_ENG_V2 4 out port=0 */
    };


    /* epoch=2 */
    LL_Switch_Deinit(STREAM_SWITCH_0_deinit_in_2, 1);

    static const LL_ATON_DisableUnits_InitTypeDef Disable_epoch_2_all_units[] = {
      { {STRENG, 5} }, /* STREAM_ENG_V2 */
      { {STRENG, 4} }, /* STREAM_ENG_V2 */
    };


    LL_ATON_DisableUnits_Init(Disable_epoch_2_all_units, 2);

  }
  ec_trace_end_epoch(2);
  ec_trace_end_blob("_ec_blob_network_1");
}

void trace_ec__ec_blob_network_8(void) {
  ec_trace_start_blob("_ec_blob_network_8");
  ec_trace_start_epoch(8);
  {
    /* Unit= 28 [NULL_UNIT 0] */
    /* kind=Reshape node=Reshape_14 */
    /* node=Reshape_14 satisfies input and output adjacency (DMA->DMA) and can be omitted */

    /* Dma inputs units to cycle: */
    /* Unit= 6 [STREAM_ENG_V2 6] */
    /* Emit conf for STREAM_ENG_V2 node=Reshape_14 input ports=0 range=1[47872,59840] */

    static const LL_Streng_TensorInitTypeDef Reshape_14_dma_init_in_0_8 = {
      /* memory canonical to batch=1 */
      .dir = 0,
      .noblk = 0,
      .align_right = 0,
      .nbits_unsigned = 0,
      .addr_base = {(unsigned char *)(0x34200000UL) /* Equivalent hex address = 0x34200000UL */}, /* Conv2D_11_out_0 */
      .offset_start = 47872,
      .offset_limit = 59904,
      .frame_count = 0,
      .fwidth = 1,
      .fheight = 32,
      .batch_depth = 1,
      .batch_offset = 374,
      .frame_offset = 1,
      .line_offset = 0,
      .loop_offset = 11968,
      .frame_loop_cnt = 374,
      .frame_tot_cnt = 374,
      .nbits_in = 8,
      .nbits_out = 8,
    };

    /* Unit=STREAM_ENG_V2 */
    LL_Streng_TensorInit(6, &Reshape_14_dma_init_in_0_8, 1);


    /* Dma input bandwidth from memory pools: */
    /* npuRAM3 -> 11968 */

    /* Dma output units from cycle: */
    /* Unit= 1 [STREAM_ENG_V2 1] */
    /* Emit conf for STREAM_ENG_V2 node=Reshape_14 output ports=0 range=1[0,11968] */

    static const LL_Streng_TensorInitTypeDef Reshape_14_dma_init_out_0_8 = {
      /* to memory with batch=1 */
      .dir = 1,
      .raw = 1,
      .noblk = 0,
      .align_right = 0,
      .nbits_unsigned = 0,
      .addr_base = {(unsigned char *)(0x34200000UL) /* Equivalent hex address = 0x34200000UL */}, /* Quantize_15_out_0 */
      .offset_start = 0,
      .offset_end = 11968,
      .offset_limit = 12032,
      .frame_count = 0,
      .fwidth = 0,
      .fheight = 0,
      .batch_depth = 0,
      .batch_offset = 0,
      .frame_offset = 11968,
      .line_offset = 0,
      .loop_offset = 0,
      .frame_loop_cnt = 0,
      .frame_tot_cnt = 1,
      .nbits_in = 8,
      .nbits_out = 8,
    };

    /* Unit=STREAM_ENG_V2 */
    LL_Streng_TensorInit(1, &Reshape_14_dma_init_out_0_8, 1);


    /* Dma output bandwidth to memory pools: */
    /* npuRAM3 <- 11968 */

    static const LL_Switch_InitTypeDef STREAM_SWITCH_0_init_in_8[] = {
      { LL_Switch_Init_Dest() = ATONN_DSTPORT(STRSWITCH, 0, STRENG, 1, 0), LL_Switch_Init_Source(0) = ATONN_SRCPORT(STRSWITCH, 0, STRENG, 6, 0), LL_Switch_Init_Context(0) = 1, LL_Switch_Init_Frames(0) = 0, }, /* Reshape_14 OUT: in unit=STREAM_ENG_V2 1 in port=0 out unit=STREAM_ENG_V2 6 out port=0 */
    };


    /* epoch=8 */
    LL_Switch_Init(STREAM_SWITCH_0_init_in_8, 1);

    static const LL_ATON_EnableUnits_InitTypeDef Enable_epoch_8_all_units[] = {
      { {STRENG, 1} }, /* STREAM_ENG_V2 */
      { {STRENG, 6} }, /* STREAM_ENG_V2 */
    };


    LL_ATON_EnableUnits_Init(Enable_epoch_8_all_units, 2);

  }

  ec_trace_wait_epoch_end(0x2);

  {
    static const LL_Switch_DeinitTypeDef STREAM_SWITCH_0_deinit_in_8[] = {
      { LL_Switch_Init_Dest() = ATONN_DSTPORT(STRSWITCH, 0, STRENG, 1, 0), LL_Switch_Init_Source(0) = ATONN_SRCPORT(STRSWITCH, 0, STRENG, 6, 0), LL_Switch_Init_Context(0) = 1, LL_Switch_Init_Frames(0) = 0, }, /* Reshape_14 OUT: in unit=STREAM_ENG_V2 1 in port=0 out unit=STREAM_ENG_V2 6 out port=0 */
    };


    /* epoch=8 */
    LL_Switch_Deinit(STREAM_SWITCH_0_deinit_in_8, 1);

    static const LL_ATON_DisableUnits_InitTypeDef Disable_epoch_8_all_units[] = {
      { {STRENG, 1} }, /* STREAM_ENG_V2 */
      { {STRENG, 6} }, /* STREAM_ENG_V2 */
    };


    LL_ATON_DisableUnits_Init(Disable_epoch_8_all_units, 2);

  }
  ec_trace_end_epoch(8);
  ec_trace_start_epoch(9);
  {
    /* Dma input bandwidth from memory pools: */
    /* npuRAM3 -> 0 */

  }
  {
  }
  ec_trace_end_epoch(9);
  ec_trace_end_blob("_ec_blob_network_8");
}


int main () {
  ec_trace_init("network_ecblobs.h", "network", false, 0, false);
  trace_ec__ec_blob_network_1();
  trace_ec__ec_blob_network_8();
  ec_trace_all_blobs_done();
}
