/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-22T20:07:48-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x15d7930dc226850d64430195fc002a21"
#define _STAI_NETWORK_DATETIME            "2026-06-22T20:07:48-0400"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 32),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_2_NAME,
      STAI_NETWORK_IN_2_FLAGS,
      STAI_NETWORK_IN_2_FORMAT,
      STAI_NETWORK_IN_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 32),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 32, 1, 374),
      STAI_DECLARE_ARRAY(float, 1, 0.06943628191947937f),
      STAI_DECLARE_ARRAY(int16_t, 1, -52)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 327960),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 1383956),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL,NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(embedding_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007210441981442273f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007210441981442273f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_Mul_0_0_add_4_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.6172205000184476e-06f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.6211422411724925e-06f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_63_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.921568403342235e-09f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(rsqrt_0_1_mul_3_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.22850461304187775f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03337350860238075f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03342914581298828f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003965076524764299f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_82_0_0_linear_2_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015435585752129555f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015425463207066059f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(4.419077231432311e-05f),
    AI_PACK_INTQ_ZP(-11)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015425463207066059f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(expand_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015425463207066059f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_74_0_0_linear_1_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030147666111588478f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03026483952999115f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002501577837392688f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03026483952999115f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03026483952999115f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(neg_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020659925416111946f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03026483952999115f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cat_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030712634325027466f),
    AI_PACK_INTQ_ZP(37)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02995060570538044f),
    AI_PACK_INTQ_ZP(-42)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unsqueeze_5_DequantizeLinear_Output_const_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843070663511753f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031248044222593307f),
    AI_PACK_INTQ_ZP(-21)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(unsqueeze_4_DequantizeLinear_Output_const_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007842984050512314f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03396735340356827f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(expand_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03396735340356827f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_169_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03396735340356827f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_175_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012009273283183575f),
    AI_PACK_INTQ_ZP(-22)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_172_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013864838983863592f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_66_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031586986035108566f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_66_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007173916092142463f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00016484150546602905f),
    AI_PACK_INTQ_ZP(30)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(neg_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cat_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03172483667731285f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034411363303661346f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031411200761795044f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03366806358098984f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_173_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011903457343578339f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_35_0_1_bitwise_and_1_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(bitwise_and_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(bitwise_and_DequantizeLinear_Output_const_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(__bool_fix_inv_0_0_val_178_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_178_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.3344405750530544e+36f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_177_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.3344405750530544e+36f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_179_0_0_val_180_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03992549702525139f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_180_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.3344405750530544e+36f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(scaled_dot_product_attention_0_0_transpose_4_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012832569889724255f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012832569889724255f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0050929030403494835f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_3_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007289691129699349f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005136104766279459f),
    AI_PACK_INTQ_ZP(9)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_1_Mul_0_0_add_8_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001574616035213694f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00015746551798656583f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(rsqrt_1_0_1_mul_9_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14992481470108032f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03725521266460419f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03728029876947403f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003982512280344963f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02244340442121029f),
    AI_PACK_INTQ_ZP(-15)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_4_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008300553308799863f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_195_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00376460631377995f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(silu_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013053648173809052f),
    AI_PACK_INTQ_ZP(-107)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01960653066635132f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_5_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007386000943370163f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_11_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03886575251817703f),
    AI_PACK_INTQ_ZP(40)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005923588294535875f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_6_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007735750987194479f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00763314263895154f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_2_Mul_0_0_add_10_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005274713039398193f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005274752038531005f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(rsqrt_2_0_1_mul_12_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09743647277355194f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03564446046948433f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0354953333735466f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392933189868927f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_203_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021515561267733574f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_203_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007912579458206892f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02165989577770233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00024592349654994905f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02165989577770233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02165989577770233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(neg_2_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021231386810541153f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02165989577770233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cat_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0228018369525671f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_15_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021387392655014992f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_14_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020335372537374496f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_11_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02787942998111248f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_306_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009856867603957653f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_211_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02850356325507164f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_211_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007217677193693817f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02870889939367771f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00023856799816712737f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02870889939367771f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #97 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02870889939367771f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #98 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(neg_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02457888424396515f),
    AI_PACK_INTQ_ZP(-23)))

/* Int quant #99 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02870889939367771f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #100 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(cat_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031967829912900925f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #101 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_17_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.028150850906968117f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #102 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03174330294132233f),
    AI_PACK_INTQ_ZP(-10)))

/* Int quant #103 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03317304700613022f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #104 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(expand_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03317304700613022f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #105 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_302_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03317304700613022f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #106 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_308_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011728443205356598f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #107 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_312_0_0_val_313_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03215616196393967f),
    AI_PACK_INTQ_ZP(-45)))

/* Int quant #108 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_313_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.3344405750530544e+36f),
    AI_PACK_INTQ_ZP(127)))

/* Int quant #109 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_219_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013596580363810062f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #110 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_219_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00063577841501683f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #111 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013600028119981289f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #112 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(5.547447290155105e-05f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #113 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013600028119981289f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #114 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(expand_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013600028119981289f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #115 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012092793360352516f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #116 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012092793360352516f),
    AI_PACK_INTQ_ZP(-9)))

/* Int quant #117 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005171344615519047f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #118 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_10_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006693590548820794f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #119 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010768704116344452f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #120 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_3_Mul_0_0_add_14_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010057041654363275f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #121 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_14_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010057081235572696f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #122 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(rsqrt_3_0_1_mul_18_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.050555404275655746f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #123 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_18_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034200724214315414f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #124 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_19_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.034479402005672455f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #125 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00403008796274662f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #126 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_11_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023213928565382957f),
    AI_PACK_INTQ_ZP(-31)))

/* Int quant #127 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_11_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007725649629719555f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #128 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(val_328_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0038246209733188152f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #129 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(silu_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015147668309509754f),
    AI_PACK_INTQ_ZP(-110)))

/* Int quant #130 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022492757067084312f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #131 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_12_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007421314367093146f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #132 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_20_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05286220833659172f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #133 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016507189720869064f),
    AI_PACK_INTQ_ZP(-12)))

/* Int quant #134 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(linear_13_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000737417780328542f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #135 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_15_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02223907597362995f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #136 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_4_Mul_0_0_add_16_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006991112604737282f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #137 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(add_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00699111633002758f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #138 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(rsqrt_4_0_1_mul_21_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038929760456085205f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #139 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_21_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.033919114619493484f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #140 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mul_22_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.036029066890478134f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #141 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004295083694159985f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #142 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06943628191947937f),
    AI_PACK_INTQ_ZP(-52)))

/* Int quant #143 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0011406512930989265f),
    AI_PACK_INTQ_ZP(0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  attention_mask_output_array, AI_ARRAY_FORMAT_S32|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 32, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _to_copy_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  input_ids_output_array, AI_ARRAY_FORMAT_S32|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 32, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  __ids_floored_output_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  __ids_floor_zero_2D_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  input_ids_clipped_output_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  __ids_cap_val_2D_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  embedding_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 95744, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  embedding_0_0_pow_1_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  pow_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  val_60_3D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  mean_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_0_0_add_4_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  add_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  val_63_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  rsqrt_0_1_mul_3_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  mul_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  mul_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  val_82_bias_0_2_val_82_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  mul_4_0_0_val_74_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  val_82_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  val_82_0_0_linear_2_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  linear_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  transpose_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  expand_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  val_74_bias_0_2_val_74_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  val_74_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  val_74_0_0_linear_1_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  linear_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  transpose_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  slice_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  neg_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  slice_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  cat_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  mul_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  unsqueeze_5_DequantizeLinear_Output_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  mul_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  unsqueeze_4_DequantizeLinear_Output_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  add_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  expand_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  val_169_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  val_175_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  val_172_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  val_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  val_66_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  val_66_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  val_66_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  linear_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  transpose_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  slice_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  neg_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  slice_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  cat_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  mul_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  mul_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  add_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  val_173_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  val_179_bias_0_2_val_179_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  val_175_0_1_val_179_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  val_173_0_0_val_179_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  val_179_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  val_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  val_34_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  val_35_0_1_bitwise_and_1_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  bitwise_and_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  bitwise_and_DequantizeLinear_Output_const_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  bitwise_and_1_0_1___bool_fix_inv_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  __bool_fix_inv_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  __bool_fix_one_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  __bool_fix_inv_0_0_val_178_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  val_178_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  val_177_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  val_179_0_0_val_180_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  val_180_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  expand_4_0_1_scaled_dot_product_attention_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  val_181_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_0_0_transpose_4_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  transpose_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  linear_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  linear_3_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  linear_3_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  add_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  add_7_0_0_pow_2_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  pow_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  mean_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  mean_1_Mul_0_0_add_8_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  add_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  rsqrt_1_0_1_mul_9_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  mul_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  mul_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  linear_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  linear_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  linear_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 512, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  linear_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  val_195_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  silu_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  linear_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  linear_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  linear_5_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  mul_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  linear_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  linear_6_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  linear_6_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  add_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  add_9_0_0_pow_3_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  pow_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  mean_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  mean_2_Mul_0_0_add_10_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  add_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  rsqrt_2_0_1_mul_12_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  mul_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  mul_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  val_203_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  val_203_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  val_203_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  linear_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  transpose_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  slice_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  neg_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  slice_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  cat_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  mul_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  mul_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  add_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  val_306_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  val_211_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  val_211_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  val_211_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  val_211_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  linear_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  transpose_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  slice_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  neg_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  slice_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  cat_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  mul_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  mul_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  add_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  expand_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  val_302_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  val_308_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  val_312_bias_0_2_val_312_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  val_306_0_0_val_312_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  val_308_0_1_val_312_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  val_312_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  val_312_0_0_val_313_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  val_313_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  val_219_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  val_219_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  val_219_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  linear_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  transpose_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  expand_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_1_bias_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  val_314_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  expand_6_0_1_scaled_dot_product_attention_1_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  transpose_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  linear_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  linear_10_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  linear_10_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  add_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  add_13_0_0_pow_4_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  pow_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  mean_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  mean_3_Mul_0_0_add_14_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  add_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  rsqrt_3_0_1_mul_18_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  mul_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  mul_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  linear_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  linear_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  linear_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  val_328_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  silu_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  linear_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  linear_12_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  linear_12_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  mul_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  linear_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  linear_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 131072, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  linear_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  add_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  add_15_0_0_pow_5_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  pow_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  mean_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  mean_4_Mul_0_0_add_16_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  add_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  rsqrt_4_0_1_mul_21_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  mul_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  mul_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 11968, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 95744, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 374, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _to_copy_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_to_copy_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  attention_mask_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &attention_mask_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  __ids_floor_zero_2D, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &__ids_floor_zero_2D_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  __ids_floored_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &__ids_floored_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  input_ids_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &input_ids_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  __ids_cap_val_2D, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &__ids_cap_val_2D_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  input_ids_clipped_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &input_ids_clipped_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  embedding_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 32, 1), AI_STRIDE_INIT(4, 1, 1, 256, 8192),
  1, &embedding_output_array, &embedding_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_embed_tokens_weight_DequantizeLinear_Output_const, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 374), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array, &mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  embedding_0_0_pow_1_conversion_output0, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &embedding_0_0_pow_1_conversion_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  pow_1_output, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &pow_1_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  val_60_3D, AI_STATIC,
  245, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_60_3D_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  mean_output, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mean_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  add_4_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &add_4_output_array, &add_4_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_0_0_add_4_conversion_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &mean_Mul_0_0_add_4_conversion_output_array, &mean_Mul_0_0_add_4_conversion_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  val_63_DequantizeLinear_Output_const_3D, AI_STATIC,
  246, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &val_63_DequantizeLinear_Output_const_3D_array, &val_63_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  embedding_output0, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &embedding_output_array, &embedding_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  mul_3_output, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_3_output_array, &mul_3_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  rsqrt_0_1_mul_3_conversion_output, AI_STATIC,
  157, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &rsqrt_0_1_mul_3_conversion_output_array, &rsqrt_0_1_mul_3_conversion_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  mul_4_output, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_4_output_array, &mul_4_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  mul_4_0_0_val_74_conversion_output0, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 32, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 32768),
  1, &mul_4_0_0_val_74_conversion_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output0, AI_STATIC,
  262, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 256, 1), AI_STRIDE_INIT(4, 4, 4, 512, 131072),
  1, &val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  val_82_bias_0_2_val_82_conversion_output, AI_STATIC,
  266, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_82_bias_0_2_val_82_conversion_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  val_82_output, AI_STATIC,
  267, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 32, 1), AI_STRIDE_INIT(4, 4, 4, 512, 16384),
  1, &val_82_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  linear_2_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &linear_2_output_array, &linear_2_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  val_82_0_0_linear_2_conversion_output0, AI_STATIC,
  264, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &val_82_0_0_linear_2_conversion_output_array, &val_82_0_0_linear_2_conversion_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  linear_2_output0, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 2, 32), AI_STRIDE_INIT(4, 1, 1, 64, 128),
  1, &linear_2_output_array, &linear_2_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  transpose_3_output, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_3_output_array, &transpose_3_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  expand_4_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 2, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 4096, 64),
  1, &expand_4_output_array, &expand_4_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  transpose_3_output0, AI_STATIC,
  188, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 1, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 2048, 64),
  1, &transpose_3_output_array, &transpose_3_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output0, AI_STATIC,
  254, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 256, 1), AI_STRIDE_INIT(4, 4, 4, 512, 131072),
  1, &val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  val_74_bias_0_2_val_74_conversion_output, AI_STATIC,
  258, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_74_bias_0_2_val_74_conversion_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  val_74_output, AI_STATIC,
  259, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 32, 1), AI_STRIDE_INIT(4, 4, 4, 512, 16384),
  1, &val_74_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  linear_1_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &linear_1_output_array, &linear_1_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  val_74_0_0_linear_1_conversion_output0, AI_STATIC,
  256, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &val_74_0_0_linear_1_conversion_output_array, &val_74_0_0_linear_1_conversion_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  linear_1_output0, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 2, 32), AI_STRIDE_INIT(4, 1, 1, 64, 128),
  1, &linear_1_output_array, &linear_1_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  transpose_2_output, AI_STATIC,
  186, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_2_output_array, &transpose_2_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  slice_4_output, AI_STATIC,
  180, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_4_output_array, &slice_4_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  neg_1_output, AI_STATIC,
  148, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &neg_1_output_array, &neg_1_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  slice_3_output, AI_STATIC,
  179, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_3_output_array, &slice_3_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  cat_2_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &cat_2_output_array, &cat_2_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  mul_8_output, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_8_output_array, &mul_8_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  unsqueeze_5_DequantizeLinear_Output_const, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &unsqueeze_5_DequantizeLinear_Output_const_array, &unsqueeze_5_DequantizeLinear_Output_const_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  mul_7_output, AI_STATIC,
  145, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_7_output_array, &mul_7_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  unsqueeze_4_DequantizeLinear_Output_const, AI_STATIC,
  197, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &unsqueeze_4_DequantizeLinear_Output_const_array, &unsqueeze_4_DequantizeLinear_Output_const_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  add_6_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &add_6_output_array, &add_6_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  add_6_output0, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 1, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 2048, 64),
  1, &add_6_output_array, &add_6_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  expand_3_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 2, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 4096, 64),
  1, &expand_3_output_array, &expand_3_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  expand_3_output0, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &expand_3_output_array, &expand_3_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  val_169_output, AI_STATIC,
  199, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &val_169_output_array, &val_169_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  val_172_DequantizeLinear_Output_const_4D, AI_STATIC,
  200, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &val_172_DequantizeLinear_Output_const_4D_array, &val_172_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  val_175_output, AI_STATIC,
  204, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &val_175_output_array, &val_175_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  val_66_bias, AI_STATIC,
  248, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &val_66_bias_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  val_66_output, AI_STATIC,
  249, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &val_66_output_array, &val_66_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  val_66_scratch0, AI_STATIC,
  250, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &val_66_scratch0_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  val_66_weights, AI_STATIC,
  251, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &val_66_weights_array, &val_66_weights_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  linear_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_output_array, &linear_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  linear_output0, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 32), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &linear_output_array, &linear_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  transpose_1_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_1_output_array, &transpose_1_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  slice_2_output, AI_STATIC,
  178, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_2_output_array, &slice_2_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  neg_output, AI_STATIC,
  151, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &neg_output_array, &neg_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  slice_1_output, AI_STATIC,
  177, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_1_output_array, &slice_1_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  cat_1_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &cat_1_output_array, &cat_1_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  mul_6_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_6_output_array, &mul_6_output_array_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  mul_5_output, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_5_output_array, &mul_5_output_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  add_5_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &add_5_output_array, &add_5_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  val_173_output, AI_STATIC,
  202, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &val_173_output_array, &val_173_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  val_173_0_0_val_179_conversion_output, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &val_173_0_0_val_179_conversion_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  val_175_0_1_val_179_conversion_output, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 4, 4, 128, 8192),
  1, &val_175_0_1_val_179_conversion_output_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  val_179_bias_0_2_val_179_conversion_output, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_179_bias_0_2_val_179_conversion_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  val_179_output, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &val_179_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  val_34, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 32), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &val_34_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  val_35_output, AI_STATIC,
  244, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &val_35_output_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  bitwise_and_1_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 32), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &bitwise_and_1_output_array, &bitwise_and_1_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  bitwise_and_DequantizeLinear_Output_const, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 32), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &bitwise_and_DequantizeLinear_Output_const_array, &bitwise_and_DequantizeLinear_Output_const_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  val_35_0_1_bitwise_and_1_conversion_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &val_35_0_1_bitwise_and_1_conversion_output_array, &val_35_0_1_bitwise_and_1_conversion_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  __bool_fix_inv_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 1), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &__bool_fix_inv_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  __bool_fix_one_4D, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &__bool_fix_one_4D_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  bitwise_and_1_0_1___bool_fix_inv_conversion_output0, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 1), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &bitwise_and_1_0_1___bool_fix_inv_conversion_output_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  __bool_fix_inv_0_0_val_178_conversion_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &__bool_fix_inv_0_0_val_178_conversion_output_array, &__bool_fix_inv_0_0_val_178_conversion_output_array_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  val_177_DequantizeLinear_Output_const_4D, AI_STATIC,
  205, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &val_177_DequantizeLinear_Output_const_4D_array, &val_177_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  val_178_output, AI_STATIC,
  206, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &val_178_output_array, &val_178_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  val_179_0_0_val_180_conversion_output, AI_STATIC,
  207, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &val_179_0_0_val_180_conversion_output_array, &val_179_0_0_val_180_conversion_output_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  val_180_output, AI_STATIC,
  212, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &val_180_output_array, &val_180_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  expand_4_0_1_scaled_dot_product_attention_conversion_output0, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &expand_4_0_1_scaled_dot_product_attention_conversion_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &scaled_dot_product_attention_output_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  val_181_output, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &val_181_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_0_0_transpose_4_conversion_output, AI_STATIC,
  167, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &scaled_dot_product_attention_0_0_transpose_4_conversion_output_array, &scaled_dot_product_attention_0_0_transpose_4_conversion_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  transpose_4_output, AI_STATIC,
  189, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 32), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_4_output_array, &transpose_4_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  linear_3_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_3_output_array, &linear_3_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  linear_3_scratch0, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_3_scratch0_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  linear_3_weights, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &linear_3_weights_array, &linear_3_weights_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  transpose_4_output0, AI_STATIC,
  190, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_4_output_array, &transpose_4_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  add_7_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &add_7_output_array, &add_7_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  add_7_0_0_pow_2_conversion_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &add_7_0_0_pow_2_conversion_output_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  pow_2_output, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &pow_2_output_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  mean_1_output, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mean_1_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  add_8_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &add_8_output_array, &add_8_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  mean_1_Mul_0_0_add_8_conversion_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &mean_1_Mul_0_0_add_8_conversion_output_array, &mean_1_Mul_0_0_add_8_conversion_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  mul_9_output, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_9_output_array, &mul_9_output_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  rsqrt_1_0_1_mul_9_conversion_output, AI_STATIC,
  158, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &rsqrt_1_0_1_mul_9_conversion_output_array, &rsqrt_1_0_1_mul_9_conversion_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  mul_10_output, AI_STATIC,
  126, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_10_output_array, &mul_10_output_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  linear_4_bias, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &linear_4_bias_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  linear_4_output, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &linear_4_output_array, &linear_4_output_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  linear_4_scratch0, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_4_scratch0_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  linear_4_weights, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &linear_4_weights_array, &linear_4_weights_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  val_195_output, AI_STATIC,
  215, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &val_195_output_array, &val_195_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  silu_output, AI_STATIC,
  176, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &silu_output_array, &silu_output_array_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  linear_5_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &linear_5_output_array, &linear_5_output_array_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  linear_5_scratch0, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_5_scratch0_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  linear_5_weights, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &linear_5_weights_array, &linear_5_weights_array_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  mul_11_output, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &mul_11_output_array, &mul_11_output_array_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  linear_6_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_6_output_array, &linear_6_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  linear_6_scratch0, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &linear_6_scratch0_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  linear_6_weights, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &linear_6_weights_array, &linear_6_weights_array_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  add_9_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &add_9_output_array, &add_9_output_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  add_9_0_0_pow_3_conversion_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &add_9_0_0_pow_3_conversion_output_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  pow_3_output, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &pow_3_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  mean_2_output, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mean_2_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  add_10_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &add_10_output_array, &add_10_output_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  mean_2_Mul_0_0_add_10_conversion_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &mean_2_Mul_0_0_add_10_conversion_output_array, &mean_2_Mul_0_0_add_10_conversion_output_array_intq)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  mul_12_output, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_12_output_array, &mul_12_output_array_intq)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  rsqrt_2_0_1_mul_12_conversion_output, AI_STATIC,
  160, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &rsqrt_2_0_1_mul_12_conversion_output_array, &rsqrt_2_0_1_mul_12_conversion_output_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  mul_13_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_13_output_array, &mul_13_output_array_intq)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  val_203_output, AI_STATIC,
  217, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &val_203_output_array, &val_203_output_array_intq)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  val_203_scratch0, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &val_203_scratch0_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  val_203_weights, AI_STATIC,
  219, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &val_203_weights_array, &val_203_weights_array_intq)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  linear_7_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_7_output_array, &linear_7_output_array_intq)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  linear_7_output0, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 32), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &linear_7_output_array, &linear_7_output_array_intq)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  transpose_5_output, AI_STATIC,
  191, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_5_output_array, &transpose_5_output_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  slice_6_output, AI_STATIC,
  182, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_6_output_array, &slice_6_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  neg_2_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &neg_2_output_array, &neg_2_output_array_intq)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  slice_5_output, AI_STATIC,
  181, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_5_output_array, &slice_5_output_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  cat_3_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &cat_3_output_array, &cat_3_output_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  mul_15_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_15_output_array, &mul_15_output_array_intq)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  mul_14_output, AI_STATIC,
  130, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_14_output_array, &mul_14_output_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  add_11_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &add_11_output_array, &add_11_output_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  val_306_output, AI_STATIC,
  229, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &val_306_output_array, &val_306_output_array_intq)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  val_211_bias, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &val_211_bias_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  val_211_output, AI_STATIC,
  221, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &val_211_output_array, &val_211_output_array_intq)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  val_211_scratch0, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &val_211_scratch0_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  val_211_weights, AI_STATIC,
  223, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 128), AI_STRIDE_INIT(4, 1, 256, 32768, 32768),
  1, &val_211_weights_array, &val_211_weights_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  linear_8_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &linear_8_output_array, &linear_8_output_array_intq)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  linear_8_output0, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 2, 32), AI_STRIDE_INIT(4, 1, 1, 64, 128),
  1, &linear_8_output_array, &linear_8_output_array_intq)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  transpose_6_output, AI_STATIC,
  192, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_6_output_array, &transpose_6_output_array_intq)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  slice_8_output, AI_STATIC,
  184, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_8_output_array, &slice_8_output_array_intq)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  neg_3_output, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &neg_3_output_array, &neg_3_output_array_intq)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  slice_7_output, AI_STATIC,
  183, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &slice_7_output_array, &slice_7_output_array_intq)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  cat_4_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &cat_4_output_array, &cat_4_output_array_intq)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  mul_17_output, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_17_output_array, &mul_17_output_array_intq)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  mul_16_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &mul_16_output_array, &mul_16_output_array_intq)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  add_12_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &add_12_output_array, &add_12_output_array_intq)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  add_12_output0, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 1, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 2048, 64),
  1, &add_12_output_array, &add_12_output_array_intq)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  expand_5_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 2, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 4096, 64),
  1, &expand_5_output_array, &expand_5_output_array_intq)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  expand_5_output0, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &expand_5_output_array, &expand_5_output_array_intq)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  val_302_output, AI_STATIC,
  227, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &val_302_output_array, &val_302_output_array_intq)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  val_308_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &val_308_output_array, &val_308_output_array_intq)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  val_306_0_0_val_312_conversion_output, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &val_306_0_0_val_312_conversion_output_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  val_308_0_1_val_312_conversion_output, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 64, 4), AI_STRIDE_INIT(4, 4, 4, 128, 8192),
  1, &val_308_0_1_val_312_conversion_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  val_312_bias_0_2_val_312_conversion_output, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_312_bias_0_2_val_312_conversion_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  val_312_output, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &val_312_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  val_312_0_0_val_313_conversion_output, AI_STATIC,
  232, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &val_312_0_0_val_313_conversion_output_array, &val_312_0_0_val_313_conversion_output_array_intq)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  val_313_output, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &val_313_output_array, &val_313_output_array_intq)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  val_219_output, AI_STATIC,
  224, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &val_219_output_array, &val_219_output_array_intq)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  val_219_scratch0, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &val_219_scratch0_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  val_219_weights, AI_STATIC,
  226, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 128), AI_STRIDE_INIT(4, 1, 256, 32768, 32768),
  1, &val_219_weights_array, &val_219_weights_array_intq)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  linear_9_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 32), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &linear_9_output_array, &linear_9_output_array_intq)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  linear_9_output0, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 2, 32), AI_STRIDE_INIT(4, 1, 1, 64, 128),
  1, &linear_9_output_array, &linear_9_output_array_intq)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  transpose_7_output, AI_STATIC,
  193, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &transpose_7_output_array, &transpose_7_output_array_intq)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  expand_6_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 2, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 4096, 64),
  1, &expand_6_output_array, &expand_6_output_array_intq)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  transpose_7_output0, AI_STATIC,
  194, 0x1,
  AI_SHAPE_INIT(5, 1, 64, 1, 2, 32), AI_STRIDE_INIT(5, 1, 1, 2048, 2048, 64),
  1, &transpose_7_output_array, &transpose_7_output_array_intq)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  expand_6_0_1_scaled_dot_product_attention_1_conversion_output0, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &expand_6_0_1_scaled_dot_product_attention_1_conversion_output_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_1_bias_0_conversion_output, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &scaled_dot_product_attention_1_bias_0_conversion_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_1_output, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 4, 4, 256, 8192),
  1, &scaled_dot_product_attention_1_output_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  val_314_output, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 4), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &val_314_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  scaled_dot_product_attention_1_0_0_transpose_8_conversion_output, AI_STATIC,
  168, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 4), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array, &scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array_intq)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  transpose_8_output, AI_STATIC,
  195, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 32), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &transpose_8_output_array, &transpose_8_output_array_intq)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  linear_10_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_10_output_array, &linear_10_output_array_intq)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  linear_10_scratch0, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_10_scratch0_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  linear_10_weights, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 65536, 65536),
  1, &linear_10_weights_array, &linear_10_weights_array_intq)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  transpose_8_output0, AI_STATIC,
  196, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &transpose_8_output_array, &transpose_8_output_array_intq)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  add_13_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &add_13_output_array, &add_13_output_array_intq)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  add_13_0_0_pow_4_conversion_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &add_13_0_0_pow_4_conversion_output_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  pow_4_output, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &pow_4_output_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  mean_3_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mean_3_output_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  add_14_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &add_14_output_array, &add_14_output_array_intq)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  mean_3_Mul_0_0_add_14_conversion_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &mean_3_Mul_0_0_add_14_conversion_output_array, &mean_3_Mul_0_0_add_14_conversion_output_array_intq)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  mul_18_output, AI_STATIC,
  134, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_18_output_array, &mul_18_output_array_intq)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  rsqrt_3_0_1_mul_18_conversion_output, AI_STATIC,
  162, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &rsqrt_3_0_1_mul_18_conversion_output_array, &rsqrt_3_0_1_mul_18_conversion_output_array_intq)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array, &mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  mul_19_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_19_output_array, &mul_19_output_array_intq)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  linear_11_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &linear_11_output_array, &linear_11_output_array_intq)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  linear_11_scratch0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_11_scratch0_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  linear_11_weights, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &linear_11_weights_array, &linear_11_weights_array_intq)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  val_328_output, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &val_328_output_array, &val_328_output_array_intq)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  silu_1_output, AI_STATIC,
  175, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &silu_1_output_array, &silu_1_output_array_intq)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  linear_12_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &linear_12_output_array, &linear_12_output_array_intq)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  linear_12_scratch0, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &linear_12_scratch0_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  linear_12_weights, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 512), AI_STRIDE_INIT(4, 1, 256, 131072, 131072),
  1, &linear_12_weights_array, &linear_12_weights_array_intq)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  mul_20_output, AI_STATIC,
  136, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 32), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &mul_20_output_array, &mul_20_output_array_intq)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  linear_13_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &linear_13_output_array, &linear_13_output_array_intq)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  linear_13_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2048, 2048),
  1, &linear_13_scratch0_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  linear_13_weights, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 512, 1, 1, 256), AI_STRIDE_INIT(4, 1, 512, 131072, 131072),
  1, &linear_13_weights_array, &linear_13_weights_array_intq)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  add_15_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &add_15_output_array, &add_15_output_array_intq)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  add_15_0_0_pow_5_conversion_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &add_15_0_0_pow_5_conversion_output_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  pow_5_output, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &pow_5_output_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  mean_4_output, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &mean_4_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  add_16_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &add_16_output_array, &add_16_output_array_intq)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  mean_4_Mul_0_0_add_16_conversion_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &mean_4_Mul_0_0_add_16_conversion_output_array, &mean_4_Mul_0_0_add_16_conversion_output_array_intq)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  mul_21_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_21_output_array, &mul_21_output_array_intq)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  rsqrt_4_0_1_mul_21_conversion_output, AI_STATIC,
  164, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &rsqrt_4_0_1_mul_21_conversion_output_array, &rsqrt_4_0_1_mul_21_conversion_output_array_intq)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  mdl_model_norm_weight_DequantizeLinear_Output_const_3D, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array, &mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array_intq)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  mul_22_output, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 32), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &mul_22_output_array, &mul_22_output_array_intq)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 374, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1496, 1496),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 374, 1, 32), AI_STRIDE_INIT(4, 1, 1, 374, 374),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 256, 1, 1, 374), AI_STRIDE_INIT(4, 1, 256, 95744, 95744),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _to_copy_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &attention_mask_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_to_copy_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _to_copy_layer, 40,
  CAST_TYPE, 0x0, NULL,
  cast, forward_cast,
  &_to_copy_chain,
  NULL, &_to_copy_layer, AI_STATIC, 
  .to_format = AI_ARRAY_FORMAT_FLOAT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  __ids_floored_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &input_ids_output, &__ids_floor_zero_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &__ids_floored_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  __ids_floored_layer, 55,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &__ids_floored_chain,
  NULL, &__ids_floored_layer, AI_STATIC, 
  .operation = ai_max_s32, 
  .buffer_operation = ai_max_buffer_s32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_ids_clipped_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &__ids_floored_output, &__ids_cap_val_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_ids_clipped_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_ids_clipped_layer, 76,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &input_ids_clipped_chain,
  NULL, &input_ids_clipped_layer, AI_STATIC, 
  .operation = ai_min_s32, 
  .buffer_operation = ai_min_buffer_s32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  embedding_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_embed_tokens_weight_DequantizeLinear_Output_const, &input_ids_clipped_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &embedding_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  embedding_layer, 97,
  GATHER_TYPE, 0x0, NULL,
  gather, forward_gather,
  &embedding_chain,
  NULL, &embedding_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pow_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &embedding_0_0_pow_1_conversion_output0, &val_60_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pow_1_layer, 103,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &pow_1_chain,
  NULL, &pow_1_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float mean_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_neutral_value_data, mean_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_layer, 105,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_chain,
  NULL, &mean_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_Mul_0_0_add_4_conversion_output, &val_63_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_4_layer, 111,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_4_chain,
  NULL, &add_4_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &embedding_output0, &rsqrt_0_1_mul_3_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_3_layer, 125,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_3_chain,
  NULL, &mul_3_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D, &mul_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_4_layer, 128,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_4_chain,
  NULL, &mul_4_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &mul_4_0_0_val_74_conversion_output0, &val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output0, &val_82_bias_0_2_val_82_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_82_layer, 133,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &val_82_chain,
  NULL, &val_82_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_82_0_0_linear_2_conversion_output0, &mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_2_layer, 142,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_2_chain,
  NULL, &linear_2_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_2_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_3_layer, 160,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_3_chain,
  NULL, &transpose_3_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i16 expand_4_repeats_data[] = { 1, 2, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    expand_4_repeats, AI_ARRAY_FORMAT_S16,
    expand_4_repeats_data, expand_4_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  expand_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_3_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expand_4_layer, 190,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &expand_4_chain,
  NULL, &expand_4_layer, AI_STATIC, 
  .repeats = &expand_4_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &mul_4_0_0_val_74_conversion_output0, &val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output0, &val_74_bias_0_2_val_74_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_74_layer, 132,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &val_74_chain,
  NULL, &val_74_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_74_0_0_linear_1_conversion_output0, &mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_1_layer, 141,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_1_chain,
  NULL, &linear_1_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_1_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_2_layer, 159,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_2_chain,
  NULL, &transpose_2_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 slice_4_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_axes, AI_ARRAY_FORMAT_U8,
    slice_4_axes_data, slice_4_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_4_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_starts, AI_ARRAY_FORMAT_S16,
    slice_4_starts_data, slice_4_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_4_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_4_ends, AI_ARRAY_FORMAT_S16,
    slice_4_ends_data, slice_4_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_4_layer, 172,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_4_chain,
  NULL, &slice_4_layer, AI_STATIC, 
  .axes = &slice_4_axes, 
  .starts = &slice_4_starts, 
  .ends = &slice_4_ends, 
)


AI_STATIC_CONST ai_i8 neg_1_nl_params_data[] = { 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 125, 123, 122, 120, 119, 118, 116, 115, 113, 112, 110, 109, 107, 106, 104, 103, 101, 100, 98, 97, 96, 94, 93, 91, 90, 88, 87, 85, 84, 82, 81, 79, 78, 76, 75, 74, 72, 71, 69, 68, 66, 65, 63, 62, 60, 59, 57, 56, 55, 53, 52, 50, 49, 47, 46, 44, 43, 41, 40, 38, 37, 35, 34, 33, 31, 30, 28, 27, 25, 24, 22, 21, 19, 18, 16, 15, 14, 12, 11, 9, 8, 6, 5, 3, 2, 0, -1, -3, -4, -6, -7, -8, -10, -11, -13, -14, -16, -17, -19, -20, -22, -23, -25, -26, -28, -29, -30, -32, -33, -35, -36, -38, -39, -41, -42, -44, -45, -47, -48, -49, -51, -52, -54, -55, -57, -58, -60, -61, -63, -64, -66, -67, -69, -70, -71, -73, -74, -76, -77, -79, -80, -82, -83, -85, -86, -88, -89, -90, -92, -93, -95, -96, -98, -99, -101, -102, -104, -105, -107, -108, -110, -111, -112, -114, -115, -117, -118, -120, -121, -123, -124, -126, -127, -128, -128, -128, -128, -128 };
AI_ARRAY_OBJ_DECLARE(
    neg_1_nl_params, AI_ARRAY_FORMAT_S8,
    neg_1_nl_params_data, neg_1_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  neg_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &neg_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  neg_1_layer, 189,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &neg_1_chain,
  NULL, &neg_1_layer, AI_STATIC, 
  .nl_params = &neg_1_nl_params, 
)


AI_STATIC_CONST ai_u8 slice_3_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_axes, AI_ARRAY_FORMAT_U8,
    slice_3_axes_data, slice_3_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_starts, AI_ARRAY_FORMAT_S16,
    slice_3_starts_data, slice_3_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_ends, AI_ARRAY_FORMAT_S16,
    slice_3_ends_data, slice_3_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_3_layer, 171,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_3_chain,
  NULL, &slice_3_layer, AI_STATIC, 
  .axes = &slice_3_axes, 
  .starts = &slice_3_starts, 
  .ends = &slice_3_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cat_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &neg_1_output, &slice_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cat_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  cat_2_layer, 198,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &cat_2_chain,
  NULL, &cat_2_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &cat_2_output, &unsqueeze_5_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_8_layer, 205,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_8_chain,
  NULL, &mul_8_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &transpose_2_output, &unsqueeze_4_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_7_layer, 170,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_7_chain,
  NULL, &mul_7_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mul_7_output, &mul_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_6_layer, 211,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_6_chain,
  NULL, &add_6_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i16 expand_3_repeats_data[] = { 1, 2, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    expand_3_repeats, AI_ARRAY_FORMAT_S16,
    expand_3_repeats_data, expand_3_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  expand_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_6_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expand_3_layer, 222,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &expand_3_chain,
  NULL, &expand_3_layer, AI_STATIC, 
  .repeats = &expand_3_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_169_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_3_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_169_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_169_layer, 224,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &val_169_chain,
  NULL, &val_169_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_175_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_169_output, &val_172_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_175_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_175_layer, 228,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_175_chain,
  NULL, &val_175_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_66_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_66_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  val_66_layer, 131,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &val_66_chain,
  NULL, &val_66_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_66_output, &mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_chain,
  NULL, &linear_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_1_layer, 158,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_1_chain,
  NULL, &transpose_1_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 slice_2_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_axes, AI_ARRAY_FORMAT_U8,
    slice_2_axes_data, slice_2_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_starts, AI_ARRAY_FORMAT_S16,
    slice_2_starts_data, slice_2_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_ends, AI_ARRAY_FORMAT_S16,
    slice_2_ends_data, slice_2_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_2_layer, 169,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_2_chain,
  NULL, &slice_2_layer, AI_STATIC, 
  .axes = &slice_2_axes, 
  .starts = &slice_2_starts, 
  .ends = &slice_2_ends, 
)


AI_STATIC_CONST ai_i8 neg_nl_params_data[] = { 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -70, -71, -72, -73, -74, -75, -76, -77, -78, -79, -80, -81, -82, -83, -84, -85, -86, -87, -88, -89, -90, -91, -92, -93, -94, -95, -96, -97, -98, -99, -100, -101, -102, -103, -104, -105, -106, -107, -108, -109, -110, -111, -112, -113, -114, -115, -116, -117, -118, -119, -120, -121, -122, -123, -124, -125, -126, -127, -128 };
AI_ARRAY_OBJ_DECLARE(
    neg_nl_params, AI_ARRAY_FORMAT_S8,
    neg_nl_params_data, neg_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  neg_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &neg_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  neg_layer, 188,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &neg_chain,
  NULL, &neg_layer, AI_STATIC, 
  .nl_params = &neg_nl_params, 
)


AI_STATIC_CONST ai_u8 slice_1_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_axes, AI_ARRAY_FORMAT_U8,
    slice_1_axes_data, slice_1_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_starts, AI_ARRAY_FORMAT_S16,
    slice_1_starts_data, slice_1_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_ends, AI_ARRAY_FORMAT_S16,
    slice_1_ends_data, slice_1_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_1_layer, 168,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_1_chain,
  NULL, &slice_1_layer, AI_STATIC, 
  .axes = &slice_1_axes, 
  .starts = &slice_1_starts, 
  .ends = &slice_1_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cat_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &neg_output, &slice_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cat_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  cat_1_layer, 197,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &cat_1_chain,
  NULL, &cat_1_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &cat_1_output, &unsqueeze_5_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_6_layer, 204,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_6_chain,
  NULL, &mul_6_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &transpose_1_output, &unsqueeze_4_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_5_layer, 167,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_5_chain,
  NULL, &mul_5_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mul_5_output, &mul_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_5_layer, 210,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_5_chain,
  NULL, &add_5_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_173_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_5_output, &val_172_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_173_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_173_layer, 216,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_173_chain,
  NULL, &val_173_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_179_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_173_0_0_val_179_conversion_output, &val_175_0_1_val_179_conversion_output, &val_179_bias_0_2_val_179_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_179_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_179_layer, 231,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &val_179_chain,
  NULL, &val_179_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_to_copy_output, &val_34),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_35_layer, 75,
  GATHER_ND_TYPE, 0x0, NULL,
  gather_nd, forward_gather_nd,
  &val_35_chain,
  NULL, &val_35_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  bitwise_and_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &bitwise_and_DequantizeLinear_Output_const, &val_35_0_1_bitwise_and_1_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &bitwise_and_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  bitwise_and_1_layer, 102,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &bitwise_and_1_chain,
  NULL, &bitwise_and_1_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  __bool_fix_inv_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &__bool_fix_one_4D, &bitwise_and_1_0_1___bool_fix_inv_conversion_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &__bool_fix_inv_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  __bool_fix_inv_layer, 114,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &__bool_fix_inv_chain,
  NULL, &__bool_fix_inv_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_178_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &__bool_fix_inv_0_0_val_178_conversion_output, &val_177_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_178_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_178_layer, 120,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_178_chain,
  NULL, &val_178_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_180_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_179_0_0_val_180_conversion_output, &val_178_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_180_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_180_layer, 234,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_180_chain,
  NULL, &val_180_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  scaled_dot_product_attention_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_181_output, &expand_4_0_1_scaled_dot_product_attention_conversion_output0, &scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &scaled_dot_product_attention_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  scaled_dot_product_attention_layer, 243,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &scaled_dot_product_attention_chain,
  NULL, &scaled_dot_product_attention_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &scaled_dot_product_attention_0_0_transpose_4_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_4_layer, 246,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_4_chain,
  NULL, &transpose_4_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_4_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_3_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_3_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_3_layer, 252,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_3_chain,
  NULL, &linear_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &embedding_output0, &linear_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_7_layer, 255,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_7_chain,
  NULL, &add_7_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pow_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_7_0_0_pow_2_conversion_output, &val_60_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pow_2_layer, 258,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &pow_2_chain,
  NULL, &pow_2_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float mean_1_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_1_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_1_neutral_value_data, mean_1_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_1_layer, 259,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_1_chain,
  NULL, &mean_1_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_1_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_1_Mul_0_0_add_8_conversion_output, &val_63_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_8_layer, 262,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_8_chain,
  NULL, &add_8_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_7_output, &rsqrt_1_0_1_mul_9_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_9_layer, 269,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_9_chain,
  NULL, &mul_9_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D, &mul_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_10_layer, 272,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_10_chain,
  NULL, &mul_10_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_4_weights, &linear_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_4_layer, 275,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_4_chain,
  NULL, &linear_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 val_195_nl_params_data[] = { -109, -108, -108, -107, -107, -106, -106, -105, -105, -105, -104, -104, -103, -103, -102, -101, -101, -100, -100, -99, -99, -98, -97, -97, -96, -96, -95, -94, -94, -93, -92, -92, -91, -90, -89, -89, -88, -87, -86, -86, -85, -84, -83, -82, -81, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -59, -58, -57, -56, -55, -54, -52, -51, -50, -49, -47, -46, -45, -44, -42, -41, -40, -38, -37, -36, -34, -33, -31, -30, -29, -27, -26, -25, -23, -22, -20, -19, -17, -16, -14, -13, -11, -10, -9, -7, -6, -4, -3, -1, 0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 36, 37, 38, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 88, 89, 90, 91, 92, 93, 94, 94, 95, 96, 97, 98, 98, 99, 100, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 108, 109, 109, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115, 116, 116, 116, 117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    val_195_nl_params, AI_ARRAY_FORMAT_S8,
    val_195_nl_params_data, val_195_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_195_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_195_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_195_layer, 281,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &val_195_chain,
  NULL, &val_195_layer, AI_STATIC, 
  .nl_params = &val_195_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  silu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &linear_4_output, &val_195_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &silu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  silu_layer, 284,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &silu_chain,
  NULL, &silu_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_5_weights, &linear_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_5_layer, 276,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_5_chain,
  NULL, &linear_5_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &silu_output, &linear_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_11_layer, 287,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_11_chain,
  NULL, &mul_11_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_6_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_6_layer, 290,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_6_chain,
  NULL, &linear_6_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_7_output, &linear_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_9_layer, 293,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_9_chain,
  NULL, &add_9_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pow_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_9_0_0_pow_3_conversion_output, &val_60_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pow_3_layer, 296,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &pow_3_chain,
  NULL, &pow_3_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float mean_2_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_2_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_2_neutral_value_data, mean_2_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_2_layer, 297,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_2_chain,
  NULL, &mean_2_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_2_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_2_Mul_0_0_add_10_conversion_output, &val_63_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_10_layer, 300,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_10_chain,
  NULL, &add_10_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_9_output, &rsqrt_2_0_1_mul_12_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_12_layer, 307,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_12_chain,
  NULL, &mul_12_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D, &mul_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_13_layer, 310,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_13_chain,
  NULL, &mul_13_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_203_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_203_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_203_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_203_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  val_203_layer, 313,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &val_203_chain,
  NULL, &val_203_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_203_output, &mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_7_layer, 322,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_7_chain,
  NULL, &linear_7_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_7_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_5_layer, 340,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_5_chain,
  NULL, &transpose_5_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 slice_6_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_axes, AI_ARRAY_FORMAT_U8,
    slice_6_axes_data, slice_6_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_starts, AI_ARRAY_FORMAT_S16,
    slice_6_starts_data, slice_6_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_ends, AI_ARRAY_FORMAT_S16,
    slice_6_ends_data, slice_6_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_6_layer, 351,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_6_chain,
  NULL, &slice_6_layer, AI_STATIC, 
  .axes = &slice_6_axes, 
  .starts = &slice_6_starts, 
  .ends = &slice_6_ends, 
)


AI_STATIC_CONST ai_i8 neg_2_nl_params_data[] = { 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61, -62, -63, -64, -65, -66, -67, -68, -69, -71, -72, -73, -74, -75, -76, -77, -78, -79, -80, -81, -82, -83, -84, -85, -86, -87, -88, -89, -90, -91, -92, -93, -94, -95, -96, -97, -98, -99, -100, -101, -102, -103, -104, -105, -106, -107, -108, -109, -110, -111, -112, -113, -114, -115, -116, -117, -118, -119, -121, -122, -123, -124, -125, -126, -127, -128, -128, -128, -128, -128, -128, -128 };
AI_ARRAY_OBJ_DECLARE(
    neg_2_nl_params, AI_ARRAY_FORMAT_S8,
    neg_2_nl_params_data, neg_2_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  neg_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &neg_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  neg_2_layer, 370,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &neg_2_chain,
  NULL, &neg_2_layer, AI_STATIC, 
  .nl_params = &neg_2_nl_params, 
)


AI_STATIC_CONST ai_u8 slice_5_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_axes, AI_ARRAY_FORMAT_U8,
    slice_5_axes_data, slice_5_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_starts, AI_ARRAY_FORMAT_S16,
    slice_5_starts_data, slice_5_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_ends, AI_ARRAY_FORMAT_S16,
    slice_5_ends_data, slice_5_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_5_layer, 350,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_5_chain,
  NULL, &slice_5_layer, AI_STATIC, 
  .axes = &slice_5_axes, 
  .starts = &slice_5_starts, 
  .ends = &slice_5_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cat_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &neg_2_output, &slice_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cat_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  cat_3_layer, 379,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &cat_3_chain,
  NULL, &cat_3_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &cat_3_output, &unsqueeze_5_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_15_layer, 386,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_15_chain,
  NULL, &mul_15_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &transpose_5_output, &unsqueeze_4_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_14_layer, 349,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_14_chain,
  NULL, &mul_14_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mul_14_output, &mul_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_11_layer, 392,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_11_chain,
  NULL, &add_11_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_306_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_11_output, &val_172_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_306_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_306_layer, 398,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_306_chain,
  NULL, &val_306_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_211_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_211_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_211_weights, &val_211_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_211_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  val_211_layer, 314,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &val_211_chain,
  NULL, &val_211_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_211_output, &mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_8_layer, 323,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_8_chain,
  NULL, &linear_8_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_8_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_6_layer, 341,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_6_chain,
  NULL, &transpose_6_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_u8 slice_8_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_8_axes, AI_ARRAY_FORMAT_U8,
    slice_8_axes_data, slice_8_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_8_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_8_starts, AI_ARRAY_FORMAT_S16,
    slice_8_starts_data, slice_8_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_8_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_8_ends, AI_ARRAY_FORMAT_S16,
    slice_8_ends_data, slice_8_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_8_layer, 354,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_8_chain,
  NULL, &slice_8_layer, AI_STATIC, 
  .axes = &slice_8_axes, 
  .starts = &slice_8_starts, 
  .ends = &slice_8_ends, 
)


AI_STATIC_CONST ai_i8 neg_3_nl_params_data[] = { 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 125, 124, 123, 122, 121, 119, 118, 117, 116, 115, 114, 112, 111, 110, 109, 108, 107, 105, 104, 103, 102, 101, 100, 98, 97, 96, 95, 94, 93, 91, 90, 89, 88, 87, 86, 84, 83, 82, 81, 80, 79, 77, 76, 75, 74, 73, 72, 70, 69, 68, 67, 66, 65, 63, 62, 61, 60, 59, 58, 56, 55, 54, 53, 52, 51, 49, 48, 47, 46, 45, 44, 42, 41, 40, 39, 38, 37, 35, 34, 33, 32, 31, 30, 28, 27, 26, 25, 24, 23, 21, 20, 19, 18, 17, 16, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 0, -1, -2, -3, -4, -5, -7, -8, -9, -10, -11, -12, -14, -15, -16, -17, -18, -19, -21, -22, -23, -24, -25, -27, -28, -29, -30, -31, -32, -34, -35, -36, -37, -38, -39, -41, -42, -43, -44, -45, -46, -48, -49, -50, -51, -52, -53, -55, -56, -57, -58, -59, -60, -62, -63, -64, -65, -66, -67, -69, -70, -71, -72, -73, -74, -76, -77, -78, -79, -80, -81, -83, -84, -85, -86, -87, -88, -90, -91, -92, -93, -94, -95, -97, -98, -99, -100, -101, -102, -104, -105, -106, -107, -108, -109, -111, -112, -113, -114, -115, -116, -118, -119, -120, -121, -122, -123, -125, -126, -127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128 };
AI_ARRAY_OBJ_DECLARE(
    neg_3_nl_params, AI_ARRAY_FORMAT_S8,
    neg_3_nl_params_data, neg_3_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  neg_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &neg_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  neg_3_layer, 371,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &neg_3_chain,
  NULL, &neg_3_layer, AI_STATIC, 
  .nl_params = &neg_3_nl_params, 
)


AI_STATIC_CONST ai_u8 slice_7_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_axes, AI_ARRAY_FORMAT_U8,
    slice_7_axes_data, slice_7_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_starts, AI_ARRAY_FORMAT_S16,
    slice_7_starts_data, slice_7_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_ends, AI_ARRAY_FORMAT_S16,
    slice_7_ends_data, slice_7_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_7_layer, 353,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_7_chain,
  NULL, &slice_7_layer, AI_STATIC, 
  .axes = &slice_7_axes, 
  .starts = &slice_7_starts, 
  .ends = &slice_7_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  cat_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &neg_3_output, &slice_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &cat_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  cat_4_layer, 380,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &cat_4_chain,
  NULL, &cat_4_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &cat_4_output, &unsqueeze_5_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_17_layer, 387,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_17_chain,
  NULL, &mul_17_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &transpose_6_output, &unsqueeze_4_DequantizeLinear_Output_const),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_16_layer, 352,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_16_chain,
  NULL, &mul_16_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mul_16_output, &mul_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_12_layer, 393,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_12_chain,
  NULL, &add_12_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i16 expand_5_repeats_data[] = { 1, 2, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    expand_5_repeats, AI_ARRAY_FORMAT_S16,
    expand_5_repeats_data, expand_5_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  expand_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_12_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expand_5_layer, 404,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &expand_5_chain,
  NULL, &expand_5_layer, AI_STATIC, 
  .repeats = &expand_5_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_302_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_5_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_302_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_302_layer, 406,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &val_302_chain,
  NULL, &val_302_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_308_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_302_output, &val_172_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_308_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_308_layer, 410,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_308_chain,
  NULL, &val_308_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_312_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_306_0_0_val_312_conversion_output, &val_308_0_1_val_312_conversion_output, &val_312_bias_0_2_val_312_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_312_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_312_layer, 413,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &val_312_chain,
  NULL, &val_312_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_313_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_312_0_0_val_313_conversion_output, &val_178_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_313_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_313_layer, 416,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &val_313_chain,
  NULL, &val_313_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_219_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_219_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_219_weights, &val_211_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_219_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  val_219_layer, 315,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &val_219_chain,
  NULL, &val_219_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &val_219_output, &mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  linear_9_layer, 324,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &linear_9_chain,
  NULL, &linear_9_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_9_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_7_layer, 342,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_7_chain,
  NULL, &transpose_7_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i16 expand_6_repeats_data[] = { 1, 2, 1, 1, 1 };
AI_ARRAY_OBJ_DECLARE(
    expand_6_repeats, AI_ARRAY_FORMAT_S16,
    expand_6_repeats_data, expand_6_repeats_data, 5, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  expand_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_7_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expand_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expand_6_layer, 372,
  TILE_TYPE, 0x0, NULL,
  tile, forward_tile,
  &expand_6_chain,
  NULL, &expand_6_layer, AI_STATIC, 
  .repeats = &expand_6_repeats, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  scaled_dot_product_attention_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &val_314_output, &expand_6_0_1_scaled_dot_product_attention_1_conversion_output0, &scaled_dot_product_attention_1_bias_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &scaled_dot_product_attention_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  scaled_dot_product_attention_1_layer, 425,
  MATMUL_TYPE, 0x0, NULL,
  matmul, forward_matmul,
  &scaled_dot_product_attention_1_chain,
  NULL, &scaled_dot_product_attention_1_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 0, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &scaled_dot_product_attention_1_0_0_transpose_8_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_8_layer, 428,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_8_chain,
  NULL, &transpose_8_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_8_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_10_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_10_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_10_layer, 434,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_10_chain,
  NULL, &linear_10_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_9_output, &linear_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_13_layer, 437,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_13_chain,
  NULL, &add_13_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pow_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_13_0_0_pow_4_conversion_output, &val_60_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pow_4_layer, 440,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &pow_4_chain,
  NULL, &pow_4_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float mean_3_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_3_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_3_neutral_value_data, mean_3_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_3_layer, 441,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_3_chain,
  NULL, &mean_3_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_3_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_3_Mul_0_0_add_14_conversion_output, &val_63_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_14_layer, 444,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_14_chain,
  NULL, &add_14_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_13_output, &rsqrt_3_0_1_mul_18_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_18_layer, 451,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_18_chain,
  NULL, &mul_18_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D, &mul_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_19_layer, 454,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_19_chain,
  NULL, &mul_19_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_11_weights, &linear_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_11_layer, 457,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_11_chain,
  NULL, &linear_11_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 val_328_nl_params_data[] = { -103, -103, -102, -101, -101, -100, -100, -99, -99, -98, -97, -97, -96, -95, -95, -94, -93, -93, -92, -91, -91, -90, -89, -88, -87, -87, -86, -85, -84, -83, -82, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -63, -62, -61, -60, -59, -58, -56, -55, -54, -53, -51, -50, -49, -48, -46, -45, -44, -42, -41, -40, -38, -37, -36, -34, -33, -31, -30, -29, -27, -26, -24, -23, -21, -20, -18, -17, -15, -14, -12, -11, -9, -8, -6, -5, -3, -2, 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 35, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 90, 91, 92, 93, 94, 94, 95, 96, 97, 97, 98, 99, 100, 100, 101, 102, 102, 103, 103, 104, 105, 105, 106, 106, 107, 108, 108, 109, 109, 110, 110, 111, 111, 112, 112, 112, 113, 113, 114, 114, 115, 115, 115, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    val_328_nl_params, AI_ARRAY_FORMAT_S8,
    val_328_nl_params_data, val_328_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  val_328_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &val_328_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  val_328_layer, 463,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &val_328_chain,
  NULL, &val_328_layer, AI_STATIC, 
  .nl_params = &val_328_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  silu_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &linear_11_output, &val_328_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &silu_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  silu_1_layer, 466,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &silu_1_chain,
  NULL, &silu_1_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_12_weights, &linear_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_12_layer, 458,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_12_chain,
  NULL, &linear_12_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &silu_1_output, &linear_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_20_layer, 469,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_20_chain,
  NULL, &mul_20_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  linear_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &linear_13_weights, &val_66_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &linear_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  linear_13_layer, 472,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &linear_13_chain,
  NULL, &linear_13_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_13_output, &linear_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_15_layer, 475,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_15_chain,
  NULL, &add_15_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pow_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_15_0_0_pow_5_conversion_output, &val_60_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pow_5_layer, 478,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &pow_5_chain,
  NULL, &pow_5_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float mean_4_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_4_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_4_neutral_value_data, mean_4_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pow_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_4_layer, 479,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_4_chain,
  NULL, &mean_4_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_4_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_4_Mul_0_0_add_16_conversion_output, &val_63_DequantizeLinear_Output_const_3D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_16_layer, 482,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &add_16_chain,
  NULL, &add_16_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &add_15_output, &rsqrt_4_0_1_mul_21_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_21_layer, 489,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_21_chain,
  NULL, &mul_21_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mul_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mdl_model_norm_weight_DequantizeLinear_Output_const_3D, &mul_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mul_22_layer, 492,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &mul_22_chain,
  NULL, &mul_22_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mul_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 495,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_cast__to_copy(_stai_network_context* net_ctx)
{
  attention_mask_output_array.data = AI_PTR(net_ctx->_inputs[1] + 0);
  attention_mask_output_array.data_start = AI_PTR(net_ctx->_inputs[1] + 0);
  _to_copy_output_array.data = AI_PTR(net_ctx->_activations[0] + 41100);
  _to_copy_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41100);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, { attention_mask_output.data->data});
  forward_cast(&_to_copy_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { _to_copy_output.data->data});
}
void forward_lite_eltwise___ids_floored(_stai_network_context* net_ctx)
{
  input_ids_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_ids_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  __ids_floor_zero_2D_array.data = AI_PTR(net_ctx->_weights[0] + 24);
  __ids_floor_zero_2D_array.data_start = AI_PTR(net_ctx->_weights[0] + 24);
  __ids_floored_output_array.data = AI_PTR(net_ctx->_activations[0] + 40972);
  __ids_floored_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40972);
  _STAI_NETWORK_EVENT_NODE_START_CB(55, 2, { input_ids_output.data->data,__ids_floor_zero_2D.data->data});
  forward_eltwise(&__ids_floored_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(55, 1, { __ids_floored_output.data->data});
}
void forward_lite_eltwise_input_ids_clipped(_stai_network_context* net_ctx)
{
  __ids_floored_output_array.data = AI_PTR(net_ctx->_activations[0] + 40972);
  __ids_floored_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40972);
  __ids_cap_val_2D_array.data = AI_PTR(net_ctx->_weights[0] + 28);
  __ids_cap_val_2D_array.data_start = AI_PTR(net_ctx->_weights[0] + 28);
  input_ids_clipped_output_array.data = AI_PTR(net_ctx->_activations[0] + 40832);
  input_ids_clipped_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40832);
  _STAI_NETWORK_EVENT_NODE_START_CB(76, 2, { __ids_floored_output.data->data,__ids_cap_val_2D.data->data});
  forward_eltwise(&input_ids_clipped_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(76, 1, { input_ids_clipped_output.data->data});
}
void forward_lite_gather_embedding(_stai_network_context* net_ctx)
{
  mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 2356);
  mdl_model_embed_tokens_weight_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 2356);
  input_ids_clipped_output_array.data = AI_PTR(net_ctx->_activations[0] + 40832);
  input_ids_clipped_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40832);
  embedding_output_array.data = AI_PTR(net_ctx->_activations[0] + 303384);
  embedding_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 303384);
  _STAI_NETWORK_EVENT_NODE_START_CB(97, 2, { mdl_model_embed_tokens_weight_DequantizeLinear_Output_const.data->data,input_ids_clipped_output.data->data});
  forward_gather(&embedding_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(97, 1, { embedding_output.data->data});
}
void forward_lite_eltwise_pow_1(_stai_network_context* net_ctx)
{
  embedding_0_0_pow_1_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  embedding_0_0_pow_1_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  val_60_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  val_60_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  pow_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  pow_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(103, 2, { embedding_0_0_pow_1_conversion_output0.data->data,val_60_3D.data->data});
  forward_eltwise(&pow_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(103, 1, { pow_1_output.data->data});
}
void forward_lite_reduce_mean(_stai_network_context* net_ctx)
{
  pow_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  pow_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  mean_output_array.data = AI_PTR(net_ctx->_activations[0] + 40972);
  mean_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40972);
  _STAI_NETWORK_EVENT_NODE_START_CB(105, 1, { pow_1_output.data->data});
  forward_reduce(&mean_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(105, 1, { mean_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_4(_stai_network_context* net_ctx)
{
  mean_Mul_0_0_add_4_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 8320);
  mean_Mul_0_0_add_4_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8320);
  val_63_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 44);
  val_63_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 44);
  add_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 40972);
  add_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40972);
  _STAI_NETWORK_EVENT_NODE_START_CB(111, 2, { mean_Mul_0_0_add_4_conversion_output.data->data,val_63_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&add_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(111, 1, { add_4_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_3(_stai_network_context* net_ctx)
{
  embedding_output_array.data = AI_PTR(net_ctx->_activations[0] + 303384);
  embedding_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 303384);
  rsqrt_0_1_mul_3_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 40928);
  rsqrt_0_1_mul_3_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40928);
  mul_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(125, 2, { embedding_output0.data->data,rsqrt_0_1_mul_3_conversion_output.data->data});
  forward_eltwise_integer_INT8(&mul_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(125, 1, { mul_3_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_4(_stai_network_context* net_ctx)
{
  mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 1588);
  mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 1588);
  mul_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 32768);
  mul_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(128, 2, { mdl_model_layers_0_input_layernorm_weight_DequantizeLinear_Output_const_3D.data->data,mul_3_output.data->data});
  forward_eltwise_integer_INT8(&mul_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(128, 1, { mul_4_output.data->data});
}
void forward_lite_matmul_val_82(_stai_network_context* net_ctx)
{
  mul_4_0_0_val_74_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_4_0_0_val_74_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 172312);
  val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 172312);
  val_82_bias_0_2_val_82_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41236);
  val_82_bias_0_2_val_82_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41236);
  val_82_output_array.data = AI_PTR(net_ctx->_activations[0] + 311576);
  val_82_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 311576);
  _STAI_NETWORK_EVENT_NODE_START_CB(133, 3, { mul_4_0_0_val_74_conversion_output0.data->data,val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_output0.data->data,val_82_bias_0_2_val_82_conversion_output.data->data});
  forward_matmul(&val_82_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(133, 1, { val_82_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear_2(_stai_network_context* net_ctx)
{
  val_82_0_0_linear_2_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 172312);
  val_82_0_0_linear_2_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 172312);
  mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 1844);
  mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 1844);
  linear_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 176408);
  linear_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 176408);
  _STAI_NETWORK_EVENT_NODE_START_CB(142, 2, { val_82_0_0_linear_2_conversion_output0.data->data,mdl_model_layers_0_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(142, 1, { linear_2_output.data->data});
}
void forward_lite_transpose_transpose_3(_stai_network_context* net_ctx)
{
  linear_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 176408);
  linear_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 176408);
  transpose_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 172312);
  transpose_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 172312);
  _STAI_NETWORK_EVENT_NODE_START_CB(160, 1, { linear_2_output0.data->data});
  forward_transpose(&transpose_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(160, 1, { transpose_3_output.data->data});
}
void forward_lite_tile_expand_4(_stai_network_context* net_ctx)
{
  transpose_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 172312);
  transpose_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 172312);
  expand_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 176408);
  expand_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 176408);
  _STAI_NETWORK_EVENT_NODE_START_CB(190, 1, { transpose_3_output0.data->data});
  forward_tile(&expand_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(190, 1, { expand_4_output.data->data});
}
void forward_lite_matmul_val_74(_stai_network_context* net_ctx)
{
  mul_4_0_0_val_74_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_4_0_0_val_74_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41240);
  val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41240);
  val_74_bias_0_2_val_74_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 40960);
  val_74_bias_0_2_val_74_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40960);
  val_74_output_array.data = AI_PTR(net_ctx->_activations[0] + 217368);
  val_74_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 217368);
  _STAI_NETWORK_EVENT_NODE_START_CB(132, 3, { mul_4_0_0_val_74_conversion_output0.data->data,val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_output0.data->data,val_74_bias_0_2_val_74_conversion_output.data->data});
  forward_matmul(&val_74_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(132, 1, { val_74_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear_1(_stai_network_context* net_ctx)
{
  val_74_0_0_linear_1_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_74_0_0_linear_1_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 1972);
  mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 1972);
  linear_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  linear_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(141, 2, { val_74_0_0_linear_1_conversion_output0.data->data,mdl_model_layers_0_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(141, 1, { linear_1_output.data->data});
}
void forward_lite_transpose_transpose_2(_stai_network_context* net_ctx)
{
  linear_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  linear_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  transpose_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(159, 1, { linear_1_output0.data->data});
  forward_transpose(&transpose_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(159, 1, { transpose_2_output.data->data});
}
void forward_lite_slice_slice_4(_stai_network_context* net_ctx)
{
  transpose_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  slice_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(172, 1, { transpose_2_output.data->data});
  forward_slice(&slice_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(172, 1, { slice_4_output.data->data});
}
void forward_lite_nl_integer_neg_1(_stai_network_context* net_ctx)
{
  slice_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  slice_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  neg_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 6144);
  neg_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6144);
  _STAI_NETWORK_EVENT_NODE_START_CB(189, 1, { slice_4_output.data->data});
  forward_nl_integer(&neg_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(189, 1, { neg_1_output.data->data});
}
void forward_lite_slice_slice_3(_stai_network_context* net_ctx)
{
  transpose_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  slice_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(171, 1, { transpose_2_output.data->data});
  forward_slice(&slice_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(171, 1, { slice_3_output.data->data});
}
void forward_lite_concat_cat_2(_stai_network_context* net_ctx)
{
  neg_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 6144);
  neg_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6144);
  slice_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  slice_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  cat_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  cat_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(198, 2, { neg_1_output.data->data,slice_3_output.data->data});
  forward_concat(&cat_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(198, 1, { cat_2_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_8(_stai_network_context* net_ctx)
{
  cat_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  cat_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  unsqueeze_5_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 101172);
  unsqueeze_5_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 101172);
  mul_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  mul_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(205, 2, { cat_2_output.data->data,unsqueeze_5_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_8_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(205, 1, { mul_8_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_7(_stai_network_context* net_ctx)
{
  transpose_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  unsqueeze_4_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 99124);
  unsqueeze_4_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 99124);
  mul_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(170, 2, { transpose_2_output.data->data,unsqueeze_4_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(170, 1, { mul_7_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_6(_stai_network_context* net_ctx)
{
  mul_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  mul_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  add_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(211, 2, { mul_7_output.data->data,mul_8_output.data->data});
  forward_eltwise_integer_INT8(&add_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(211, 1, { add_6_output.data->data});
}
void forward_lite_tile_expand_3(_stai_network_context* net_ctx)
{
  add_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  expand_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  expand_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(222, 1, { add_6_output0.data->data});
  forward_tile(&expand_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(222, 1, { expand_3_output.data->data});
}
void forward_lite_transpose_val_169(_stai_network_context* net_ctx)
{
  expand_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 4096);
  expand_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4096);
  val_169_output_array.data = AI_PTR(net_ctx->_activations[0] + 12288);
  val_169_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12288);
  _STAI_NETWORK_EVENT_NODE_START_CB(224, 1, { expand_3_output0.data->data});
  forward_transpose(&val_169_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(224, 1, { val_169_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_175(_stai_network_context* net_ctx)
{
  val_169_output_array.data = AI_PTR(net_ctx->_activations[0] + 12288);
  val_169_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12288);
  val_172_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 48);
  val_172_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 48);
  val_175_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_175_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(228, 2, { val_169_output.data->data,val_172_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&val_175_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(228, 1, { val_175_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_val_66(_stai_network_context* net_ctx)
{
  mul_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 32768);
  mul_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32768);
  val_66_weights_array.data = AI_PTR(net_ctx->_weights[0] + 168764);
  val_66_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 168764);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_66_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_66_output_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  val_66_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  _STAI_NETWORK_EVENT_NODE_START_CB(131, 1, { mul_4_output.data->data});
  forward_conv2d_integer_SSSA(&val_66_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(131, 1, { val_66_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear(_stai_network_context* net_ctx)
{
  val_66_output_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  val_66_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 2100);
  mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 2100);
  linear_output_array.data = AI_PTR(net_ctx->_activations[0] + 9216);
  linear_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9216);
  _STAI_NETWORK_EVENT_NODE_START_CB(140, 2, { val_66_output.data->data,mdl_model_layers_0_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(140, 1, { linear_output.data->data});
}
void forward_lite_transpose_transpose_1(_stai_network_context* net_ctx)
{
  linear_output_array.data = AI_PTR(net_ctx->_activations[0] + 9216);
  linear_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9216);
  transpose_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(158, 1, { linear_output0.data->data});
  forward_transpose(&transpose_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(158, 1, { transpose_1_output.data->data});
}
void forward_lite_slice_slice_2(_stai_network_context* net_ctx)
{
  transpose_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  slice_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(169, 1, { transpose_1_output.data->data});
  forward_slice(&slice_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(169, 1, { slice_2_output.data->data});
}
void forward_lite_nl_integer_neg(_stai_network_context* net_ctx)
{
  slice_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  slice_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  neg_output_array.data = AI_PTR(net_ctx->_activations[0] + 12288);
  neg_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12288);
  _STAI_NETWORK_EVENT_NODE_START_CB(188, 1, { slice_2_output.data->data});
  forward_nl_integer(&neg_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(188, 1, { neg_output.data->data});
}
void forward_lite_slice_slice_1(_stai_network_context* net_ctx)
{
  transpose_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  slice_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(168, 1, { transpose_1_output.data->data});
  forward_slice(&slice_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(168, 1, { slice_1_output.data->data});
}
void forward_lite_concat_cat_1(_stai_network_context* net_ctx)
{
  neg_output_array.data = AI_PTR(net_ctx->_activations[0] + 12288);
  neg_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12288);
  slice_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  slice_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  cat_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  cat_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  _STAI_NETWORK_EVENT_NODE_START_CB(197, 2, { neg_output.data->data,slice_1_output.data->data});
  forward_concat(&cat_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(197, 1, { cat_1_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_6(_stai_network_context* net_ctx)
{
  cat_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  cat_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  unsqueeze_5_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 101172);
  unsqueeze_5_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 101172);
  mul_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(204, 2, { cat_1_output.data->data,unsqueeze_5_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(204, 1, { mul_6_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_5(_stai_network_context* net_ctx)
{
  transpose_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  transpose_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  unsqueeze_4_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 99124);
  unsqueeze_4_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 99124);
  mul_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  mul_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  _STAI_NETWORK_EVENT_NODE_START_CB(167, 2, { transpose_1_output.data->data,unsqueeze_4_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(167, 1, { mul_5_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_5(_stai_network_context* net_ctx)
{
  mul_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  mul_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  mul_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mul_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  add_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(210, 2, { mul_5_output.data->data,mul_6_output.data->data});
  forward_eltwise_integer_INT8(&add_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(210, 1, { add_5_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_173(_stai_network_context* net_ctx)
{
  add_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_172_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 48);
  val_172_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 48);
  val_173_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  val_173_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(216, 2, { add_5_output.data->data,val_172_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&val_173_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(216, 1, { val_173_output.data->data});
}
void forward_lite_matmul_val_179(_stai_network_context* net_ctx)
{
  val_173_0_0_val_179_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 74004);
  val_173_0_0_val_179_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74004);
  val_175_0_1_val_179_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41236);
  val_175_0_1_val_179_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41236);
  val_179_bias_0_2_val_179_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  val_179_bias_0_2_val_179_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_179_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_179_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(231, 3, { val_173_0_0_val_179_conversion_output.data->data,val_175_0_1_val_179_conversion_output.data->data,val_179_bias_0_2_val_179_conversion_output.data->data});
  forward_matmul(&val_179_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(231, 1, { val_179_output.data->data});
}
void forward_lite_gather_nd_val_35(_stai_network_context* net_ctx)
{
  _to_copy_output_array.data = AI_PTR(net_ctx->_activations[0] + 41100);
  _to_copy_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41100);
  val_34_array.data = AI_PTR(net_ctx->_weights[0] + 235324);
  val_34_array.data_start = AI_PTR(net_ctx->_weights[0] + 235324);
  val_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 2, { _to_copy_output.data->data,val_34.data->data});
  forward_gather_nd(&val_35_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, { val_35_output.data->data});
}
void forward_lite_eltwise_integer_INT8_bitwise_and_1(_stai_network_context* net_ctx)
{
  bitwise_and_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 98100);
  bitwise_and_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 98100);
  val_35_0_1_bitwise_and_1_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  val_35_0_1_bitwise_and_1_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  bitwise_and_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 160);
  bitwise_and_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 160);
  _STAI_NETWORK_EVENT_NODE_START_CB(102, 2, { bitwise_and_DequantizeLinear_Output_const.data->data,val_35_0_1_bitwise_and_1_conversion_output.data->data});
  forward_eltwise_integer_INT8(&bitwise_and_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(102, 1, { bitwise_and_1_output.data->data});
}
void forward_lite_eltwise___bool_fix_inv(_stai_network_context* net_ctx)
{
  __bool_fix_one_4D_array.data = AI_PTR(net_ctx->_weights[0] + 36);
  __bool_fix_one_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 36);
  bitwise_and_1_0_1___bool_fix_inv_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 1184);
  bitwise_and_1_0_1___bool_fix_inv_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1184);
  __bool_fix_inv_output_array.data = AI_PTR(net_ctx->_activations[0] + 5280);
  __bool_fix_inv_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5280);
  _STAI_NETWORK_EVENT_NODE_START_CB(114, 2, { __bool_fix_one_4D.data->data,bitwise_and_1_0_1___bool_fix_inv_conversion_output0.data->data});
  forward_eltwise(&__bool_fix_inv_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(114, 1, { __bool_fix_inv_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_178(_stai_network_context* net_ctx)
{
  __bool_fix_inv_0_0_val_178_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  __bool_fix_inv_0_0_val_178_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_177_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 40);
  val_177_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 40);
  val_178_output_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  val_178_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  _STAI_NETWORK_EVENT_NODE_START_CB(120, 2, { __bool_fix_inv_0_0_val_178_conversion_output.data->data,val_177_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&val_178_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(120, 1, { val_178_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_180(_stai_network_context* net_ctx)
{
  val_179_0_0_val_180_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  val_179_0_0_val_180_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  val_178_output_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  val_178_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  val_180_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  val_180_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  _STAI_NETWORK_EVENT_NODE_START_CB(234, 2, { val_179_0_0_val_180_conversion_output.data->data,val_178_output.data->data});
  forward_eltwise_integer_INT8(&val_180_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(234, 1, { val_180_output.data->data});
}
void forward_lite_matmul_scaled_dot_product_attention(_stai_network_context* net_ctx)
{
  val_181_output_array.data = AI_PTR(net_ctx->_activations[0] + 22528);
  val_181_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 22528);
  expand_4_0_1_scaled_dot_product_attention_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 184600);
  expand_4_0_1_scaled_dot_product_attention_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 184600);
  scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 40964);
  scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40964);
  scaled_dot_product_attention_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  scaled_dot_product_attention_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(243, 3, { val_181_output.data->data,expand_4_0_1_scaled_dot_product_attention_conversion_output0.data->data,scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_output.data->data});
  forward_matmul(&scaled_dot_product_attention_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(243, 1, { scaled_dot_product_attention_output.data->data});
}
void forward_lite_transpose_transpose_4(_stai_network_context* net_ctx)
{
  scaled_dot_product_attention_0_0_transpose_4_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  scaled_dot_product_attention_0_0_transpose_4_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  transpose_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(246, 1, { scaled_dot_product_attention_0_0_transpose_4_conversion_output.data->data});
  forward_transpose(&transpose_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(246, 1, { transpose_4_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_3(_stai_network_context* net_ctx)
{
  transpose_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  linear_3_weights_array.data = AI_PTR(net_ctx->_weights[0] + 235580);
  linear_3_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 235580);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  linear_3_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_3_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  linear_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  _STAI_NETWORK_EVENT_NODE_START_CB(252, 1, { transpose_4_output0.data->data});
  forward_conv2d_integer_SSSA(&linear_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(252, 1, { linear_3_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_7(_stai_network_context* net_ctx)
{
  embedding_output_array.data = AI_PTR(net_ctx->_activations[0] + 303384);
  embedding_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 303384);
  linear_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  linear_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  add_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(255, 2, { embedding_output0.data->data,linear_3_output.data->data});
  forward_eltwise_integer_INT8(&add_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(255, 1, { add_7_output.data->data});
}
void forward_lite_eltwise_pow_2(_stai_network_context* net_ctx)
{
  add_7_0_0_pow_2_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  add_7_0_0_pow_2_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_60_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  val_60_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  pow_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  pow_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  _STAI_NETWORK_EVENT_NODE_START_CB(258, 2, { add_7_0_0_pow_2_conversion_output.data->data,val_60_3D.data->data});
  forward_eltwise(&pow_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(258, 1, { pow_2_output.data->data});
}
void forward_lite_reduce_mean_1(_stai_network_context* net_ctx)
{
  pow_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  pow_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  mean_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(259, 1, { pow_2_output.data->data});
  forward_reduce(&mean_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(259, 1, { mean_1_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_8(_stai_network_context* net_ctx)
{
  mean_1_Mul_0_0_add_8_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_1_Mul_0_0_add_8_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_63_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 44);
  val_63_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 44);
  add_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 32);
  add_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32);
  _STAI_NETWORK_EVENT_NODE_START_CB(262, 2, { mean_1_Mul_0_0_add_8_conversion_output.data->data,val_63_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&add_8_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(262, 1, { add_8_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_9(_stai_network_context* net_ctx)
{
  add_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  rsqrt_1_0_1_mul_9_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  rsqrt_1_0_1_mul_9_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  mul_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  mul_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  _STAI_NETWORK_EVENT_NODE_START_CB(269, 2, { add_7_output.data->data,rsqrt_1_0_1_mul_9_conversion_output.data->data});
  forward_eltwise_integer_INT8(&mul_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(269, 1, { mul_9_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_10(_stai_network_context* net_ctx)
{
  mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 1332);
  mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 1332);
  mul_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  mul_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  mul_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  _STAI_NETWORK_EVENT_NODE_START_CB(272, 2, { mdl_model_layers_0_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D.data->data,mul_9_output.data->data});
  forward_eltwise_integer_INT8(&mul_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(272, 1, { mul_10_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_4(_stai_network_context* net_ctx)
{
  mul_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  linear_4_weights_array.data = AI_PTR(net_ctx->_weights[0] + 301116);
  linear_4_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 301116);
  linear_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_4_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_4_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(275, 1, { mul_10_output.data->data});
  forward_conv2d_integer_SSSA(&linear_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(275, 1, { linear_4_output.data->data});
}
void forward_lite_nl_integer_val_195(_stai_network_context* net_ctx)
{
  linear_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_195_output_array.data = AI_PTR(net_ctx->_activations[0] + 57616);
  val_195_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57616);
  _STAI_NETWORK_EVENT_NODE_START_CB(281, 1, { linear_4_output.data->data});
  forward_nl_integer(&val_195_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(281, 1, { val_195_output.data->data});
}
void forward_lite_eltwise_integer_INT8_silu(_stai_network_context* net_ctx)
{
  linear_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_195_output_array.data = AI_PTR(net_ctx->_activations[0] + 57616);
  val_195_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57616);
  silu_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  silu_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  _STAI_NETWORK_EVENT_NODE_START_CB(284, 2, { linear_4_output.data->data,val_195_output.data->data});
  forward_eltwise_integer_INT8(&silu_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(284, 1, { silu_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_5(_stai_network_context* net_ctx)
{
  mul_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  linear_5_weights_array.data = AI_PTR(net_ctx->_weights[0] + 434236);
  linear_5_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 434236);
  linear_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_5_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_5_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(276, 1, { mul_10_output.data->data});
  forward_conv2d_integer_SSSA(&linear_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(276, 1, { linear_5_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_11(_stai_network_context* net_ctx)
{
  silu_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  silu_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  linear_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  mul_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  _STAI_NETWORK_EVENT_NODE_START_CB(287, 2, { silu_output.data->data,linear_5_output.data->data});
  forward_eltwise_integer_INT8(&mul_11_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(287, 1, { mul_11_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_6(_stai_network_context* net_ctx)
{
  mul_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  linear_6_weights_array.data = AI_PTR(net_ctx->_weights[0] + 565308);
  linear_6_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 565308);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  linear_6_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  linear_6_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  linear_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(290, 1, { mul_11_output.data->data});
  forward_conv2d_integer_SSSA(&linear_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(290, 1, { linear_6_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_9(_stai_network_context* net_ctx)
{
  add_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  linear_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  linear_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  add_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  add_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  _STAI_NETWORK_EVENT_NODE_START_CB(293, 2, { add_7_output.data->data,linear_6_output.data->data});
  forward_eltwise_integer_INT8(&add_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(293, 1, { add_9_output.data->data});
}
void forward_lite_eltwise_pow_3(_stai_network_context* net_ctx)
{
  add_9_0_0_pow_3_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  add_9_0_0_pow_3_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_60_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  val_60_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  pow_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  pow_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  _STAI_NETWORK_EVENT_NODE_START_CB(296, 2, { add_9_0_0_pow_3_conversion_output.data->data,val_60_3D.data->data});
  forward_eltwise(&pow_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(296, 1, { pow_3_output.data->data});
}
void forward_lite_reduce_mean_2(_stai_network_context* net_ctx)
{
  pow_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  pow_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  mean_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(297, 1, { pow_3_output.data->data});
  forward_reduce(&mean_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(297, 1, { mean_2_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_10(_stai_network_context* net_ctx)
{
  mean_2_Mul_0_0_add_10_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_2_Mul_0_0_add_10_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_63_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 44);
  val_63_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 44);
  add_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 32);
  add_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32);
  _STAI_NETWORK_EVENT_NODE_START_CB(300, 2, { mean_2_Mul_0_0_add_10_conversion_output.data->data,val_63_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&add_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(300, 1, { add_10_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_12(_stai_network_context* net_ctx)
{
  add_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  add_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  rsqrt_2_0_1_mul_12_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  rsqrt_2_0_1_mul_12_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  mul_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  mul_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(307, 2, { add_9_output.data->data,rsqrt_2_0_1_mul_12_conversion_output.data->data});
  forward_eltwise_integer_INT8(&mul_12_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(307, 1, { mul_12_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_13(_stai_network_context* net_ctx)
{
  mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 564);
  mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 564);
  mul_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  mul_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  mul_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  _STAI_NETWORK_EVENT_NODE_START_CB(310, 2, { mdl_model_layers_1_input_layernorm_weight_DequantizeLinear_Output_const_3D.data->data,mul_12_output.data->data});
  forward_eltwise_integer_INT8(&mul_13_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(310, 1, { mul_13_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_val_203(_stai_network_context* net_ctx)
{
  mul_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  val_203_weights_array.data = AI_PTR(net_ctx->_weights[0] + 696380);
  val_203_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 696380);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  val_203_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_203_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_203_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_203_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(313, 1, { mul_13_output.data->data});
  forward_conv2d_integer_SSSA(&val_203_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(313, 1, { val_203_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear_7(_stai_network_context* net_ctx)
{
  val_203_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_203_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 1076);
  mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 1076);
  linear_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(322, 2, { val_203_output.data->data,mdl_model_layers_1_self_attn_q_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(322, 1, { linear_7_output.data->data});
}
void forward_lite_transpose_transpose_5(_stai_network_context* net_ctx)
{
  linear_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  transpose_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(340, 1, { linear_7_output0.data->data});
  forward_transpose(&transpose_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(340, 1, { transpose_5_output.data->data});
}
void forward_lite_slice_slice_6(_stai_network_context* net_ctx)
{
  transpose_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  slice_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  slice_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(351, 1, { transpose_5_output.data->data});
  forward_slice(&slice_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(351, 1, { slice_6_output.data->data});
}
void forward_lite_nl_integer_neg_2(_stai_network_context* net_ctx)
{
  slice_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  slice_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  neg_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 30720);
  neg_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 30720);
  _STAI_NETWORK_EVENT_NODE_START_CB(370, 1, { slice_6_output.data->data});
  forward_nl_integer(&neg_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(370, 1, { neg_2_output.data->data});
}
void forward_lite_slice_slice_5(_stai_network_context* net_ctx)
{
  transpose_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  slice_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  slice_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(350, 1, { transpose_5_output.data->data});
  forward_slice(&slice_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(350, 1, { slice_5_output.data->data});
}
void forward_lite_concat_cat_3(_stai_network_context* net_ctx)
{
  neg_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 30720);
  neg_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 30720);
  slice_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  slice_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  cat_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  cat_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(379, 2, { neg_2_output.data->data,slice_5_output.data->data});
  forward_concat(&cat_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(379, 1, { cat_3_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_15(_stai_network_context* net_ctx)
{
  cat_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  cat_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  unsqueeze_5_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 101172);
  unsqueeze_5_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 101172);
  mul_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(386, 2, { cat_3_output.data->data,unsqueeze_5_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_15_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(386, 1, { mul_15_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_14(_stai_network_context* net_ctx)
{
  transpose_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  unsqueeze_4_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 99124);
  unsqueeze_4_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 99124);
  mul_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  mul_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  _STAI_NETWORK_EVENT_NODE_START_CB(349, 2, { transpose_5_output.data->data,unsqueeze_4_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_14_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(349, 1, { mul_14_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_11(_stai_network_context* net_ctx)
{
  mul_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  mul_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  mul_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  add_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(392, 2, { mul_14_output.data->data,mul_15_output.data->data});
  forward_eltwise_integer_INT8(&add_11_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(392, 1, { add_11_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_306(_stai_network_context* net_ctx)
{
  add_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  val_172_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 48);
  val_172_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 48);
  val_306_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  val_306_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(398, 2, { add_11_output.data->data,val_172_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&val_306_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(398, 1, { val_306_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_val_211(_stai_network_context* net_ctx)
{
  mul_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  val_211_weights_array.data = AI_PTR(net_ctx->_weights[0] + 761916);
  val_211_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 761916);
  val_211_bias_array.data = AI_PTR(net_ctx->_weights[0] + 794684);
  val_211_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 794684);
  val_211_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_211_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_211_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_211_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(314, 1, { mul_13_output.data->data});
  forward_conv2d_integer_SSSA(&val_211_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(314, 1, { val_211_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear_8(_stai_network_context* net_ctx)
{
  val_211_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_211_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 948);
  mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 948);
  linear_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  linear_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(323, 2, { val_211_output.data->data,mdl_model_layers_1_self_attn_k_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_8_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(323, 1, { linear_8_output.data->data});
}
void forward_lite_transpose_transpose_6(_stai_network_context* net_ctx)
{
  linear_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  linear_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  transpose_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(341, 1, { linear_8_output0.data->data});
  forward_transpose(&transpose_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(341, 1, { transpose_6_output.data->data});
}
void forward_lite_slice_slice_8(_stai_network_context* net_ctx)
{
  transpose_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  slice_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  slice_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(354, 1, { transpose_6_output.data->data});
  forward_slice(&slice_8_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(354, 1, { slice_8_output.data->data});
}
void forward_lite_nl_integer_neg_3(_stai_network_context* net_ctx)
{
  slice_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  slice_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  neg_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  neg_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  _STAI_NETWORK_EVENT_NODE_START_CB(371, 1, { slice_8_output.data->data});
  forward_nl_integer(&neg_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(371, 1, { neg_3_output.data->data});
}
void forward_lite_slice_slice_7(_stai_network_context* net_ctx)
{
  transpose_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(353, 1, { transpose_6_output.data->data});
  forward_slice(&slice_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(353, 1, { slice_7_output.data->data});
}
void forward_lite_concat_cat_4(_stai_network_context* net_ctx)
{
  neg_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  neg_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  cat_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  cat_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(380, 2, { neg_3_output.data->data,slice_7_output.data->data});
  forward_concat(&cat_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(380, 1, { cat_4_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_17(_stai_network_context* net_ctx)
{
  cat_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  cat_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  unsqueeze_5_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 101172);
  unsqueeze_5_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 101172);
  mul_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  mul_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(387, 2, { cat_4_output.data->data,unsqueeze_5_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_17_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(387, 1, { mul_17_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_16(_stai_network_context* net_ctx)
{
  transpose_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  unsqueeze_4_DequantizeLinear_Output_const_array.data = AI_PTR(net_ctx->_weights[0] + 99124);
  unsqueeze_4_DequantizeLinear_Output_const_array.data_start = AI_PTR(net_ctx->_weights[0] + 99124);
  mul_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(352, 2, { transpose_6_output.data->data,unsqueeze_4_DequantizeLinear_Output_const.data->data});
  forward_eltwise_integer_INT8(&mul_16_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(352, 1, { mul_16_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_12(_stai_network_context* net_ctx)
{
  mul_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  mul_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  add_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(393, 2, { mul_16_output.data->data,mul_17_output.data->data});
  forward_eltwise_integer_INT8(&add_12_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(393, 1, { add_12_output.data->data});
}
void forward_lite_tile_expand_5(_stai_network_context* net_ctx)
{
  add_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  add_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  expand_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  expand_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(404, 1, { add_12_output0.data->data});
  forward_tile(&expand_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(404, 1, { expand_5_output.data->data});
}
void forward_lite_transpose_val_302(_stai_network_context* net_ctx)
{
  expand_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  expand_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  val_302_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_302_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(406, 1, { expand_5_output0.data->data});
  forward_transpose(&val_302_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(406, 1, { val_302_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_308(_stai_network_context* net_ctx)
{
  val_302_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_302_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  val_172_DequantizeLinear_Output_const_4D_array.data = AI_PTR(net_ctx->_weights[0] + 48);
  val_172_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 48);
  val_308_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  val_308_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(410, 2, { val_302_output.data->data,val_172_DequantizeLinear_Output_const_4D.data->data});
  forward_eltwise_integer_INT8(&val_308_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(410, 1, { val_308_output.data->data});
}
void forward_lite_matmul_val_312(_stai_network_context* net_ctx)
{
  val_306_0_0_val_312_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41232);
  val_306_0_0_val_312_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41232);
  val_308_0_1_val_312_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 74000);
  val_308_0_1_val_312_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 74000);
  val_312_bias_0_2_val_312_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 41228);
  val_312_bias_0_2_val_312_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 41228);
  val_312_output_array.data = AI_PTR(net_ctx->_activations[0] + 106768);
  val_312_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 106768);
  _STAI_NETWORK_EVENT_NODE_START_CB(413, 3, { val_306_0_0_val_312_conversion_output.data->data,val_308_0_1_val_312_conversion_output.data->data,val_312_bias_0_2_val_312_conversion_output.data->data});
  forward_matmul(&val_312_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(413, 1, { val_312_output.data->data});
}
void forward_lite_eltwise_integer_INT8_val_313(_stai_network_context* net_ctx)
{
  val_312_0_0_val_313_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_312_0_0_val_313_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  val_178_output_array.data = AI_PTR(net_ctx->_activations[0] + 1024);
  val_178_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1024);
  val_313_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  val_313_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(416, 2, { val_312_0_0_val_313_conversion_output.data->data,val_178_output.data->data});
  forward_eltwise_integer_INT8(&val_313_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(416, 1, { val_313_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_val_219(_stai_network_context* net_ctx)
{
  mul_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  mul_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  val_219_weights_array.data = AI_PTR(net_ctx->_weights[0] + 795196);
  val_219_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 795196);
  val_211_bias_array.data = AI_PTR(net_ctx->_weights[0] + 794684);
  val_211_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 794684);
  val_219_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  val_219_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_219_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_219_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(315, 1, { mul_13_output.data->data});
  forward_conv2d_integer_SSSA(&val_219_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(315, 1, { val_219_output.data->data});
}
void forward_lite_eltwise_integer_INT8_linear_9(_stai_network_context* net_ctx)
{
  val_219_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  val_219_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 820);
  mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 820);
  linear_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  linear_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(324, 2, { val_219_output.data->data,mdl_model_layers_1_self_attn_v_proj_bias_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&linear_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(324, 1, { linear_9_output.data->data});
}
void forward_lite_transpose_transpose_7(_stai_network_context* net_ctx)
{
  linear_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  linear_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  transpose_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(342, 1, { linear_9_output0.data->data});
  forward_transpose(&transpose_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(342, 1, { transpose_7_output.data->data});
}
void forward_lite_tile_expand_6(_stai_network_context* net_ctx)
{
  transpose_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  expand_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 14336);
  expand_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14336);
  _STAI_NETWORK_EVENT_NODE_START_CB(372, 1, { transpose_7_output0.data->data});
  forward_tile(&expand_6_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(372, 1, { expand_6_output.data->data});
}
void forward_lite_matmul_scaled_dot_product_attention_1(_stai_network_context* net_ctx)
{
  val_314_output_array.data = AI_PTR(net_ctx->_activations[0] + 57356);
  val_314_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57356);
  expand_6_0_1_scaled_dot_product_attention_1_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 73740);
  expand_6_0_1_scaled_dot_product_attention_1_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 73740);
  scaled_dot_product_attention_1_bias_0_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 40968);
  scaled_dot_product_attention_1_bias_0_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40968);
  scaled_dot_product_attention_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 106508);
  scaled_dot_product_attention_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 106508);
  _STAI_NETWORK_EVENT_NODE_START_CB(425, 3, { val_314_output.data->data,expand_6_0_1_scaled_dot_product_attention_1_conversion_output0.data->data,scaled_dot_product_attention_1_bias_0_conversion_output.data->data});
  forward_matmul(&scaled_dot_product_attention_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(425, 1, { scaled_dot_product_attention_1_output.data->data});
}
void forward_lite_transpose_transpose_8(_stai_network_context* net_ctx)
{
  scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  scaled_dot_product_attention_1_0_0_transpose_8_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  transpose_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  transpose_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  _STAI_NETWORK_EVENT_NODE_START_CB(428, 1, { scaled_dot_product_attention_1_0_0_transpose_8_conversion_output.data->data});
  forward_transpose(&transpose_8_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(428, 1, { transpose_8_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_10(_stai_network_context* net_ctx)
{
  transpose_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  transpose_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  linear_10_weights_array.data = AI_PTR(net_ctx->_weights[0] + 827964);
  linear_10_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 827964);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  linear_10_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_10_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  linear_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  _STAI_NETWORK_EVENT_NODE_START_CB(434, 1, { transpose_8_output0.data->data});
  forward_conv2d_integer_SSSA(&linear_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(434, 1, { linear_10_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_13(_stai_network_context* net_ctx)
{
  add_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  add_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  linear_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 10240);
  linear_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10240);
  add_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  add_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  _STAI_NETWORK_EVENT_NODE_START_CB(437, 2, { add_9_output.data->data,linear_10_output.data->data});
  forward_eltwise_integer_INT8(&add_13_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(437, 1, { add_13_output.data->data});
}
void forward_lite_eltwise_pow_4(_stai_network_context* net_ctx)
{
  add_13_0_0_pow_4_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  add_13_0_0_pow_4_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  val_60_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  val_60_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  pow_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 59392);
  pow_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 59392);
  _STAI_NETWORK_EVENT_NODE_START_CB(440, 2, { add_13_0_0_pow_4_conversion_output.data->data,val_60_3D.data->data});
  forward_eltwise(&pow_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(440, 1, { pow_4_output.data->data});
}
void forward_lite_reduce_mean_3(_stai_network_context* net_ctx)
{
  pow_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 59392);
  pow_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 59392);
  mean_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(441, 1, { pow_4_output.data->data});
  forward_reduce(&mean_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(441, 1, { mean_3_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_14(_stai_network_context* net_ctx)
{
  mean_3_Mul_0_0_add_14_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mean_3_Mul_0_0_add_14_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_63_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 44);
  val_63_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 44);
  add_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 32);
  add_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32);
  _STAI_NETWORK_EVENT_NODE_START_CB(444, 2, { mean_3_Mul_0_0_add_14_conversion_output.data->data,val_63_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&add_14_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(444, 1, { add_14_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_18(_stai_network_context* net_ctx)
{
  add_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  add_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  rsqrt_3_0_1_mul_18_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  rsqrt_3_0_1_mul_18_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  mul_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 160);
  mul_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 160);
  _STAI_NETWORK_EVENT_NODE_START_CB(451, 2, { add_13_output.data->data,rsqrt_3_0_1_mul_18_conversion_output.data->data});
  forward_eltwise_integer_INT8(&mul_18_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(451, 1, { mul_18_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_19(_stai_network_context* net_ctx)
{
  mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 308);
  mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 308);
  mul_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 160);
  mul_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 160);
  mul_19_output_array.data = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_19_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8352);
  _STAI_NETWORK_EVENT_NODE_START_CB(454, 2, { mdl_model_layers_1_post_attention_layernorm_weight_DequantizeLinear_Output_const_3D.data->data,mul_18_output.data->data});
  forward_eltwise_integer_INT8(&mul_19_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(454, 1, { mul_19_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_11(_stai_network_context* net_ctx)
{
  mul_19_output_array.data = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_19_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8352);
  linear_11_weights_array.data = AI_PTR(net_ctx->_weights[0] + 893500);
  linear_11_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 893500);
  linear_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_11_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_11_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(457, 1, { mul_19_output.data->data});
  forward_conv2d_integer_SSSA(&linear_11_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(457, 1, { linear_11_output.data->data});
}
void forward_lite_nl_integer_val_328(_stai_network_context* net_ctx)
{
  linear_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  val_328_output_array.data = AI_PTR(net_ctx->_activations[0] + 43008);
  val_328_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43008);
  _STAI_NETWORK_EVENT_NODE_START_CB(463, 1, { linear_11_output.data->data});
  forward_nl_integer(&val_328_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(463, 1, { val_328_output.data->data});
}
void forward_lite_eltwise_integer_INT8_silu_1(_stai_network_context* net_ctx)
{
  linear_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  val_328_output_array.data = AI_PTR(net_ctx->_activations[0] + 43008);
  val_328_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43008);
  silu_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 59392);
  silu_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 59392);
  _STAI_NETWORK_EVENT_NODE_START_CB(466, 2, { linear_11_output.data->data,val_328_output.data->data});
  forward_eltwise_integer_INT8(&silu_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(466, 1, { silu_1_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_12(_stai_network_context* net_ctx)
{
  mul_19_output_array.data = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_19_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8352);
  linear_12_weights_array.data = AI_PTR(net_ctx->_weights[0] + 1024572);
  linear_12_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 1024572);
  linear_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 432188);
  linear_12_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  linear_12_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(458, 1, { mul_19_output.data->data});
  forward_conv2d_integer_SSSA(&linear_12_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(458, 1, { linear_12_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_20(_stai_network_context* net_ctx)
{
  silu_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 59392);
  silu_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 59392);
  linear_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  mul_20_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_20_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(469, 2, { silu_1_output.data->data,linear_12_output.data->data});
  forward_eltwise_integer_INT8(&mul_20_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(469, 1, { mul_20_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_linear_13(_stai_network_context* net_ctx)
{
  mul_20_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_20_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  linear_13_weights_array.data = AI_PTR(net_ctx->_weights[0] + 1155644);
  linear_13_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 1155644);
  val_66_bias_array.data = AI_PTR(net_ctx->_weights[0] + 234300);
  val_66_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 234300);
  linear_13_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 16384);
  linear_13_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 16384);
  linear_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  _STAI_NETWORK_EVENT_NODE_START_CB(472, 1, { mul_20_output.data->data});
  forward_conv2d_integer_SSSA(&linear_13_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(472, 1, { linear_13_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_15(_stai_network_context* net_ctx)
{
  add_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 18432);
  add_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 18432);
  linear_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 26624);
  linear_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26624);
  add_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(475, 2, { add_13_output.data->data,linear_13_output.data->data});
  forward_eltwise_integer_INT8(&add_15_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(475, 1, { add_15_output.data->data});
}
void forward_lite_eltwise_pow_5(_stai_network_context* net_ctx)
{
  add_15_0_0_pow_5_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  add_15_0_0_pow_5_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  val_60_3D_array.data = AI_PTR(net_ctx->_weights[0] + 32);
  val_60_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 32);
  pow_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 40960);
  pow_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40960);
  _STAI_NETWORK_EVENT_NODE_START_CB(478, 2, { add_15_0_0_pow_5_conversion_output.data->data,val_60_3D.data->data});
  forward_eltwise(&pow_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(478, 1, { pow_5_output.data->data});
}
void forward_lite_reduce_mean_4(_stai_network_context* net_ctx)
{
  pow_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 40960);
  pow_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 40960);
  mean_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mean_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(479, 1, { pow_5_output.data->data});
  forward_reduce(&mean_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(479, 1, { mean_4_output.data->data});
}
void forward_lite_eltwise_integer_INT8_add_16(_stai_network_context* net_ctx)
{
  mean_4_Mul_0_0_add_16_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  mean_4_Mul_0_0_add_16_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  val_63_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 44);
  val_63_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 44);
  add_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 8224);
  add_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8224);
  _STAI_NETWORK_EVENT_NODE_START_CB(482, 2, { mean_4_Mul_0_0_add_16_conversion_output.data->data,val_63_DequantizeLinear_Output_const_3D.data->data});
  forward_eltwise_integer_INT8(&add_16_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(482, 1, { add_16_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_21(_stai_network_context* net_ctx)
{
  add_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  rsqrt_4_0_1_mul_21_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 8320);
  rsqrt_4_0_1_mul_21_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8320);
  mul_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8352);
  _STAI_NETWORK_EVENT_NODE_START_CB(489, 2, { add_15_output.data->data,rsqrt_4_0_1_mul_21_conversion_output.data->data});
  forward_eltwise_integer_INT8(&mul_21_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(489, 1, { mul_21_output.data->data});
}
void forward_lite_eltwise_integer_INT8_mul_22(_stai_network_context* net_ctx)
{
  mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array.data = AI_PTR(net_ctx->_weights[0] + 52);
  mdl_model_norm_weight_DequantizeLinear_Output_const_3D_array.data_start = AI_PTR(net_ctx->_weights[0] + 52);
  mul_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8352);
  mul_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(492, 2, { mdl_model_norm_weight_DequantizeLinear_Output_const_3D.data->data,mul_21_output.data->data});
  forward_eltwise_integer_INT8(&mul_22_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(492, 1, { mul_22_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_logits_QuantizeLinear_Input(_stai_network_context* net_ctx)
{
  mul_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  mul_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  logits_QuantizeLinear_Input_weights_array.data = AI_PTR(net_ctx->_weights[0] + 1286716);
  logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 1286716);
  logits_QuantizeLinear_Input_bias_array.data = AI_PTR(net_ctx->_weights[0] + 1382460);
  logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 1382460);
  logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 8192);
  logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 8192);
  logits_QuantizeLinear_Input_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(495, 1, { mul_22_output.data->data});
  forward_conv2d_integer_SSSA(&logits_QuantizeLinear_Input_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(495, 1, { logits_QuantizeLinear_Input_output.data->data});
}

/*****************************************************************************/


static const ai_u32 scaled_dot_product_attention_1_bias_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float scaled_dot_product_attention_1_bias_0_conversion_t_in_0_fmt_scale_const_f32 = 0.012092793360352516f;
static const ai_i8 scaled_dot_product_attention_1_bias_0_conversion_t_in_0_fmt_zero_const_s8 = -9;

static const ai_u32 val_312_bias_0_2_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float val_312_bias_0_2_val_312_conversion_t_in_0_fmt_scale_const_f32 = 0.03215616196393967f;
static const ai_i8 val_312_bias_0_2_val_312_conversion_t_in_0_fmt_zero_const_s8 = -45;

static const ai_u32 scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_fmt_scale_const_f32 = 0.012832569889724255f;
static const ai_i8 scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_fmt_zero_const_s8 = 10;

static const ai_u32 val_179_bias_0_2_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float val_179_bias_0_2_val_179_conversion_t_in_0_fmt_scale_const_f32 = 0.03992549702525139f;
static const ai_i8 val_179_bias_0_2_val_179_conversion_t_in_0_fmt_zero_const_s8 = -9;

static const ai_u32 val_74_bias_0_2_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float val_74_bias_0_2_val_74_conversion_t_in_0_fmt_scale_const_f32 = 0.030147666111588478f;
static const ai_i8 val_74_bias_0_2_val_74_conversion_t_in_0_fmt_zero_const_s8 = 40;

static const ai_u32 val_82_bias_0_2_val_82_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1;
static const ai_float val_82_bias_0_2_val_82_conversion_t_in_0_fmt_scale_const_f32 = 0.015435585752129555f;
static const ai_i8 val_82_bias_0_2_val_82_conversion_t_in_0_fmt_zero_const_s8 = -6;

static const ai_u32 val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32768;
static const ai_float val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_fmt_scale_const_f32 = 0.0007635335205122828f;
static const ai_i8 val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_fmt_zero_const_s8 = -4;

static const ai_u32 val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32768;
static const ai_float val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_fmt_scale_const_f32 = 0.0006966242217458785f;
static const ai_i8 val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_fmt_zero_const_s8 = -2;





static const ai_u32 embedding_0_0_pow_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float embedding_0_0_pow_1_conversion_t_in_0_fmt_scale_const_f32 = 0.0007210441981442273f;
static const ai_i8 embedding_0_0_pow_1_conversion_t_in_0_fmt_zero_const_s8 = 2;




static const ai_u32 mean_Mul_0_0_add_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float mean_Mul_0_0_add_4_conversion_t_out_0_fmt_scale_const_f32 = 3.6172205000184476e-06f;
static const ai_i8 mean_Mul_0_0_add_4_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 add_4_0_0_val_64_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float add_4_0_0_val_64_conversion_t_in_0_fmt_scale_const_f32 = 3.6211422411724925e-06f;
static const ai_i8 add_4_0_0_val_64_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_i32 val_64_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_i32 rsqrt_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_u32 rsqrt_0_1_mul_3_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float rsqrt_0_1_mul_3_conversion_t_out_0_fmt_scale_const_f32 = 0.22850461304187775f;
static const ai_i8 rsqrt_0_1_mul_3_conversion_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u32 mul_4_0_0_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float mul_4_0_0_val_74_conversion_t_in_0_fmt_scale_const_f32 = 0.03342914581298828f;
static const ai_i8 mul_4_0_0_val_74_conversion_t_in_0_fmt_zero_const_s8 = -3;


static const ai_u32 val_82_0_0_linear_2_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_82_0_0_linear_2_conversion_t_out_0_fmt_scale_const_f32 = 0.015435585752129555f;
static const ai_i8 val_82_0_0_linear_2_conversion_t_out_0_fmt_zero_const_s8 = -6;




static const ai_u32 expand_4_0_1_scaled_dot_product_attention_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_fmt_scale_const_f32 = 0.015425463207066059f;
static const ai_i8 expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_fmt_zero_const_s8 = -6;


static const ai_u32 val_74_0_0_linear_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_74_0_0_linear_1_conversion_t_out_0_fmt_scale_const_f32 = 0.030147666111588478f;
static const ai_i8 val_74_0_0_linear_1_conversion_t_out_0_fmt_zero_const_s8 = 40;













static const ai_u32 val_175_0_1_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float val_175_0_1_val_179_conversion_t_in_0_fmt_scale_const_f32 = 0.012009273283183575f;
static const ai_i8 val_175_0_1_val_179_conversion_t_in_0_fmt_zero_const_s8 = -22;












static const ai_u32 val_173_0_0_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float val_173_0_0_val_179_conversion_t_in_0_fmt_scale_const_f32 = 0.011903457343578339f;
static const ai_i8 val_173_0_0_val_179_conversion_t_in_0_fmt_zero_const_s8 = -8;


static const ai_u32 val_179_0_0_val_180_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_179_0_0_val_180_conversion_t_out_0_fmt_scale_const_f32 = 0.03992549702525139f;
static const ai_i8 val_179_0_0_val_180_conversion_t_out_0_fmt_zero_const_s8 = -9;


static const ai_u32 val_35_0_1_bitwise_and_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float val_35_0_1_bitwise_and_1_conversion_t_out_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_i8 val_35_0_1_bitwise_and_1_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 bitwise_and_1_0_1___bool_fix_inv_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1024;
static const ai_float bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_i8 bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_fmt_zero_const_s8 = -128;


static const ai_u32 __bool_fix_inv_0_0_val_178_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 1024;
static const ai_float __bool_fix_inv_0_0_val_178_conversion_t_out_0_fmt_scale_const_f32 = 0.003921568859368563f;
static const ai_i8 __bool_fix_inv_0_0_val_178_conversion_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u32 val_180_0_0_val_181_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_180_0_0_val_181_conversion_t_in_0_fmt_scale_const_f32 = 1.3344405750530544e+36f;
static const ai_i8 val_180_0_0_val_181_conversion_t_in_0_fmt_zero_const_s8 = 127;

static const ai_i32 val_181_t_in_0_shape_ch_h_w_prod_const_s32 = 4096;


static const ai_u32 scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_fmt_scale_const_f32 = 0.012832569889724255f;
static const ai_i8 scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_fmt_zero_const_s8 = 10;




static const ai_u32 add_7_0_0_pow_2_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float add_7_0_0_pow_2_conversion_t_in_0_fmt_scale_const_f32 = 0.005136104766279459f;
static const ai_i8 add_7_0_0_pow_2_conversion_t_in_0_fmt_zero_const_s8 = 9;




static const ai_u32 mean_1_Mul_0_0_add_8_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float mean_1_Mul_0_0_add_8_conversion_t_out_0_fmt_scale_const_f32 = 0.0001574616035213694f;
static const ai_i8 mean_1_Mul_0_0_add_8_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 add_8_0_0_val_193_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float add_8_0_0_val_193_conversion_t_in_0_fmt_scale_const_f32 = 0.00015746551798656583f;
static const ai_i8 add_8_0_0_val_193_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_i32 val_193_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_i32 rsqrt_1_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_u32 rsqrt_1_0_1_mul_9_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float rsqrt_1_0_1_mul_9_conversion_t_out_0_fmt_scale_const_f32 = 0.14992481470108032f;
static const ai_i8 rsqrt_1_0_1_mul_9_conversion_t_out_0_fmt_zero_const_s8 = -128;










static const ai_u32 add_9_0_0_pow_3_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float add_9_0_0_pow_3_conversion_t_in_0_fmt_scale_const_f32 = 0.00763314263895154f;
static const ai_i8 add_9_0_0_pow_3_conversion_t_in_0_fmt_zero_const_s8 = 8;




static const ai_u32 mean_2_Mul_0_0_add_10_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float mean_2_Mul_0_0_add_10_conversion_t_out_0_fmt_scale_const_f32 = 0.0005274713039398193f;
static const ai_i8 mean_2_Mul_0_0_add_10_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 add_10_0_0_val_201_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float add_10_0_0_val_201_conversion_t_in_0_fmt_scale_const_f32 = 0.0005274752038531005f;
static const ai_i8 add_10_0_0_val_201_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_i32 val_201_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_i32 rsqrt_2_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_u32 rsqrt_2_0_1_mul_12_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float rsqrt_2_0_1_mul_12_conversion_t_out_0_fmt_scale_const_f32 = 0.09743647277355194f;
static const ai_i8 rsqrt_2_0_1_mul_12_conversion_t_out_0_fmt_zero_const_s8 = -128;














static const ai_u32 val_306_0_0_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float val_306_0_0_val_312_conversion_t_in_0_fmt_scale_const_f32 = 0.009856867603957653f;
static const ai_i8 val_306_0_0_val_312_conversion_t_in_0_fmt_zero_const_s8 = -1;














static const ai_u32 val_308_0_1_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float val_308_0_1_val_312_conversion_t_in_0_fmt_scale_const_f32 = 0.011728443205356598f;
static const ai_i8 val_308_0_1_val_312_conversion_t_in_0_fmt_zero_const_s8 = 7;


static const ai_u32 val_312_0_0_val_313_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_312_0_0_val_313_conversion_t_out_0_fmt_scale_const_f32 = 0.03215616196393967f;
static const ai_i8 val_312_0_0_val_313_conversion_t_out_0_fmt_zero_const_s8 = -45;


static const ai_u32 val_313_0_0_val_314_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 4096;
static const ai_float val_313_0_0_val_314_conversion_t_in_0_fmt_scale_const_f32 = 1.3344405750530544e+36f;
static const ai_i8 val_313_0_0_val_314_conversion_t_in_0_fmt_zero_const_s8 = 127;

static const ai_i32 val_314_t_in_0_shape_ch_h_w_prod_const_s32 = 4096;





static const ai_u32 expand_6_0_1_scaled_dot_product_attention_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_fmt_scale_const_f32 = 0.013600028119981289f;
static const ai_i8 expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_fmt_zero_const_s8 = 6;


static const ai_u32 scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_fmt_scale_const_f32 = 0.012092793360352516f;
static const ai_i8 scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_fmt_zero_const_s8 = -9;




static const ai_u32 add_13_0_0_pow_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float add_13_0_0_pow_4_conversion_t_in_0_fmt_scale_const_f32 = 0.010768704116344452f;
static const ai_i8 add_13_0_0_pow_4_conversion_t_in_0_fmt_zero_const_s8 = 7;




static const ai_u32 mean_3_Mul_0_0_add_14_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float mean_3_Mul_0_0_add_14_conversion_t_out_0_fmt_scale_const_f32 = 0.0010057041654363275f;
static const ai_i8 mean_3_Mul_0_0_add_14_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 add_14_0_0_val_326_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float add_14_0_0_val_326_conversion_t_in_0_fmt_scale_const_f32 = 0.0010057081235572696f;
static const ai_i8 add_14_0_0_val_326_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_i32 val_326_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_i32 rsqrt_3_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_u32 rsqrt_3_0_1_mul_18_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float rsqrt_3_0_1_mul_18_conversion_t_out_0_fmt_scale_const_f32 = 0.050555404275655746f;
static const ai_i8 rsqrt_3_0_1_mul_18_conversion_t_out_0_fmt_zero_const_s8 = -128;










static const ai_u32 add_15_0_0_pow_5_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 8192;
static const ai_float add_15_0_0_pow_5_conversion_t_in_0_fmt_scale_const_f32 = 0.02223907597362995f;
static const ai_i8 add_15_0_0_pow_5_conversion_t_in_0_fmt_zero_const_s8 = -7;




static const ai_u32 mean_4_Mul_0_0_add_16_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float mean_4_Mul_0_0_add_16_conversion_t_out_0_fmt_scale_const_f32 = 0.006991112604737282f;
static const ai_i8 mean_4_Mul_0_0_add_16_conversion_t_out_0_fmt_zero_const_s8 = -128;


static const ai_u32 add_16_0_0_val_334_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float add_16_0_0_val_334_conversion_t_in_0_fmt_scale_const_f32 = 0.00699111633002758f;
static const ai_i8 add_16_0_0_val_334_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_i32 val_334_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_i32 rsqrt_4_t_in_0_shape_ch_h_prod_const_s32 = 32;

static const ai_u32 rsqrt_4_0_1_mul_21_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 32;
static const ai_float rsqrt_4_0_1_mul_21_conversion_t_out_0_fmt_scale_const_f32 = 0.038929760456085205f;
static const ai_i8 rsqrt_4_0_1_mul_21_conversion_t_out_0_fmt_zero_const_s8 = -128;



STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention_1_bias_0_conversion */
  {
      const ai_i8* scaled_dot_product_attention_1_bias_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    ai_float* scaled_dot_product_attention_1_bias_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 40968);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(425, 1, {(stai_ptr) scaled_dot_product_attention_1_bias_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(scaled_dot_product_attention_1_bias_0_conversion_t_in_0_ptr_const_s8, scaled_dot_product_attention_1_bias_0_conversion_t_out_0_ptr_f32, scaled_dot_product_attention_1_bias_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, scaled_dot_product_attention_1_bias_0_conversion_t_in_0_fmt_scale_const_f32, scaled_dot_product_attention_1_bias_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(425, 1, {(stai_ptr) scaled_dot_product_attention_1_bias_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention_1_bias_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_312_bias_0_2_val_312_conversion */
  {
      const ai_i8* val_312_bias_0_2_val_312_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4);
    ai_float* val_312_bias_0_2_val_312_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41228);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(413, 1, {(stai_ptr) val_312_bias_0_2_val_312_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_312_bias_0_2_val_312_conversion_t_in_0_ptr_const_s8, val_312_bias_0_2_val_312_conversion_t_out_0_ptr_f32, val_312_bias_0_2_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_312_bias_0_2_val_312_conversion_t_in_0_fmt_scale_const_f32, val_312_bias_0_2_val_312_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(413, 1, {(stai_ptr) val_312_bias_0_2_val_312_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_312_bias_0_2_val_312_conversion */
  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion */
  {
      const ai_i8* scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 8);
    ai_float* scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 40964);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(243, 1, {(stai_ptr) scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8, scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_out_0_ptr_f32, scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_fmt_scale_const_f32, scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(243, 1, {(stai_ptr) scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention_bias_0_2_scaled_dot_product_attention_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_179_bias_0_2_val_179_conversion */
  {
      const ai_i8* val_179_bias_0_2_val_179_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 12);
    ai_float* val_179_bias_0_2_val_179_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(231, 1, {(stai_ptr) val_179_bias_0_2_val_179_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_179_bias_0_2_val_179_conversion_t_in_0_ptr_const_s8, val_179_bias_0_2_val_179_conversion_t_out_0_ptr_f32, val_179_bias_0_2_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_179_bias_0_2_val_179_conversion_t_in_0_fmt_scale_const_f32, val_179_bias_0_2_val_179_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(231, 1, {(stai_ptr) val_179_bias_0_2_val_179_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_179_bias_0_2_val_179_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_74_bias_0_2_val_74_conversion */
  {
      const ai_i8* val_74_bias_0_2_val_74_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16);
    ai_float* val_74_bias_0_2_val_74_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 40960);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(132, 1, {(stai_ptr) val_74_bias_0_2_val_74_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_74_bias_0_2_val_74_conversion_t_in_0_ptr_const_s8, val_74_bias_0_2_val_74_conversion_t_out_0_ptr_f32, val_74_bias_0_2_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_74_bias_0_2_val_74_conversion_t_in_0_fmt_scale_const_f32, val_74_bias_0_2_val_74_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(132, 1, {(stai_ptr) val_74_bias_0_2_val_74_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_74_bias_0_2_val_74_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_82_bias_0_2_val_82_conversion */
  {
      const ai_i8* val_82_bias_0_2_val_82_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 20);
    ai_float* val_82_bias_0_2_val_82_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41236);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(133, 1, {(stai_ptr) val_82_bias_0_2_val_82_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_82_bias_0_2_val_82_conversion_t_in_0_ptr_const_s8, val_82_bias_0_2_val_82_conversion_t_out_0_ptr_f32, val_82_bias_0_2_val_82_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_82_bias_0_2_val_82_conversion_t_in_0_fmt_scale_const_f32, val_82_bias_0_2_val_82_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(133, 1, {(stai_ptr) val_82_bias_0_2_val_82_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_82_bias_0_2_val_82_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_73_DequantizeLinear_Output_const_0_1_val_74_conversion */
  {
      const ai_i8* val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 103220);
    ai_float* val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41240);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(94, 1, {(stai_ptr) val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_ptr_const_s8, val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_out_0_ptr_f32, val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_fmt_scale_const_f32, val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(94, 1, {(stai_ptr) val_73_DequantizeLinear_Output_const_0_1_val_74_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_73_DequantizeLinear_Output_const_0_1_val_74_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_81_DequantizeLinear_Output_const_0_1_val_82_conversion */
  {
      const ai_i8* val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 135988);
    ai_float* val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 172312);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(95, 1, {(stai_ptr) val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_ptr_const_s8, val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_out_0_ptr_f32, val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_fmt_scale_const_f32, val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(95, 1, {(stai_ptr) val_81_DequantizeLinear_Output_const_0_1_val_82_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_81_DequantizeLinear_Output_const_0_1_val_82_conversion */
  /* LITE_KERNEL_SECTION BEGIN _to_copy */
  {
    
  forward_lite_cast__to_copy(net_ctx);
  }
  /* LITE_KERNEL_SECTION END _to_copy */
  /* LITE_KERNEL_SECTION BEGIN __ids_floored */
  {
    
  forward_lite_eltwise___ids_floored(net_ctx);
  }
  /* LITE_KERNEL_SECTION END __ids_floored */
  /* LITE_KERNEL_SECTION BEGIN input_ids_clipped */
  {
    
  forward_lite_eltwise_input_ids_clipped(net_ctx);
  }
  /* LITE_KERNEL_SECTION END input_ids_clipped */
  /* LITE_KERNEL_SECTION BEGIN embedding */
  {
    
  forward_lite_gather_embedding(net_ctx);
  }
  /* LITE_KERNEL_SECTION END embedding */
  /* LITE_KERNEL_SECTION BEGIN embedding_0_0_pow_1_conversion */
  {
      const ai_i8* embedding_0_0_pow_1_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 303384);
    ai_float* embedding_0_0_pow_1_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(97, 1, {(stai_ptr) embedding_0_0_pow_1_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(embedding_0_0_pow_1_conversion_t_in_0_ptr_const_s8, embedding_0_0_pow_1_conversion_t_out_0_ptr_f32, embedding_0_0_pow_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, embedding_0_0_pow_1_conversion_t_in_0_fmt_scale_const_f32, embedding_0_0_pow_1_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(97, 1, {(stai_ptr) embedding_0_0_pow_1_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END embedding_0_0_pow_1_conversion */
  /* LITE_KERNEL_SECTION BEGIN pow_1 */
  {
    
  forward_lite_eltwise_pow_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pow_1 */
  /* LITE_KERNEL_SECTION BEGIN mean */
  {
    
  forward_lite_reduce_mean(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mean */
  /* LITE_KERNEL_SECTION BEGIN mean_Mul */
  {
      ai_float* mean_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    const ai_float* mean_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 40972);
    const ai_float* mean_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168756);
    const ai_float* mean_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(105, 1, {(stai_ptr) mean_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(mean_Mul_t_out_0_ptr_f32, mean_Mul_t_in_0_ptr_const_f32, mean_Mul_t_weight_0_ptr_const_f32, mean_Mul_t_weight_1_ptr_const_f32, (ai_u32)(32), (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(105, 1, {(stai_ptr) mean_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mean_Mul */
  /* LITE_KERNEL_SECTION BEGIN mean_Mul_0_0_add_4_conversion */
  {
      const ai_float* mean_Mul_0_0_add_4_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    ai_i8* mean_Mul_0_0_add_4_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8320);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(105, 1, {(stai_ptr) mean_Mul_0_0_add_4_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(mean_Mul_0_0_add_4_conversion_t_in_0_ptr_const_f32, mean_Mul_0_0_add_4_conversion_t_out_0_ptr_s8, mean_Mul_0_0_add_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mean_Mul_0_0_add_4_conversion_t_out_0_fmt_scale_const_f32, mean_Mul_0_0_add_4_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(105, 1, {(stai_ptr) mean_Mul_0_0_add_4_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END mean_Mul_0_0_add_4_conversion */
  /* LITE_KERNEL_SECTION BEGIN add_4 */
  {
    
  forward_lite_eltwise_integer_INT8_add_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_4 */
  /* LITE_KERNEL_SECTION BEGIN add_4_0_0_val_64_conversion */
  {
      const ai_i8* add_4_0_0_val_64_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 40972);
    ai_float* add_4_0_0_val_64_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(111, 1, {(stai_ptr) add_4_0_0_val_64_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_4_0_0_val_64_conversion_t_in_0_ptr_const_s8, add_4_0_0_val_64_conversion_t_out_0_ptr_f32, add_4_0_0_val_64_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_4_0_0_val_64_conversion_t_in_0_fmt_scale_const_f32, add_4_0_0_val_64_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(111, 1, {(stai_ptr) add_4_0_0_val_64_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_4_0_0_val_64_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_64 */
  {
      ai_handle val_64_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 8320);
    const ai_handle val_64_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(117, 1, {(stai_ptr) val_64_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sqrt_if32of32(val_64_t_out_0_ptr_handle, val_64_t_in_0_ptr_const_handle, val_64_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(117, 1, {(stai_ptr) val_64_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_64 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt */
  {
      ai_handle rsqrt_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 40972);
    const ai_handle rsqrt_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8320);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(119, 1, {(stai_ptr) rsqrt_t_in_0_ptr_const_handle});
    
  forward_lite_nl_reciprocal_if32of32(rsqrt_t_out_0_ptr_handle, rsqrt_t_in_0_ptr_const_handle, rsqrt_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(119, 1, {(stai_ptr) rsqrt_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END rsqrt */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_0_1_mul_3_conversion */
  {
      const ai_float* rsqrt_0_1_mul_3_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 40972);
    ai_i8* rsqrt_0_1_mul_3_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 40928);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(119, 1, {(stai_ptr) rsqrt_0_1_mul_3_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(rsqrt_0_1_mul_3_conversion_t_in_0_ptr_const_f32, rsqrt_0_1_mul_3_conversion_t_out_0_ptr_s8, rsqrt_0_1_mul_3_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, rsqrt_0_1_mul_3_conversion_t_out_0_fmt_scale_const_f32, rsqrt_0_1_mul_3_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(119, 1, {(stai_ptr) rsqrt_0_1_mul_3_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END rsqrt_0_1_mul_3_conversion */
  /* LITE_KERNEL_SECTION BEGIN mul_3 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_3 */
  /* LITE_KERNEL_SECTION BEGIN mul_4 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_4 */
  /* LITE_KERNEL_SECTION BEGIN mul_4_0_0_val_74_conversion */
  {
      const ai_i8* mul_4_0_0_val_74_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 32768);
    ai_float* mul_4_0_0_val_74_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(128, 1, {(stai_ptr) mul_4_0_0_val_74_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(mul_4_0_0_val_74_conversion_t_in_0_ptr_const_s8, mul_4_0_0_val_74_conversion_t_out_0_ptr_f32, mul_4_0_0_val_74_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mul_4_0_0_val_74_conversion_t_in_0_fmt_scale_const_f32, mul_4_0_0_val_74_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(128, 1, {(stai_ptr) mul_4_0_0_val_74_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mul_4_0_0_val_74_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_82 */
  {
    
  forward_lite_matmul_val_82(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_82 */
  /* LITE_KERNEL_SECTION BEGIN val_82_0_0_linear_2_conversion */
  {
      const ai_float* val_82_0_0_linear_2_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 311576);
    ai_i8* val_82_0_0_linear_2_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 172312);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(133, 1, {(stai_ptr) val_82_0_0_linear_2_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(val_82_0_0_linear_2_conversion_t_in_0_ptr_const_f32, val_82_0_0_linear_2_conversion_t_out_0_ptr_s8, val_82_0_0_linear_2_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_82_0_0_linear_2_conversion_t_out_0_fmt_scale_const_f32, val_82_0_0_linear_2_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(133, 1, {(stai_ptr) val_82_0_0_linear_2_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END val_82_0_0_linear_2_conversion */
  /* LITE_KERNEL_SECTION BEGIN linear_2 */
  {
    
  forward_lite_eltwise_integer_INT8_linear_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_2 */
  /* LITE_KERNEL_SECTION BEGIN transpose_3 */
  {
    
  forward_lite_transpose_transpose_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_3 */
  /* LITE_KERNEL_SECTION BEGIN expand_4 */
  {
    
  forward_lite_tile_expand_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END expand_4 */
  /* LITE_KERNEL_SECTION BEGIN expand_4_0_1_scaled_dot_product_attention_conversion */
  {
      const ai_i8* expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 176408);
    ai_float* expand_4_0_1_scaled_dot_product_attention_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 184600);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(190, 1, {(stai_ptr) expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_ptr_const_s8, expand_4_0_1_scaled_dot_product_attention_conversion_t_out_0_ptr_f32, expand_4_0_1_scaled_dot_product_attention_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_fmt_scale_const_f32, expand_4_0_1_scaled_dot_product_attention_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(190, 1, {(stai_ptr) expand_4_0_1_scaled_dot_product_attention_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END expand_4_0_1_scaled_dot_product_attention_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_74 */
  {
    
  forward_lite_matmul_val_74(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_74 */
  /* LITE_KERNEL_SECTION BEGIN val_74_0_0_linear_1_conversion */
  {
      const ai_float* val_74_0_0_linear_1_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 217368);
    ai_i8* val_74_0_0_linear_1_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(132, 1, {(stai_ptr) val_74_0_0_linear_1_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(val_74_0_0_linear_1_conversion_t_in_0_ptr_const_f32, val_74_0_0_linear_1_conversion_t_out_0_ptr_s8, val_74_0_0_linear_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_74_0_0_linear_1_conversion_t_out_0_fmt_scale_const_f32, val_74_0_0_linear_1_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(132, 1, {(stai_ptr) val_74_0_0_linear_1_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END val_74_0_0_linear_1_conversion */
  /* LITE_KERNEL_SECTION BEGIN linear_1 */
  {
    
  forward_lite_eltwise_integer_INT8_linear_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_1 */
  /* LITE_KERNEL_SECTION BEGIN transpose_2 */
  {
    
  forward_lite_transpose_transpose_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_2 */
  /* LITE_KERNEL_SECTION BEGIN slice_4 */
  {
    
  forward_lite_slice_slice_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_4 */
  /* LITE_KERNEL_SECTION BEGIN neg_1 */
  {
    
  forward_lite_nl_integer_neg_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END neg_1 */
  /* LITE_KERNEL_SECTION BEGIN slice_3 */
  {
    
  forward_lite_slice_slice_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_3 */
  /* LITE_KERNEL_SECTION BEGIN cat_2 */
  {
    
  forward_lite_concat_cat_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cat_2 */
  /* LITE_KERNEL_SECTION BEGIN mul_8 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_8 */
  /* LITE_KERNEL_SECTION BEGIN mul_7 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_7 */
  /* LITE_KERNEL_SECTION BEGIN add_6 */
  {
    
  forward_lite_eltwise_integer_INT8_add_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_6 */
  /* LITE_KERNEL_SECTION BEGIN expand_3 */
  {
    
  forward_lite_tile_expand_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END expand_3 */
  /* LITE_KERNEL_SECTION BEGIN val_169 */
  {
    
  forward_lite_transpose_val_169(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_169 */
  /* LITE_KERNEL_SECTION BEGIN val_175 */
  {
    
  forward_lite_eltwise_integer_INT8_val_175(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_175 */
  /* LITE_KERNEL_SECTION BEGIN val_175_0_1_val_179_conversion */
  {
      const ai_i8* val_175_0_1_val_179_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_float* val_175_0_1_val_179_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41236);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(228, 1, {(stai_ptr) val_175_0_1_val_179_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_175_0_1_val_179_conversion_t_in_0_ptr_const_s8, val_175_0_1_val_179_conversion_t_out_0_ptr_f32, val_175_0_1_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_175_0_1_val_179_conversion_t_in_0_fmt_scale_const_f32, val_175_0_1_val_179_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(228, 1, {(stai_ptr) val_175_0_1_val_179_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_175_0_1_val_179_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_66 */
  {
    
  forward_lite_conv2d_integer_SSSA_val_66(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_66 */
  /* LITE_KERNEL_SECTION BEGIN linear */
  {
    
  forward_lite_eltwise_integer_INT8_linear(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear */
  /* LITE_KERNEL_SECTION BEGIN transpose_1 */
  {
    
  forward_lite_transpose_transpose_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_1 */
  /* LITE_KERNEL_SECTION BEGIN slice_2 */
  {
    
  forward_lite_slice_slice_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_2 */
  /* LITE_KERNEL_SECTION BEGIN neg */
  {
    
  forward_lite_nl_integer_neg(net_ctx);
  }
  /* LITE_KERNEL_SECTION END neg */
  /* LITE_KERNEL_SECTION BEGIN slice_1 */
  {
    
  forward_lite_slice_slice_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_1 */
  /* LITE_KERNEL_SECTION BEGIN cat_1 */
  {
    
  forward_lite_concat_cat_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cat_1 */
  /* LITE_KERNEL_SECTION BEGIN mul_6 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_6 */
  /* LITE_KERNEL_SECTION BEGIN mul_5 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_5 */
  /* LITE_KERNEL_SECTION BEGIN add_5 */
  {
    
  forward_lite_eltwise_integer_INT8_add_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_5 */
  /* LITE_KERNEL_SECTION BEGIN val_173 */
  {
    
  forward_lite_eltwise_integer_INT8_val_173(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_173 */
  /* LITE_KERNEL_SECTION BEGIN val_173_0_0_val_179_conversion */
  {
      const ai_i8* val_173_0_0_val_179_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8192);
    ai_float* val_173_0_0_val_179_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 74004);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(216, 1, {(stai_ptr) val_173_0_0_val_179_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_173_0_0_val_179_conversion_t_in_0_ptr_const_s8, val_173_0_0_val_179_conversion_t_out_0_ptr_f32, val_173_0_0_val_179_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_173_0_0_val_179_conversion_t_in_0_fmt_scale_const_f32, val_173_0_0_val_179_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(216, 1, {(stai_ptr) val_173_0_0_val_179_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_173_0_0_val_179_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_179 */
  {
    
  forward_lite_matmul_val_179(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_179 */
  /* LITE_KERNEL_SECTION BEGIN val_179_0_0_val_180_conversion */
  {
      const ai_float* val_179_0_0_val_180_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* val_179_0_0_val_180_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(231, 1, {(stai_ptr) val_179_0_0_val_180_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(val_179_0_0_val_180_conversion_t_in_0_ptr_const_f32, val_179_0_0_val_180_conversion_t_out_0_ptr_s8, val_179_0_0_val_180_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_179_0_0_val_180_conversion_t_out_0_fmt_scale_const_f32, val_179_0_0_val_180_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(231, 1, {(stai_ptr) val_179_0_0_val_180_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END val_179_0_0_val_180_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_35 */
  {
    
  forward_lite_gather_nd_val_35(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_35 */
  /* LITE_KERNEL_SECTION BEGIN val_35_0_1_bitwise_and_1_conversion */
  {
      const ai_float* val_35_0_1_bitwise_and_1_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* val_35_0_1_bitwise_and_1_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 1, {(stai_ptr) val_35_0_1_bitwise_and_1_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(val_35_0_1_bitwise_and_1_conversion_t_in_0_ptr_const_f32, val_35_0_1_bitwise_and_1_conversion_t_out_0_ptr_s8, val_35_0_1_bitwise_and_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_35_0_1_bitwise_and_1_conversion_t_out_0_fmt_scale_const_f32, val_35_0_1_bitwise_and_1_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, {(stai_ptr) val_35_0_1_bitwise_and_1_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END val_35_0_1_bitwise_and_1_conversion */
  /* LITE_KERNEL_SECTION BEGIN bitwise_and_1 */
  {
    
  forward_lite_eltwise_integer_INT8_bitwise_and_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END bitwise_and_1 */
  /* LITE_KERNEL_SECTION BEGIN bitwise_and_1_0_1___bool_fix_inv_conversion */
  {
      const ai_i8* bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 160);
    ai_float* bitwise_and_1_0_1___bool_fix_inv_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1184);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(102, 1, {(stai_ptr) bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_ptr_const_s8, bitwise_and_1_0_1___bool_fix_inv_conversion_t_out_0_ptr_f32, bitwise_and_1_0_1___bool_fix_inv_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_fmt_scale_const_f32, bitwise_and_1_0_1___bool_fix_inv_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(102, 1, {(stai_ptr) bitwise_and_1_0_1___bool_fix_inv_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END bitwise_and_1_0_1___bool_fix_inv_conversion */
  /* LITE_KERNEL_SECTION BEGIN __bool_fix_inv */
  {
    
  forward_lite_eltwise___bool_fix_inv(net_ctx);
  }
  /* LITE_KERNEL_SECTION END __bool_fix_inv */
  /* LITE_KERNEL_SECTION BEGIN __bool_fix_inv_0_0_val_178_conversion */
  {
      const ai_float* __bool_fix_inv_0_0_val_178_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 5280);
    ai_i8* __bool_fix_inv_0_0_val_178_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(114, 1, {(stai_ptr) __bool_fix_inv_0_0_val_178_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(__bool_fix_inv_0_0_val_178_conversion_t_in_0_ptr_const_f32, __bool_fix_inv_0_0_val_178_conversion_t_out_0_ptr_s8, __bool_fix_inv_0_0_val_178_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, __bool_fix_inv_0_0_val_178_conversion_t_out_0_fmt_scale_const_f32, __bool_fix_inv_0_0_val_178_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(114, 1, {(stai_ptr) __bool_fix_inv_0_0_val_178_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END __bool_fix_inv_0_0_val_178_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_178 */
  {
    
  forward_lite_eltwise_integer_INT8_val_178(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_178 */
  /* LITE_KERNEL_SECTION BEGIN val_180 */
  {
    
  forward_lite_eltwise_integer_INT8_val_180(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_180 */
  /* LITE_KERNEL_SECTION BEGIN val_180_0_0_val_181_conversion */
  {
      const ai_i8* val_180_0_0_val_181_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2048);
    ai_float* val_180_0_0_val_181_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 6144);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(234, 1, {(stai_ptr) val_180_0_0_val_181_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_180_0_0_val_181_conversion_t_in_0_ptr_const_s8, val_180_0_0_val_181_conversion_t_out_0_ptr_f32, val_180_0_0_val_181_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_180_0_0_val_181_conversion_t_in_0_fmt_scale_const_f32, val_180_0_0_val_181_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(234, 1, {(stai_ptr) val_180_0_0_val_181_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_180_0_0_val_181_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_181 */
  {
      ai_handle val_181_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 22528);
    const ai_handle val_181_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 6144);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(237, 1, {(stai_ptr) val_181_t_in_0_ptr_const_handle});
    
  forward_lite_nl_softmax_if32of32(val_181_t_out_0_ptr_handle, val_181_t_in_0_ptr_const_handle, val_181_t_in_0_shape_ch_h_w_prod_const_s32, 1, 32);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(237, 1, {(stai_ptr) val_181_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_181 */
  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention */
  {
    
  forward_lite_matmul_scaled_dot_product_attention(net_ctx);
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention */
  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention_0_0_transpose_4_conversion */
  {
      const ai_float* scaled_dot_product_attention_0_0_transpose_4_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 41232);
    ai_i8* scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 2048);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(243, 1, {(stai_ptr) scaled_dot_product_attention_0_0_transpose_4_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(scaled_dot_product_attention_0_0_transpose_4_conversion_t_in_0_ptr_const_f32, scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_ptr_s8, scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_fmt_scale_const_f32, scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(243, 1, {(stai_ptr) scaled_dot_product_attention_0_0_transpose_4_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention_0_0_transpose_4_conversion */
  /* LITE_KERNEL_SECTION BEGIN transpose_4 */
  {
    
  forward_lite_transpose_transpose_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_4 */
  /* LITE_KERNEL_SECTION BEGIN linear_3 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_3 */
  /* LITE_KERNEL_SECTION BEGIN add_7 */
  {
    
  forward_lite_eltwise_integer_INT8_add_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_7 */
  /* LITE_KERNEL_SECTION BEGIN add_7_0_0_pow_2_conversion */
  {
      const ai_i8* add_7_0_0_pow_2_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 10240);
    ai_float* add_7_0_0_pow_2_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(255, 1, {(stai_ptr) add_7_0_0_pow_2_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_7_0_0_pow_2_conversion_t_in_0_ptr_const_s8, add_7_0_0_pow_2_conversion_t_out_0_ptr_f32, add_7_0_0_pow_2_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_7_0_0_pow_2_conversion_t_in_0_fmt_scale_const_f32, add_7_0_0_pow_2_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(255, 1, {(stai_ptr) add_7_0_0_pow_2_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_7_0_0_pow_2_conversion */
  /* LITE_KERNEL_SECTION BEGIN pow_2 */
  {
    
  forward_lite_eltwise_pow_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pow_2 */
  /* LITE_KERNEL_SECTION BEGIN mean_1 */
  {
    
  forward_lite_reduce_mean_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mean_1 */
  /* LITE_KERNEL_SECTION BEGIN mean_1_Mul */
  {
      ai_float* mean_1_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    const ai_float* mean_1_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_float* mean_1_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168756);
    const ai_float* mean_1_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(259, 1, {(stai_ptr) mean_1_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(mean_1_Mul_t_out_0_ptr_f32, mean_1_Mul_t_in_0_ptr_const_f32, mean_1_Mul_t_weight_0_ptr_const_f32, mean_1_Mul_t_weight_1_ptr_const_f32, (ai_u32)(32), (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(259, 1, {(stai_ptr) mean_1_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mean_1_Mul */
  /* LITE_KERNEL_SECTION BEGIN mean_1_Mul_0_0_add_8_conversion */
  {
      const ai_float* mean_1_Mul_0_0_add_8_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    ai_i8* mean_1_Mul_0_0_add_8_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(259, 1, {(stai_ptr) mean_1_Mul_0_0_add_8_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(mean_1_Mul_0_0_add_8_conversion_t_in_0_ptr_const_f32, mean_1_Mul_0_0_add_8_conversion_t_out_0_ptr_s8, mean_1_Mul_0_0_add_8_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mean_1_Mul_0_0_add_8_conversion_t_out_0_fmt_scale_const_f32, mean_1_Mul_0_0_add_8_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(259, 1, {(stai_ptr) mean_1_Mul_0_0_add_8_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END mean_1_Mul_0_0_add_8_conversion */
  /* LITE_KERNEL_SECTION BEGIN add_8 */
  {
    
  forward_lite_eltwise_integer_INT8_add_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_8 */
  /* LITE_KERNEL_SECTION BEGIN add_8_0_0_val_193_conversion */
  {
      const ai_i8* add_8_0_0_val_193_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 32);
    ai_float* add_8_0_0_val_193_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(262, 1, {(stai_ptr) add_8_0_0_val_193_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_8_0_0_val_193_conversion_t_in_0_ptr_const_s8, add_8_0_0_val_193_conversion_t_out_0_ptr_f32, add_8_0_0_val_193_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_8_0_0_val_193_conversion_t_in_0_fmt_scale_const_f32, add_8_0_0_val_193_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(262, 1, {(stai_ptr) add_8_0_0_val_193_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_8_0_0_val_193_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_193 */
  {
      ai_handle val_193_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 192);
    const ai_handle val_193_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(265, 1, {(stai_ptr) val_193_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sqrt_if32of32(val_193_t_out_0_ptr_handle, val_193_t_in_0_ptr_const_handle, val_193_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(265, 1, {(stai_ptr) val_193_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_193 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_1 */
  {
      ai_handle rsqrt_1_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle rsqrt_1_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(266, 1, {(stai_ptr) rsqrt_1_t_in_0_ptr_const_handle});
    
  forward_lite_nl_reciprocal_if32of32(rsqrt_1_t_out_0_ptr_handle, rsqrt_1_t_in_0_ptr_const_handle, rsqrt_1_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(266, 1, {(stai_ptr) rsqrt_1_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END rsqrt_1 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_1_0_1_mul_9_conversion */
  {
      const ai_float* rsqrt_1_0_1_mul_9_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* rsqrt_1_0_1_mul_9_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(266, 1, {(stai_ptr) rsqrt_1_0_1_mul_9_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(rsqrt_1_0_1_mul_9_conversion_t_in_0_ptr_const_f32, rsqrt_1_0_1_mul_9_conversion_t_out_0_ptr_s8, rsqrt_1_0_1_mul_9_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, rsqrt_1_0_1_mul_9_conversion_t_out_0_fmt_scale_const_f32, rsqrt_1_0_1_mul_9_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(266, 1, {(stai_ptr) rsqrt_1_0_1_mul_9_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END rsqrt_1_0_1_mul_9_conversion */
  /* LITE_KERNEL_SECTION BEGIN mul_9 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_9 */
  /* LITE_KERNEL_SECTION BEGIN mul_10 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_10 */
  /* LITE_KERNEL_SECTION BEGIN linear_4 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_4 */
  /* LITE_KERNEL_SECTION BEGIN val_195 */
  {
    
  forward_lite_nl_integer_val_195(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_195 */
  /* LITE_KERNEL_SECTION BEGIN silu */
  {
    
  forward_lite_eltwise_integer_INT8_silu(net_ctx);
  }
  /* LITE_KERNEL_SECTION END silu */
  /* LITE_KERNEL_SECTION BEGIN linear_5 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_5 */
  /* LITE_KERNEL_SECTION BEGIN mul_11 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_11(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_11 */
  /* LITE_KERNEL_SECTION BEGIN linear_6 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_6 */
  /* LITE_KERNEL_SECTION BEGIN add_9 */
  {
    
  forward_lite_eltwise_integer_INT8_add_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_9 */
  /* LITE_KERNEL_SECTION BEGIN add_9_0_0_pow_3_conversion */
  {
      const ai_i8* add_9_0_0_pow_3_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2048);
    ai_float* add_9_0_0_pow_3_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(293, 1, {(stai_ptr) add_9_0_0_pow_3_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_9_0_0_pow_3_conversion_t_in_0_ptr_const_s8, add_9_0_0_pow_3_conversion_t_out_0_ptr_f32, add_9_0_0_pow_3_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_9_0_0_pow_3_conversion_t_in_0_fmt_scale_const_f32, add_9_0_0_pow_3_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(293, 1, {(stai_ptr) add_9_0_0_pow_3_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_9_0_0_pow_3_conversion */
  /* LITE_KERNEL_SECTION BEGIN pow_3 */
  {
    
  forward_lite_eltwise_pow_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pow_3 */
  /* LITE_KERNEL_SECTION BEGIN mean_2 */
  {
    
  forward_lite_reduce_mean_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mean_2 */
  /* LITE_KERNEL_SECTION BEGIN mean_2_Mul */
  {
      ai_float* mean_2_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    const ai_float* mean_2_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_float* mean_2_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168756);
    const ai_float* mean_2_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(297, 1, {(stai_ptr) mean_2_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(mean_2_Mul_t_out_0_ptr_f32, mean_2_Mul_t_in_0_ptr_const_f32, mean_2_Mul_t_weight_0_ptr_const_f32, mean_2_Mul_t_weight_1_ptr_const_f32, (ai_u32)(32), (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(297, 1, {(stai_ptr) mean_2_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mean_2_Mul */
  /* LITE_KERNEL_SECTION BEGIN mean_2_Mul_0_0_add_10_conversion */
  {
      const ai_float* mean_2_Mul_0_0_add_10_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    ai_i8* mean_2_Mul_0_0_add_10_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(297, 1, {(stai_ptr) mean_2_Mul_0_0_add_10_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(mean_2_Mul_0_0_add_10_conversion_t_in_0_ptr_const_f32, mean_2_Mul_0_0_add_10_conversion_t_out_0_ptr_s8, mean_2_Mul_0_0_add_10_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mean_2_Mul_0_0_add_10_conversion_t_out_0_fmt_scale_const_f32, mean_2_Mul_0_0_add_10_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(297, 1, {(stai_ptr) mean_2_Mul_0_0_add_10_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END mean_2_Mul_0_0_add_10_conversion */
  /* LITE_KERNEL_SECTION BEGIN add_10 */
  {
    
  forward_lite_eltwise_integer_INT8_add_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_10 */
  /* LITE_KERNEL_SECTION BEGIN add_10_0_0_val_201_conversion */
  {
      const ai_i8* add_10_0_0_val_201_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 32);
    ai_float* add_10_0_0_val_201_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(300, 1, {(stai_ptr) add_10_0_0_val_201_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_10_0_0_val_201_conversion_t_in_0_ptr_const_s8, add_10_0_0_val_201_conversion_t_out_0_ptr_f32, add_10_0_0_val_201_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_10_0_0_val_201_conversion_t_in_0_fmt_scale_const_f32, add_10_0_0_val_201_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(300, 1, {(stai_ptr) add_10_0_0_val_201_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_10_0_0_val_201_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_201 */
  {
      ai_handle val_201_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 192);
    const ai_handle val_201_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(303, 1, {(stai_ptr) val_201_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sqrt_if32of32(val_201_t_out_0_ptr_handle, val_201_t_in_0_ptr_const_handle, val_201_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(303, 1, {(stai_ptr) val_201_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_201 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_2 */
  {
      ai_handle rsqrt_2_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle rsqrt_2_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(304, 1, {(stai_ptr) rsqrt_2_t_in_0_ptr_const_handle});
    
  forward_lite_nl_reciprocal_if32of32(rsqrt_2_t_out_0_ptr_handle, rsqrt_2_t_in_0_ptr_const_handle, rsqrt_2_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(304, 1, {(stai_ptr) rsqrt_2_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END rsqrt_2 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_2_0_1_mul_12_conversion */
  {
      const ai_float* rsqrt_2_0_1_mul_12_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* rsqrt_2_0_1_mul_12_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(304, 1, {(stai_ptr) rsqrt_2_0_1_mul_12_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(rsqrt_2_0_1_mul_12_conversion_t_in_0_ptr_const_f32, rsqrt_2_0_1_mul_12_conversion_t_out_0_ptr_s8, rsqrt_2_0_1_mul_12_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, rsqrt_2_0_1_mul_12_conversion_t_out_0_fmt_scale_const_f32, rsqrt_2_0_1_mul_12_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(304, 1, {(stai_ptr) rsqrt_2_0_1_mul_12_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END rsqrt_2_0_1_mul_12_conversion */
  /* LITE_KERNEL_SECTION BEGIN mul_12 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_12(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_12 */
  /* LITE_KERNEL_SECTION BEGIN mul_13 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_13(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_13 */
  /* LITE_KERNEL_SECTION BEGIN val_203 */
  {
    
  forward_lite_conv2d_integer_SSSA_val_203(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_203 */
  /* LITE_KERNEL_SECTION BEGIN linear_7 */
  {
    
  forward_lite_eltwise_integer_INT8_linear_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_7 */
  /* LITE_KERNEL_SECTION BEGIN transpose_5 */
  {
    
  forward_lite_transpose_transpose_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_5 */
  /* LITE_KERNEL_SECTION BEGIN slice_6 */
  {
    
  forward_lite_slice_slice_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_6 */
  /* LITE_KERNEL_SECTION BEGIN neg_2 */
  {
    
  forward_lite_nl_integer_neg_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END neg_2 */
  /* LITE_KERNEL_SECTION BEGIN slice_5 */
  {
    
  forward_lite_slice_slice_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_5 */
  /* LITE_KERNEL_SECTION BEGIN cat_3 */
  {
    
  forward_lite_concat_cat_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cat_3 */
  /* LITE_KERNEL_SECTION BEGIN mul_15 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_15(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_15 */
  /* LITE_KERNEL_SECTION BEGIN mul_14 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_14(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_14 */
  /* LITE_KERNEL_SECTION BEGIN add_11 */
  {
    
  forward_lite_eltwise_integer_INT8_add_11(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_11 */
  /* LITE_KERNEL_SECTION BEGIN val_306 */
  {
    
  forward_lite_eltwise_integer_INT8_val_306(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_306 */
  /* LITE_KERNEL_SECTION BEGIN val_306_0_0_val_312_conversion */
  {
      const ai_i8* val_306_0_0_val_312_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 26624);
    ai_float* val_306_0_0_val_312_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 41232);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(398, 1, {(stai_ptr) val_306_0_0_val_312_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_306_0_0_val_312_conversion_t_in_0_ptr_const_s8, val_306_0_0_val_312_conversion_t_out_0_ptr_f32, val_306_0_0_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_306_0_0_val_312_conversion_t_in_0_fmt_scale_const_f32, val_306_0_0_val_312_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(398, 1, {(stai_ptr) val_306_0_0_val_312_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_306_0_0_val_312_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_211 */
  {
    
  forward_lite_conv2d_integer_SSSA_val_211(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_211 */
  /* LITE_KERNEL_SECTION BEGIN linear_8 */
  {
    
  forward_lite_eltwise_integer_INT8_linear_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_8 */
  /* LITE_KERNEL_SECTION BEGIN transpose_6 */
  {
    
  forward_lite_transpose_transpose_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_6 */
  /* LITE_KERNEL_SECTION BEGIN slice_8 */
  {
    
  forward_lite_slice_slice_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_8 */
  /* LITE_KERNEL_SECTION BEGIN neg_3 */
  {
    
  forward_lite_nl_integer_neg_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END neg_3 */
  /* LITE_KERNEL_SECTION BEGIN slice_7 */
  {
    
  forward_lite_slice_slice_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_7 */
  /* LITE_KERNEL_SECTION BEGIN cat_4 */
  {
    
  forward_lite_concat_cat_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END cat_4 */
  /* LITE_KERNEL_SECTION BEGIN mul_17 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_17(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_17 */
  /* LITE_KERNEL_SECTION BEGIN mul_16 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_16(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_16 */
  /* LITE_KERNEL_SECTION BEGIN add_12 */
  {
    
  forward_lite_eltwise_integer_INT8_add_12(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_12 */
  /* LITE_KERNEL_SECTION BEGIN expand_5 */
  {
    
  forward_lite_tile_expand_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END expand_5 */
  /* LITE_KERNEL_SECTION BEGIN val_302 */
  {
    
  forward_lite_transpose_val_302(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_302 */
  /* LITE_KERNEL_SECTION BEGIN val_308 */
  {
    
  forward_lite_eltwise_integer_INT8_val_308(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_308 */
  /* LITE_KERNEL_SECTION BEGIN val_308_0_1_val_312_conversion */
  {
      const ai_i8* val_308_0_1_val_312_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 26624);
    ai_float* val_308_0_1_val_312_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 74000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(410, 1, {(stai_ptr) val_308_0_1_val_312_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_308_0_1_val_312_conversion_t_in_0_ptr_const_s8, val_308_0_1_val_312_conversion_t_out_0_ptr_f32, val_308_0_1_val_312_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_308_0_1_val_312_conversion_t_in_0_fmt_scale_const_f32, val_308_0_1_val_312_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(410, 1, {(stai_ptr) val_308_0_1_val_312_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_308_0_1_val_312_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_312 */
  {
    
  forward_lite_matmul_val_312(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_312 */
  /* LITE_KERNEL_SECTION BEGIN val_312_0_0_val_313_conversion */
  {
      const ai_float* val_312_0_0_val_313_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 106768);
    ai_i8* val_312_0_0_val_313_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10240);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(413, 1, {(stai_ptr) val_312_0_0_val_313_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(val_312_0_0_val_313_conversion_t_in_0_ptr_const_f32, val_312_0_0_val_313_conversion_t_out_0_ptr_s8, val_312_0_0_val_313_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_312_0_0_val_313_conversion_t_out_0_fmt_scale_const_f32, val_312_0_0_val_313_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(413, 1, {(stai_ptr) val_312_0_0_val_313_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END val_312_0_0_val_313_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_313 */
  {
    
  forward_lite_eltwise_integer_INT8_val_313(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_313 */
  /* LITE_KERNEL_SECTION BEGIN val_313_0_0_val_314_conversion */
  {
      const ai_i8* val_313_0_0_val_314_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14336);
    ai_float* val_313_0_0_val_314_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 40972);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(416, 1, {(stai_ptr) val_313_0_0_val_314_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(val_313_0_0_val_314_conversion_t_in_0_ptr_const_s8, val_313_0_0_val_314_conversion_t_out_0_ptr_f32, val_313_0_0_val_314_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, val_313_0_0_val_314_conversion_t_in_0_fmt_scale_const_f32, val_313_0_0_val_314_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(416, 1, {(stai_ptr) val_313_0_0_val_314_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END val_313_0_0_val_314_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_314 */
  {
      ai_handle val_314_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 57356);
    const ai_handle val_314_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 40972);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(419, 1, {(stai_ptr) val_314_t_in_0_ptr_const_handle});
    
  forward_lite_nl_softmax_if32of32(val_314_t_out_0_ptr_handle, val_314_t_in_0_ptr_const_handle, val_314_t_in_0_shape_ch_h_w_prod_const_s32, 1, 32);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(419, 1, {(stai_ptr) val_314_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_314 */
  /* LITE_KERNEL_SECTION BEGIN val_219 */
  {
    
  forward_lite_conv2d_integer_SSSA_val_219(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_219 */
  /* LITE_KERNEL_SECTION BEGIN linear_9 */
  {
    
  forward_lite_eltwise_integer_INT8_linear_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_9 */
  /* LITE_KERNEL_SECTION BEGIN transpose_7 */
  {
    
  forward_lite_transpose_transpose_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_7 */
  /* LITE_KERNEL_SECTION BEGIN expand_6 */
  {
    
  forward_lite_tile_expand_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END expand_6 */
  /* LITE_KERNEL_SECTION BEGIN expand_6_0_1_scaled_dot_product_attention_1_conversion */
  {
      const ai_i8* expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 14336);
    ai_float* expand_6_0_1_scaled_dot_product_attention_1_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 73740);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(372, 1, {(stai_ptr) expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_ptr_const_s8, expand_6_0_1_scaled_dot_product_attention_1_conversion_t_out_0_ptr_f32, expand_6_0_1_scaled_dot_product_attention_1_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_fmt_scale_const_f32, expand_6_0_1_scaled_dot_product_attention_1_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(372, 1, {(stai_ptr) expand_6_0_1_scaled_dot_product_attention_1_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END expand_6_0_1_scaled_dot_product_attention_1_conversion */
  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention_1 */
  {
    
  forward_lite_matmul_scaled_dot_product_attention_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention_1 */
  /* LITE_KERNEL_SECTION BEGIN scaled_dot_product_attention_1_0_0_transpose_8_conversion */
  {
      const ai_float* scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 106508);
    ai_i8* scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10240);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(425, 1, {(stai_ptr) scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_in_0_ptr_const_f32, scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_ptr_s8, scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_fmt_scale_const_f32, scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(425, 1, {(stai_ptr) scaled_dot_product_attention_1_0_0_transpose_8_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END scaled_dot_product_attention_1_0_0_transpose_8_conversion */
  /* LITE_KERNEL_SECTION BEGIN transpose_8 */
  {
    
  forward_lite_transpose_transpose_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_8 */
  /* LITE_KERNEL_SECTION BEGIN linear_10 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_10 */
  /* LITE_KERNEL_SECTION BEGIN add_13 */
  {
    
  forward_lite_eltwise_integer_INT8_add_13(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_13 */
  /* LITE_KERNEL_SECTION BEGIN add_13_0_0_pow_4_conversion */
  {
      const ai_i8* add_13_0_0_pow_4_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18432);
    ai_float* add_13_0_0_pow_4_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 26624);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(437, 1, {(stai_ptr) add_13_0_0_pow_4_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_13_0_0_pow_4_conversion_t_in_0_ptr_const_s8, add_13_0_0_pow_4_conversion_t_out_0_ptr_f32, add_13_0_0_pow_4_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_13_0_0_pow_4_conversion_t_in_0_fmt_scale_const_f32, add_13_0_0_pow_4_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(437, 1, {(stai_ptr) add_13_0_0_pow_4_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_13_0_0_pow_4_conversion */
  /* LITE_KERNEL_SECTION BEGIN pow_4 */
  {
    
  forward_lite_eltwise_pow_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pow_4 */
  /* LITE_KERNEL_SECTION BEGIN mean_3 */
  {
    
  forward_lite_reduce_mean_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mean_3 */
  /* LITE_KERNEL_SECTION BEGIN mean_3_Mul */
  {
      ai_float* mean_3_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    const ai_float* mean_3_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_float* mean_3_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168756);
    const ai_float* mean_3_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(441, 1, {(stai_ptr) mean_3_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(mean_3_Mul_t_out_0_ptr_f32, mean_3_Mul_t_in_0_ptr_const_f32, mean_3_Mul_t_weight_0_ptr_const_f32, mean_3_Mul_t_weight_1_ptr_const_f32, (ai_u32)(32), (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(441, 1, {(stai_ptr) mean_3_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mean_3_Mul */
  /* LITE_KERNEL_SECTION BEGIN mean_3_Mul_0_0_add_14_conversion */
  {
      const ai_float* mean_3_Mul_0_0_add_14_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 128);
    ai_i8* mean_3_Mul_0_0_add_14_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(441, 1, {(stai_ptr) mean_3_Mul_0_0_add_14_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(mean_3_Mul_0_0_add_14_conversion_t_in_0_ptr_const_f32, mean_3_Mul_0_0_add_14_conversion_t_out_0_ptr_s8, mean_3_Mul_0_0_add_14_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mean_3_Mul_0_0_add_14_conversion_t_out_0_fmt_scale_const_f32, mean_3_Mul_0_0_add_14_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(441, 1, {(stai_ptr) mean_3_Mul_0_0_add_14_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END mean_3_Mul_0_0_add_14_conversion */
  /* LITE_KERNEL_SECTION BEGIN add_14 */
  {
    
  forward_lite_eltwise_integer_INT8_add_14(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_14 */
  /* LITE_KERNEL_SECTION BEGIN add_14_0_0_val_326_conversion */
  {
      const ai_i8* add_14_0_0_val_326_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 32);
    ai_float* add_14_0_0_val_326_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(444, 1, {(stai_ptr) add_14_0_0_val_326_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_14_0_0_val_326_conversion_t_in_0_ptr_const_s8, add_14_0_0_val_326_conversion_t_out_0_ptr_f32, add_14_0_0_val_326_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_14_0_0_val_326_conversion_t_in_0_fmt_scale_const_f32, add_14_0_0_val_326_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(444, 1, {(stai_ptr) add_14_0_0_val_326_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_14_0_0_val_326_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_326 */
  {
      ai_handle val_326_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 192);
    const ai_handle val_326_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 64);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(447, 1, {(stai_ptr) val_326_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sqrt_if32of32(val_326_t_out_0_ptr_handle, val_326_t_in_0_ptr_const_handle, val_326_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(447, 1, {(stai_ptr) val_326_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_326 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_3 */
  {
      ai_handle rsqrt_3_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle rsqrt_3_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(448, 1, {(stai_ptr) rsqrt_3_t_in_0_ptr_const_handle});
    
  forward_lite_nl_reciprocal_if32of32(rsqrt_3_t_out_0_ptr_handle, rsqrt_3_t_in_0_ptr_const_handle, rsqrt_3_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(448, 1, {(stai_ptr) rsqrt_3_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END rsqrt_3 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_3_0_1_mul_18_conversion */
  {
      const ai_float* rsqrt_3_0_1_mul_18_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    ai_i8* rsqrt_3_0_1_mul_18_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(448, 1, {(stai_ptr) rsqrt_3_0_1_mul_18_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(rsqrt_3_0_1_mul_18_conversion_t_in_0_ptr_const_f32, rsqrt_3_0_1_mul_18_conversion_t_out_0_ptr_s8, rsqrt_3_0_1_mul_18_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, rsqrt_3_0_1_mul_18_conversion_t_out_0_fmt_scale_const_f32, rsqrt_3_0_1_mul_18_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(448, 1, {(stai_ptr) rsqrt_3_0_1_mul_18_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END rsqrt_3_0_1_mul_18_conversion */
  /* LITE_KERNEL_SECTION BEGIN mul_18 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_18(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_18 */
  /* LITE_KERNEL_SECTION BEGIN mul_19 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_19(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_19 */
  /* LITE_KERNEL_SECTION BEGIN linear_11 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_11(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_11 */
  /* LITE_KERNEL_SECTION BEGIN val_328 */
  {
    
  forward_lite_nl_integer_val_328(net_ctx);
  }
  /* LITE_KERNEL_SECTION END val_328 */
  /* LITE_KERNEL_SECTION BEGIN silu_1 */
  {
    
  forward_lite_eltwise_integer_INT8_silu_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END silu_1 */
  /* LITE_KERNEL_SECTION BEGIN linear_12 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_12(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_12 */
  /* LITE_KERNEL_SECTION BEGIN mul_20 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_20(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_20 */
  /* LITE_KERNEL_SECTION BEGIN linear_13 */
  {
    
  forward_lite_conv2d_integer_SSSA_linear_13(net_ctx);
  }
  /* LITE_KERNEL_SECTION END linear_13 */
  /* LITE_KERNEL_SECTION BEGIN add_15 */
  {
    
  forward_lite_eltwise_integer_INT8_add_15(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_15 */
  /* LITE_KERNEL_SECTION BEGIN add_15_0_0_pow_5_conversion */
  {
      const ai_i8* add_15_0_0_pow_5_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_float* add_15_0_0_pow_5_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(475, 1, {(stai_ptr) add_15_0_0_pow_5_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_15_0_0_pow_5_conversion_t_in_0_ptr_const_s8, add_15_0_0_pow_5_conversion_t_out_0_ptr_f32, add_15_0_0_pow_5_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_15_0_0_pow_5_conversion_t_in_0_fmt_scale_const_f32, add_15_0_0_pow_5_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(475, 1, {(stai_ptr) add_15_0_0_pow_5_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_15_0_0_pow_5_conversion */
  /* LITE_KERNEL_SECTION BEGIN pow_5 */
  {
    
  forward_lite_eltwise_pow_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pow_5 */
  /* LITE_KERNEL_SECTION BEGIN mean_4 */
  {
    
  forward_lite_reduce_mean_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mean_4 */
  /* LITE_KERNEL_SECTION BEGIN mean_4_Mul */
  {
      ai_float* mean_4_Mul_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8320);
    const ai_float* mean_4_Mul_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    const ai_float* mean_4_Mul_t_weight_0_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168756);
    const ai_float* mean_4_Mul_t_weight_1_ptr_const_f32 = (ai_float*)(net_ctx->_weights[0] + 168760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(479, 1, {(stai_ptr) mean_4_Mul_t_in_0_ptr_const_f32});
    
  forward_lite_bn_if32of32wf32(mean_4_Mul_t_out_0_ptr_f32, mean_4_Mul_t_in_0_ptr_const_f32, mean_4_Mul_t_weight_0_ptr_const_f32, mean_4_Mul_t_weight_1_ptr_const_f32, (ai_u32)(32), (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(479, 1, {(stai_ptr) mean_4_Mul_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END mean_4_Mul */
  /* LITE_KERNEL_SECTION BEGIN mean_4_Mul_0_0_add_16_conversion */
  {
      const ai_float* mean_4_Mul_0_0_add_16_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8320);
    ai_i8* mean_4_Mul_0_0_add_16_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(479, 1, {(stai_ptr) mean_4_Mul_0_0_add_16_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(mean_4_Mul_0_0_add_16_conversion_t_in_0_ptr_const_f32, mean_4_Mul_0_0_add_16_conversion_t_out_0_ptr_s8, mean_4_Mul_0_0_add_16_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, mean_4_Mul_0_0_add_16_conversion_t_out_0_fmt_scale_const_f32, mean_4_Mul_0_0_add_16_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(479, 1, {(stai_ptr) mean_4_Mul_0_0_add_16_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END mean_4_Mul_0_0_add_16_conversion */
  /* LITE_KERNEL_SECTION BEGIN add_16 */
  {
    
  forward_lite_eltwise_integer_INT8_add_16(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_16 */
  /* LITE_KERNEL_SECTION BEGIN add_16_0_0_val_334_conversion */
  {
      const ai_i8* add_16_0_0_val_334_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 8224);
    ai_float* add_16_0_0_val_334_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(482, 1, {(stai_ptr) add_16_0_0_val_334_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(add_16_0_0_val_334_conversion_t_in_0_ptr_const_s8, add_16_0_0_val_334_conversion_t_out_0_ptr_f32, add_16_0_0_val_334_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, add_16_0_0_val_334_conversion_t_in_0_fmt_scale_const_f32, add_16_0_0_val_334_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(482, 1, {(stai_ptr) add_16_0_0_val_334_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END add_16_0_0_val_334_conversion */
  /* LITE_KERNEL_SECTION BEGIN val_334 */
  {
      ai_handle val_334_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 8384);
    const ai_handle val_334_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(485, 1, {(stai_ptr) val_334_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sqrt_if32of32(val_334_t_out_0_ptr_handle, val_334_t_in_0_ptr_const_handle, val_334_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(485, 1, {(stai_ptr) val_334_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END val_334 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_4 */
  {
      ai_handle rsqrt_4_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 8192);
    const ai_handle rsqrt_4_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(486, 1, {(stai_ptr) rsqrt_4_t_in_0_ptr_const_handle});
    
  forward_lite_nl_reciprocal_if32of32(rsqrt_4_t_out_0_ptr_handle, rsqrt_4_t_in_0_ptr_const_handle, rsqrt_4_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(486, 1, {(stai_ptr) rsqrt_4_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END rsqrt_4 */
  /* LITE_KERNEL_SECTION BEGIN rsqrt_4_0_1_mul_21_conversion */
  {
      const ai_float* rsqrt_4_0_1_mul_21_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    ai_i8* rsqrt_4_0_1_mul_21_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8320);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(486, 1, {(stai_ptr) rsqrt_4_0_1_mul_21_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(rsqrt_4_0_1_mul_21_conversion_t_in_0_ptr_const_f32, rsqrt_4_0_1_mul_21_conversion_t_out_0_ptr_s8, rsqrt_4_0_1_mul_21_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, rsqrt_4_0_1_mul_21_conversion_t_out_0_fmt_scale_const_f32, rsqrt_4_0_1_mul_21_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(486, 1, {(stai_ptr) rsqrt_4_0_1_mul_21_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END rsqrt_4_0_1_mul_21_conversion */
  /* LITE_KERNEL_SECTION BEGIN mul_21 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_21(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_21 */
  /* LITE_KERNEL_SECTION BEGIN mul_22 */
  {
    
  forward_lite_eltwise_integer_INT8_mul_22(net_ctx);
  }
  /* LITE_KERNEL_SECTION END mul_22 */
  /* LITE_KERNEL_SECTION BEGIN logits_QuantizeLinear_Input */
  {
    
  forward_lite_conv2d_integer_SSSA_logits_QuantizeLinear_Input(net_ctx);
  }
  /* LITE_KERNEL_SECTION END logits_QuantizeLinear_Input */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 40972;

  net_ctx->_inputs[1] = activations[0] + 41100;

  net_ctx->_outputs[0] = activations[0] + 9216;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

