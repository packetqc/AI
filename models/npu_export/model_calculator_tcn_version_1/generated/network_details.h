/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-06-25T23:07:03-0400
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[8]) {
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 256, 32}}, .scale = {1, (const float[1]){0.029472799971699715}}, .zeropoint = {1, (const int16_t[1]){-4}}, .name = "embeddings_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 32, 256}}, .scale = {1, (const float[1]){0.029472799971699715}}, .zeropoint = {1, (const int16_t[1]){-4}}, .name = "embeddings_Transpose_output" },
   { .size_bytes = 8704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 34, 256}}, .scale = {1, (const float[1]){0.029472799971699715}}, .zeropoint = {1, (const int16_t[1]){-4}}, .name = "_pad1_Pad_output_0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 32, 256}}, .scale = {1, (const float[1]){0.04838947206735611}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_act_Relu_output_0_output" },
   { .size_bytes = 8704, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 34, 256}}, .scale = {1, (const float[1]){0.04838947206735611}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_pad2_Pad_output_0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 32, 256}}, .scale = {1, (const float[1]){0.10813739150762558}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "_act_1_Relu_output_0_output" },
   { .size_bytes = 11968, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 32, 374}}, .scale = {1, (const float[1]){0.5506329536437988}}, .zeropoint = {1, (const int16_t[1]){-1}}, .name = "logits_QuantizeLinear_Input_output" },
   { .size_bytes = 11968, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 374, 32}}, .scale = {1, (const float[1]){0.5506329536437988}}, .zeropoint = {1, (const int16_t[1]){-1}}, .name = "logits_QuantizeLinear_Input_Transpose_0_output" }
  },
  .nodes = (const stai_node_details[7]){
    {.id = 2, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* embeddings_Transpose */
    {.id = 45, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _pad1_Pad_output_0 */
    {.id = 48, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _act_Relu_output_0 */
    {.id = 51, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _pad2_Pad_output_0 */
    {.id = 54, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _act_1_Relu_output_0 */
    {.id = 57, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* logits_QuantizeLinear_Input */
    {.id = 1, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} } /* logits_QuantizeLinear_Input_Transpose_0 */
  },
  .n_nodes = 7
};
#endif

