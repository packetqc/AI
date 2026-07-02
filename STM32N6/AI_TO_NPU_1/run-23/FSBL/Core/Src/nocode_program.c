/* nocode_program.c — GENERATED model-driven NATIVE CODE. DO NOT EDIT.
 * source: scripts/classes/class_logic_transposer.py (emit_native_program)
 * bodies: FSBL/AI/nocode_ops.c compiled -DMODEL_CREATION; grammars: calculator
 * Each blob is relocation-free ARM Thumb machine code (ABI: long fn(NcCtx*)), extracted from
 * the project ops object and INJECTED at runtime by nocode_inject.c (the device exec()).
 * The bodies are #ifdef MODEL_CREATION, so the normal firmware bakes NONE of this logic.
 */
#include "nocode_inject.h"

static const uint8_t nc_code_calculator_op_add[] = { 0xd0, 0xe9, 0x09, 0x20, 0x10, 0x44, 0x70, 0x47 };  /* 8 B, from nc_op_calculator_op_add() */
static const uint8_t nc_code_calculator_op_sub[] = { 0xd0, 0xe9, 0x09, 0x20, 0x10, 0x1a, 0x70, 0x47 };  /* 8 B, from nc_op_calculator_op_sub() */
static const uint8_t nc_code_calculator_op_mul[] = { 0xd0, 0xe9, 0x09, 0x30, 0x58, 0x43, 0x70, 0x47 };  /* 8 B, from nc_op_calculator_op_mul() */
static const uint8_t nc_code_calculator_op_div[] = { 0x83, 0x6a, 0x13, 0xb1, 0x42, 0x6a, 0x92, 0xfb, 0xf3, 0xf3, 0x18, 0x46, 0x70, 0x47 };  /* 14 B, from nc_op_calculator_op_div() */
static const uint8_t nc_code_calculator_number[] = { 0xd0, 0xf8, 0xac, 0x20, 0x00, 0xf1, 0x28, 0x03, 0x00, 0x20, 0x82, 0x42, 0x00, 0xb5, 0x02, 0xf1, 0x01, 0x0e, 0xb8, 0xbf, 0x4f, 0xf0, 0x01, 0x0e, 0x0a, 0x21, 0x4e, 0xf0, 0x01, 0xe0, 0xbe, 0xf1, 0x01, 0x0e, 0x00, 0xd1, 0x00, 0xbd, 0x53, 0xf8, 0x04, 0x2f, 0x01, 0xfb, 0x00, 0x20, 0xf6, 0xe7 };  /* 48 B, from nc_op_calculator_number() */

const NcProgram NC_PROGRAMS[] = {
    { "calculator", "op_add", nc_code_calculator_op_add, sizeof(nc_code_calculator_op_add) },
    { "calculator", "op_sub", nc_code_calculator_op_sub, sizeof(nc_code_calculator_op_sub) },
    { "calculator", "op_mul", nc_code_calculator_op_mul, sizeof(nc_code_calculator_op_mul) },
    { "calculator", "op_div", nc_code_calculator_op_div, sizeof(nc_code_calculator_op_div) },
    { "calculator", "number", nc_code_calculator_number, sizeof(nc_code_calculator_number) },
};
const int NC_PROGRAM_COUNT = (int)(sizeof(NC_PROGRAMS) / sizeof(NC_PROGRAMS[0]));
