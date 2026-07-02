/* nocode_ops.c — the nocode op/grammar bodies AS STM32 project code (MODEL-CREATION ONLY).
 *
 * FAIL-SAFE by construction: this whole file is `#ifdef MODEL_CREATION`. It is compiled ONLY when the
 * host grammar/model-creation step builds the project with -DMODEL_CREATION — that build exists purely
 * so the host can EXTRACT each function's Thumb machine code (relocation-free .text) and embed it in the
 * model (-> generated nocode_program.c). In the NORMAL firmware build MODEL_CREATION is undefined, so
 * this file compiles to NOTHING: the firmware provably carries NO baked op logic, only the injector
 * (nocode_inject.c) running the model-provided code. So a load-and-run of the normal firmware that still
 * computes correctly proves the op code is model-driven, not hand-written CPU logic.
 *
 * ABI: every op is `long fn(NcCtx*)`, pure arithmetic over the ctx (call-free) so the extracted .text is
 * position-independent and injectable. Naming convention (the transposer maps it): nc_op_<grammar>_<token>.
 */
#ifdef MODEL_CREATION

#include "nocode_inject.h"

/* --- grammar: calculator (evaluate_ops) --- */
long nc_op_calculator_op_add(NcCtx* c) { return c->a + c->b; }
long nc_op_calculator_op_sub(NcCtx* c) { return c->a - c->b; }
long nc_op_calculator_op_mul(NcCtx* c) { return c->a * c->b; }
long nc_op_calculator_op_div(NcCtx* c) { return c->b != 0 ? c->a / c->b : 0; }
long nc_op_calculator_number(NcCtx* c)
{
    long r = 0;
    int i;
    for (i = 0; i < c->ndigits; ++i) r = r * 10 + c->digits[i];
    return r;
}

#endif /* MODEL_CREATION */
