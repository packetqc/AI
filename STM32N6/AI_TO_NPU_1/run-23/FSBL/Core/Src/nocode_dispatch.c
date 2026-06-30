/* nocode_dispatch.c — GENERATED from the function-vocabulary. DO NOT EDIT.
 * source: scripts/classes/class_logic_transposer.py (emit_c_dispatch)
 * grammars: calculator, dataflow_demo
 */
#include "nocode_dispatch.h"
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

static long nc_calculator_number(NcCtx* c){ long r = 0; for (int i = 0; i < c->ndigits; ++i) r = r * 10 + c->digits[i]; return r; }
static long nc_calculator_op_add(NcCtx* c){ return c->a + c->b; }
static long nc_calculator_op_sub(NcCtx* c){ return c->a - c->b; }
static long nc_calculator_op_mul(NcCtx* c){ return c->a * c->b; }
static long nc_calculator_op_div(NcCtx* c){ return c->b != 0 ? c->a / c->b : 0; }
static long nc_dataflow_demo_gather(NcCtx* c){ long v = (c->argc > 0 && c->args[0]) ? (long)strlen(c->args[0]) : 0; nc_put(c, "gather", v); return v; }
static long nc_dataflow_demo_consume(NcCtx* c){ int f = 0; long g = nc_get(c, "gather", &f); return f ? g * 10 : -1; }

const NcEntry NC_DISPATCH[] = {
    { "calculator", "number", nc_calculator_number },
    { "calculator", "op_add", nc_calculator_op_add },
    { "calculator", "op_sub", nc_calculator_op_sub },
    { "calculator", "op_mul", nc_calculator_op_mul },
    { "calculator", "op_div", nc_calculator_op_div },
    { "dataflow_demo", "gather", nc_dataflow_demo_gather },
    { "dataflow_demo", "consume", nc_dataflow_demo_consume },
};
const int NC_DISPATCH_COUNT = (int)(sizeof(NC_DISPATCH) / sizeof(NC_DISPATCH[0]));

NcFn nc_resolve(const char* g, const char* t){
    for (int i = 0; i < NC_DISPATCH_COUNT; ++i)
        if (strcmp(NC_DISPATCH[i].grammar, g) == 0 && strcmp(NC_DISPATCH[i].token, t) == 0)
            return NC_DISPATCH[i].fn;
    return (NcFn)0;
}
#ifdef __cplusplus
}
#endif
