/* nocode_inject.c — the code-injection engine (the device's "exec()").
 *
 * Copies model-provided ARM Thumb machine code into an executable RAM buffer, performs the M55
 * self-modifying-code cache maintenance (clean D-cache so the bytes reach unified memory, invalidate
 * I-cache so the fetch side sees them, DSB+ISB to order the writes and flush the pipeline), then
 * calls it via a Thumb function pointer. Bare-metal safe: fixed static buffer, no heap, no recursion.
 *
 * Executability: the FSBL runs its own .text from AXISRAM with the MPU relaxed, so this .bss buffer
 * (same memory) is executable under the ARMv8-M default map. If a build ever marks it XN, add an RWX
 * MPU region covering s_inject.
 */
#include "nocode_inject.h"
#include "stm32n6xx.h"     /* SCB_* cache ops, __DSB/__ISB (core_cm55.h) */

/* 32-byte aligned so cache maintenance by address covers whole lines; sized for the small op blobs
 * (a calculator op compiles to < 32 B of Thumb). */
static uint8_t __attribute__((aligned(32))) s_inject[256];

long nc_run(const NcProgram* p, NcCtx* ctx)
{
    if (p == 0 || p->code == 0 || p->code_len == 0 || p->code_len > sizeof(s_inject))
        return 0;

    memcpy(s_inject, p->code, p->code_len);

    /* make the freshly written bytes fetchable + executable */
    SCB_CleanDCache_by_Addr((uint32_t *)(void *)s_inject, (int32_t)sizeof(s_inject));
    SCB_InvalidateICache();
    __DSB();
    __ISB();

    long (*fn)(NcCtx *) = (long (*)(NcCtx *))((uintptr_t)s_inject | 1u);  /* Thumb bit */
    return fn(ctx);
}

const NcProgram* nc_program(const char* grammar, const char* token)
{
    int i;
    for (i = 0; i < NC_PROGRAM_COUNT; ++i)
        if (strcmp(NC_PROGRAMS[i].grammar, grammar) == 0 &&
            strcmp(NC_PROGRAMS[i].token, token) == 0)
            return &NC_PROGRAMS[i];
    return (const NcProgram*)0;
}
