#ifndef NOCODE_INJECT_H
#define NOCODE_INJECT_H
/* nocode_inject.h — device "nocode" by NATIVE CODE INJECTION (the host model -> NPU path, device end).
 *
 * The device analog of the host runner's exec(): the host injects model-emitted PYTHON source into
 * CPython (a universal machine); the device injects model-provided ARM Thumb MACHINE CODE into the
 * CPU (the real universal machine) and runs it natively. No interpreter, no VM — the op logic IS real
 * STM32 code, COMPILED AGAINST THIS PROJECT at model-creation time by the host create script
 * (class_logic_transposer.emit_native_program, which #includes this very header so the ABI is
 * byte-exact) and carried model-driven. nc_run() copies the bytes into executable RAM, makes the
 * I/D caches coherent, and calls them.
 *
 * Ladder parity with the host exec-policy: stage 1 = carried machine-code table (the token_select
 * rung); stage 2 = the NPU emits the SAME machine code at runtime (the generative rung — pure oracle).
 *
 * Shared by the firmware AND the host compile step, so it stays freestanding (stdint/string only).
 * The struct layout is the ABI contract — the injected code reads ctx by offset — so any change here
 * MUST be followed by regenerating nocode_program.c.
 */
#include <stdint.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifndef NC_MAX_ARGS
#define NC_MAX_ARGS    8
#endif
#ifndef NC_MAX_DIGITS
#define NC_MAX_DIGITS  32
#endif
#ifndef NC_MAX_RESULTS
#define NC_MAX_RESULTS 16
#endif

/* Shared execution context — the ABI the injected native code reads by offset. */
typedef struct NcCtx {
    const char* args[NC_MAX_ARGS];
    int         argc;
    long        a, b;
    long        digits[NC_MAX_DIGITS];
    int         ndigits;
    long        result;
    int         has_result;
    struct { const char* token; long value; } results[NC_MAX_RESULTS];
    int         nresults;
} NcCtx;

/* A model-provided native routine: `code` is relocation-free Thumb machine code with the ABI
 * `long fn(NcCtx*)`, compiled against this header at model-creation time. */
typedef struct NcProgram {
    const char*    grammar;
    const char*    token;
    const uint8_t* code;
    uint16_t       code_len;
} NcProgram;

extern const NcProgram NC_PROGRAMS[];
extern const int       NC_PROGRAM_COUNT;

/* Inject p->code into executable RAM, make caches coherent, call it as long fn(NcCtx*). */
long             nc_run(const NcProgram* p, NcCtx* ctx);
const NcProgram* nc_program(const char* grammar, const char* token);

#ifdef __cplusplus
}
#endif
#endif /* NOCODE_INJECT_H */
