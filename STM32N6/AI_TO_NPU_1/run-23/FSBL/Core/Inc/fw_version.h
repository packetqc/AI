#ifndef FW_VERSION_H
#define FW_VERSION_H
/* fw_version.h — firmware version stamp for run-23 (nocode calculator).
 *
 * Read the running build LIVE — two independent ways:
 *   (1) UART boot banner: FW_PrintVersion() is called from main.c after "MAIN APP ON".
 *   (2) SWD memory read of the const g_fw_version struct (it lives in rodata, so it is
 *       present at dev=0 in the flashed image). Find its address in the map file
 *       Makefile/FSBL/build/FSBL.map (symbol `g_fw_version`) — stable per build — and
 *       read sizeof(FwVersion) bytes. The 'NOCD' magic (0x4E4F4344) makes the block
 *       scannable over SWD even without the map.
 *
 * nocode_stage tracks the host->device "nocode" evolution, so a live read tells EXACTLY
 * which architecture is on the board:
 *   0 = playbook grammar + compiled nocode_dispatch (baked C ops)          [pre-VM]
 *   1 = hardcoded stack VM runs a model-derived bytecode table            [nocode core]
 *   2 = NPU emits grammar + bytecode at runtime (generative oracle-in-loop)[pure oracle]
 */
#include <stdint.h>

#ifndef FW_GIT_HASH
#define FW_GIT_HASH "nogit"      /* overridden by -DFW_GIT_HASH="..." from the Makefile */
#endif

#define FW_VER_MAJOR      0
#define FW_VER_MINOR      4
#define FW_VER_PATCH      0
#define FW_NOCODE_STAGE   0
#define FW_NOCODE_ARCH    "playbook+compiled-dispatch"
#define FW_VERSION_STR    "run-23 0.4.0"
#define FW_VERSION_MAGIC  0x4E4F4344u   /* 'NOCD' — locate/validate the block over SWD */

typedef struct {
    uint32_t magic;           /* FW_VERSION_MAGIC 'NOCD' */
    uint16_t ver_major;
    uint16_t ver_minor;
    uint16_t ver_patch;
    uint16_t nocode_stage;    /* 0/1/2 — see header comment */
    char     version[24];     /* FW_VERSION_STR */
    char     git_hash[16];    /* short git hash captured at build */
    char     build_date[16];  /* __DATE__ */
    char     build_time[12];  /* __TIME__ */
    char     nocode_arch[32]; /* FW_NOCODE_ARCH */
} FwVersion;

#ifdef __cplusplus
extern "C" {
#endif

extern const FwVersion g_fw_version;   /* the stamp (rodata) */
void FW_PrintVersion(void);            /* print the banner via printf (BSP COM) */

#ifdef __cplusplus
}
#endif
#endif /* FW_VERSION_H */
