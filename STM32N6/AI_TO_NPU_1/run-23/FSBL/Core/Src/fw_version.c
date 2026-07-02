/* fw_version.c — the version stamp instance.
 *
 * `g_fw_version` is `const`, so it is placed in rodata and baked into the flashed image
 * (present at dev=0). __DATE__/__TIME__ are captured by the compiler; the git hash is
 * injected as -DFW_GIT_HASH from the Makefile. The unit is force-rebuilt every make
 * invocation (see Makefile.local force_version), so the date/time/hash are always current.
 */
#include "fw_version.h"
#include <stdio.h>

const FwVersion g_fw_version = {
    .magic        = FW_VERSION_MAGIC,
    .ver_major    = FW_VER_MAJOR,
    .ver_minor    = FW_VER_MINOR,
    .ver_patch    = FW_VER_PATCH,
    .nocode_stage = FW_NOCODE_STAGE,
    .version      = FW_VERSION_STR,
    .git_hash     = FW_GIT_HASH,
    .build_date   = __DATE__,
    .build_time   = __TIME__,
    .nocode_arch  = FW_NOCODE_ARCH,
};

void FW_PrintVersion(void)
{
    printf("\r\n==== FW %s  (git %s, built %s %s)\r\n",
           g_fw_version.version, g_fw_version.git_hash,
           g_fw_version.build_date, g_fw_version.build_time);
    printf("     nocode stage %u: %s\r\n",
           (unsigned)g_fw_version.nocode_stage, g_fw_version.nocode_arch);
}
