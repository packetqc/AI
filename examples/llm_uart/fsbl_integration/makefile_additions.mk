# makefile_additions.mk
#
# Add these lines to Makefile/FSBL/Makefile to compile the AI inference
# stack into the FSBL.  Paste each block into the matching section.
#
# ── C_SOURCES — add after existing FSBL sources ──────────────────────────
C_SOURCES += \
../../Appli/AI/App/network.c          \
../../Appli/AI/App/network_data.c     \
../../Appli/AI/App/network_weights.c  \
../../Appli/AI/App/user_init.c        \
../../Appli/AI/App/npu_init.c         \
../../Appli/AI/App/npu_cache.c        \
../../Appli/Core/Src/llm_tokenizer.c  \
../../Appli/Core/Src/llm_fsbl.c       \
../../Appli/Core/Src/llm_test_fsbl.c

# ── C_INCLUDES — add after existing FSBL includes ────────────────────────
C_INCLUDES += \
-I../../Appli/AI/App               \
-I../../Appli/Core/Inc             \
-I../../Middlewares/ST/AI/Inc

# ── LIBDIR — AI runtime library path ─────────────────────────────────────
LIBDIR += -L../../Middlewares/ST/AI/Lib

# ── LIBS — link STAI runtime (add before -lc -lm -lnosys) ────────────────
LIBS += -lNetworkRuntime1200_CM55_GCC

# ── C_DEFS — mark this as the FSBL build for conditional compilation ──────
C_DEFS += -DLLM_FSBL_BUILD

# ──────────────────────────────────────────────────────────────────────────
# Note on network_weights.c:
#   In the generated Appli project, weights may be split across
#   network_data.c and network_weights.c — include both.
#   Check ls Appli/AI/App/*.c to confirm file names.
#
# Note on npu_cache.c:
#   Only compiles code when LL_ATON_PLATFORM is defined (NPU path).
#   For CPU-only (no Neural ART), it compiles to nothing — safe to include.
#
# Note on linker:
#   Also update LDSCRIPT to the modified linker file:
#     LDSCRIPT = STM32N657XX_AXISRAM2_fsbl_ai.ld
# ──────────────────────────────────────────────────────────────────────────
