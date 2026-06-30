/**
 * @file    lvgl_port_n6.h
 * @brief   Bare-metal LVGL 9.x port — LTDC + PSRAM/AXISRAM framebuffer
 * @target  STM32N6570-DK (adaptable to any N6 board with LTDC + RGB565 panel)
 * @deps    LVGL 9.x (tested 9.2.3), CMSIS core (DWT->CYCCNT, __DSB)
 * @version 1.0.0
 *
 * Architecture:
 *   Caller owns:
 *     - framebuffer allocation (PSRAM 0x90000000 or any LTDC-readable region)
 *     - LTDC peripheral init (pixel format + layer config + CFBAR pointing at fb_addr)
 *     - an optional build_scene callback that creates the initial widgets
 *
 *   Library wires up:
 *     - lv_init + lv_display_create + lv_display_set_buffers(LV_DISPLAY_RENDER_MODE_DIRECT)
 *     - lv_tick_set_cb using DWT->CYCCNT divided by HCLK/1000 (ms)
 *     - lv_display_set_flush_cb that does only __DSB + lv_display_flush_ready
 *       (DIRECT mode — the buffer IS the hardware framebuffer, no copy needed)
 *
 * Usage (bare-metal super-loop):
 *   lvgl_port_n6_cfg_t cfg = {
 *       .fb_addr         = (void *)0x90000000U,
 *       .fb_width        = 800,
 *       .fb_height       = 480,
 *       .fb_bytes_per_px = 2,                     // RGB565
 *       .hclk_hz         = 600000000U,            // N6 default
 *       .build_scene_cb  = my_build_scene,        // may be NULL
 *   };
 *   if (lvgl_port_n6_init(&cfg) != LVGL_PORT_N6_OK) { fault(); }
 *   for (;;) { lvgl_port_n6_run_once(); delay_until_next_frame(); }
 *
 * Usage (ThreadX):
 *   Call lvgl_port_n6_init from main (before tx_kernel_enter).
 *   In a dedicated TX_THREAD body: for(;;) { lvgl_port_n6_run_once(); tx_thread_sleep(3); }
 *   Do NOT call lvgl_port_n6_run_once from multiple tasks — LVGL is single-threaded.
 *
 * Compile flags (MANDATORY on Cortex-M55 + GCC 14.x):
 *   -mno-unaligned-access
 *   Without it, lv_theme_default_init HardFaults on an STRH at odd struct offset 19
 *   inside lv_theme_t::base.color_secondary. See the sibling methodology § 4.
 *
 * Telemetry:
 *   Four volatile globals (lvgl_port_n6_state / tick_ms / flush_count / loop_count)
 *   let a debugger observe execution without halting. Read via GDB or render them
 *   into the scene as a live status label.
 *
 * Prerequisites on the board:
 *   - TIM2_IRQn permanently disabled after HAL_Init (N6 htim2.Instance unstable; methodology § 5).
 *   - HAL_GetTick overridden to read DWT->CYCCNT (since TIM2 tick is disabled).
 *   - LTDC L1 programmed with PFCR=RGB565, CFBAR=fb_addr, CR bit 0 set; GCR bit 0 set.
 *   - MPU/RIF: fb_addr must be CPU-writable AND LTDC-master readable (avoid RIF-blocked regions).
 *
 * Extracted from: FSBL/Display/lvgl_port.c at commit d0ba24c
 * Sibling methodology: Knowledge/K_STM32_MCP/methodology/methodology-lvgl-baremetal-n6.md
 */
#ifndef LVGL_PORT_N6_H
#define LVGL_PORT_N6_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Status codes -------------------------------------------------- */

typedef enum {
    LVGL_PORT_N6_OK        =  0,
    LVGL_PORT_N6_ERR_PARAM = -1,
    LVGL_PORT_N6_ERR_STATE = -2,
    LVGL_PORT_N6_ERR_LVGL  = -3,
} lvgl_port_n6_status_t;

/* ---------- Callback for scene construction ------------------------------- */

typedef void (*lvgl_port_n6_build_scene_cb_t)(void);

/* ---------- Configuration ------------------------------------------------- */

typedef struct {
    void *fb_addr;                                  /* LTDC-readable framebuffer base */
    uint32_t fb_width;                              /* pixels */
    uint32_t fb_height;                             /* pixels */
    uint32_t fb_bytes_per_px;                       /* 2 for RGB565 (only supported for now) */
    uint32_t hclk_hz;                               /* for DWT→ms conversion */
    lvgl_port_n6_build_scene_cb_t build_scene_cb;   /* may be NULL */
} lvgl_port_n6_cfg_t;

/* ---------- Public API ---------------------------------------------------- */

/**
 * @brief  Initialize LVGL on the provided framebuffer in DIRECT render mode.
 * @param  cfg  Library configuration (must not be NULL; valid fb_addr + non-zero dims + hclk_hz).
 * @return LVGL_PORT_N6_OK on success; LVGL_PORT_N6_ERR_PARAM on bad cfg;
 *         LVGL_PORT_N6_ERR_STATE if already initialized;
 *         LVGL_PORT_N6_ERR_LVGL if lv_display_create failed.
 */
lvgl_port_n6_status_t lvgl_port_n6_init(const lvgl_port_n6_cfg_t *cfg);

/**
 * @brief  Drive LVGL once — calls lv_timer_handler, increments loop_count.
 *         Call from main loop at ~30 FPS (33 ms) or from a ThreadX task body.
 *         No-op if init was not successful.
 */
void lvgl_port_n6_run_once(void);

/**
 * @brief  Tear down (rare — most bare-metal apps never deinit).
 */
void lvgl_port_n6_deinit(void);

/* ---------- Live telemetry (observable via ST-LINK / GDB) ----------------- */

/**
 * State sentinel:
 *   0  uninit
 *   10 lv_init OK
 *   20 tick_set_cb OK
 *   30 display_create OK
 *   40 set_color_format OK
 *   50 set_buffers OK
 *   60 set_flush_cb OK
 *   70 build_scene OK
 *   1  init all-done (post-70)
 *   2  first frame flushed out
 *   9  display_create failed
 */
extern volatile uint8_t  lvgl_port_n6_state;

/** Latest DWT-derived ms (updated on every tick query). */
extern volatile uint32_t lvgl_port_n6_tick_ms;

/** Total flush_cb invocations — increments every frame flushed. */
extern volatile uint32_t lvgl_port_n6_flush_count;

/** Total lvgl_port_n6_run_once invocations. */
extern volatile uint32_t lvgl_port_n6_loop_count;

#ifdef __cplusplus
}
#endif
#endif /* LVGL_PORT_N6_H */
