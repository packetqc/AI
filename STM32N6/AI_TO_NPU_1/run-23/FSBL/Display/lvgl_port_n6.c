/**
 * @file    lvgl_port_n6.c
 * @brief   Implementation for lvgl_port_n6.h
 * @version 1.0.0
 *
 * Extracted from FSBL/Display/lvgl_port.c at commit d0ba24c.
 * Confirmed running on STM32N6570-DK — see the sibling methodology for visual proof.
 */
#include "lvgl_port_n6.h"
#include "lvgl.h"

/* DWT is provided by CMSIS core headers bundled with the HAL. */
#if defined(STM32N657xx) || defined(STM32N6)
  #include "stm32n6xx.h"
#else
  /* Fallback: user must provide a CMSIS header that defines DWT and __DSB. */
  #include "stm32_hal.h"
#endif

/* ---------- Private state ------------------------------------------------- */

typedef struct {
    bool initialized;
    lvgl_port_n6_cfg_t cfg;
    uint32_t cycles_per_ms;
    lv_display_t *disp;
} lvgl_port_n6_ctx_t;

static lvgl_port_n6_ctx_t s_ctx = {0};

/* Public telemetry */
volatile uint8_t  lvgl_port_n6_state       = 0;
volatile uint32_t lvgl_port_n6_tick_ms     = 0;
volatile uint32_t lvgl_port_n6_flush_count = 0;
volatile uint32_t lvgl_port_n6_loop_count  = 0;

/* ---------- LVGL callbacks ------------------------------------------------ */

static uint32_t lvgl_tick_cb(void)
{
    uint32_t cyc = DWT->CYCCNT;
    uint32_t ms  = (s_ctx.cycles_per_ms != 0U) ? (cyc / s_ctx.cycles_per_ms) : cyc;
    lvgl_port_n6_tick_ms = ms;
    return ms;
}

static void lvgl_flush_cb(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map)
{
    /* DIRECT render mode — buffer IS the framebuffer. Nothing to copy.
     * __DSB drains pending CPU writes so the LTDC master sees the new pixels
     * on its next scanout. If the framebuffer region is CPU-cacheable,
     * callers should add SCB_CleanDCache_by_Addr in their panel init — most
     * N6 setups configure PSRAM as non-cacheable so no clean is needed. */
    (void)area;
    (void)px_map;
    __DSB();
    lvgl_port_n6_flush_count++;
    lv_display_flush_ready(disp);
}

/* ---------- Public API ---------------------------------------------------- */

lvgl_port_n6_status_t lvgl_port_n6_init(const lvgl_port_n6_cfg_t *cfg)
{
    if (cfg == NULL)                 return LVGL_PORT_N6_ERR_PARAM;
    if (cfg->fb_addr == NULL)        return LVGL_PORT_N6_ERR_PARAM;
    if (cfg->fb_width == 0U)         return LVGL_PORT_N6_ERR_PARAM;
    if (cfg->fb_height == 0U)        return LVGL_PORT_N6_ERR_PARAM;
    if (cfg->fb_bytes_per_px != 2U)  return LVGL_PORT_N6_ERR_PARAM;  /* RGB565 only */
    if (cfg->hclk_hz == 0U)          return LVGL_PORT_N6_ERR_PARAM;
    if (s_ctx.initialized)           return LVGL_PORT_N6_ERR_STATE;

    s_ctx.cfg           = *cfg;
    s_ctx.cycles_per_ms = cfg->hclk_hz / 1000U;
    lvgl_port_n6_state  = 0;

    lv_init();
    lvgl_port_n6_state = 10;

    lv_tick_set_cb(lvgl_tick_cb);
    lvgl_port_n6_state = 20;

    lv_display_t *disp = lv_display_create((int32_t)cfg->fb_width, (int32_t)cfg->fb_height);
    if (disp == NULL) {
        lvgl_port_n6_state = 9;
        return LVGL_PORT_N6_ERR_LVGL;
    }
    s_ctx.disp         = disp;
    lvgl_port_n6_state = 30;

    lv_display_set_color_format(disp, LV_COLOR_FORMAT_RGB565);
    lvgl_port_n6_state = 40;

    const uint32_t fb_size = cfg->fb_width * cfg->fb_height * cfg->fb_bytes_per_px;
    lv_display_set_buffers(disp, cfg->fb_addr, NULL, fb_size, LV_DISPLAY_RENDER_MODE_DIRECT);
    lvgl_port_n6_state = 50;

    lv_display_set_flush_cb(disp, lvgl_flush_cb);
    lvgl_port_n6_state = 60;

    if (cfg->build_scene_cb != NULL) {
        cfg->build_scene_cb();
    }
    lvgl_port_n6_state = 70;

    s_ctx.initialized  = true;
    lvgl_port_n6_state = 1;    /* all-done sentinel */
    return LVGL_PORT_N6_OK;
}

void lvgl_port_n6_run_once(void)
{
    if (!s_ctx.initialized) return;

    (void)lv_timer_handler();
    lvgl_port_n6_loop_count++;

    if (lvgl_port_n6_flush_count > 0U && lvgl_port_n6_state == 1U) {
        lvgl_port_n6_state = 2;     /* first-frame-out sentinel */
    }
}

void lvgl_port_n6_deinit(void)
{
    if (!s_ctx.initialized) return;

    if (s_ctx.disp != NULL) {
        lv_display_delete(s_ctx.disp);
        s_ctx.disp = NULL;
    }
    lv_deinit();

    s_ctx.initialized  = false;
    lvgl_port_n6_state = 0;
}
