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
#include "ltdc.h"   /* run-23: double-buffer VBlank swap needs the LTDC handle + layer */

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
    /* DOUBLE-BUFFER DIRECT (ping-pong): LVGL renders the full frame into the INACTIVE buffer; on the
     * last flush we point the LTDC front layer at it (px_map = that buffer's base) and reload at the
     * next VERTICAL BLANKING (SRCR.VBR) — an atomic swap, so the LTDC never DMA-reads a buffer being
     * drawn. This is the documented fix for the LVGL tearing/glitch (methodology-stm32n6-ltdc-display
     * §2.4). __DSB drains CPU writes (FB is MPU non-cacheable) before the swap. */
    /* Clean (write back) the just-rendered pixels from the M55 D-cache to PSRAM BEFORE the LTDC reads
     * them. If the framebuffer's MPU region is anything but *truly* non-cacheable, the rendered rows can
     * sit in the D-cache: the LTDC reads PSRAM by DMA and never sees them, and the NPU's post-inference
     * SCB_InvalidateDCache can DISCARD those still-dirty lines -> the framebuffer corrupts exactly at
     * inference time (methodology-stm32n6-ltdc-display §"clean before reload"). On a genuinely
     * non-cacheable region this is a cheap near-no-op. Clean only this dirty area's row span. */
    const uint32_t stride = 800u * 2u;                          /* RGB565 row bytes (full width) */
    uint8_t       *rows   = px_map + (uint32_t)area->y1 * stride;
    uint32_t       nbytes = (uint32_t)(area->y2 - area->y1 + 1) * stride;
    SCB_CleanDCache_by_Addr((uint32_t *)rows, (int32_t)nbytes);
    __DSB();
    if (lv_display_flush_is_last(disp)) {
        LTDC_Layer1->CFBAR   = (uint32_t)px_map;   /* front layer -> just-completed buffer */
        hltdc.Instance->SRCR = LTDC_SRCR_VBR;      /* swap at the next VBlank (no tear) */
        /* NOTE: do NOT busy-wait for the reload here — a per-frame VBlank spin (~16 ms) starves the
         * UART RX poll in the super-loop (dropped keystrokes). The once-per-inference settle is done
         * explicitly via lvgl_port_n6_wait_idle() in main.c, right before the NPU runs. */
    }
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
    /* Ping-pong double-buffer: second FB 1 MB after the first — both inside the 4 MB MPU
     * non-cacheable PSRAM region (0x90000000-0x903FFFFF). The flush_cb swaps the LTDC between them
     * at VBlank so it never reads a buffer mid-draw. */
    uint8_t *fb1 = (uint8_t *)cfg->fb_addr + 0x100000U;
    lv_display_set_buffers(disp, cfg->fb_addr, fb1, fb_size, LV_DISPLAY_RENDER_MODE_DIRECT);
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

void lvgl_port_n6_wait_idle(void)
{
    /* Bare-metal stand-in for the reference's ThreadX display event-flag: spin until the LTDC has
     * actually performed the shadow-register reload requested in flush_cb (SRCR.VBR). VBR is "set by
     * software, cleared only by hardware after reload" — so once it reads 0 the ping-pong swap is done
     * and the LTDC is stably scanning one complete buffer. Calling this just before NPU inference means
     * no swap is in flight and nothing writes the framebuffer while the NPU saturates the AXI bus,
     * which is what eliminates the inference-time display corruption. Bounded (~2 frames @60Hz) so a
     * stalled LTDC can never hang the super-loop. */
    uint32_t guard = 30000000U;
    while ((hltdc.Instance->SRCR & LTDC_SRCR_VBR) != 0U && guard != 0U) {
        guard--;
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
