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
    /* SINGLE-BUFFER DIRECT: LVGL renders straight into the ONE framebuffer the LTDC scans (fb_addr,
     * programmed as LTDC_Layer1->CFBAR in LCD_Init). There is NO ping-pong buffer, so there is NO swap
     * to time and NO swap-race — the blank/"losing layout" came from the flush requesting a swap the
     * LTDC hadn't completed, letting LVGL draw into the buffer being scanned. The reference does
     * tear-free double-buffering from the LTDC LINE-EVENT ISR; run-23 has LTDC IRQs disabled (the
     * error-IRQ hang fix), so we take the simpler robust route: one buffer, no swap. __DSB drains the
     * CPU pixel writes to the (MPU non-cacheable) PSRAM framebuffer before LVGL marks the flush done,
     * so the LTDC DMA sees fresh pixels. Only the small regions that change each frame can tear. */
    (void)px_map;
    (void)area;
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
    /* SINGLE buffer (second arg NULL): LVGL renders directly into the LTDC's framebuffer. No second
     * buffer, no swap — the flush_cb is a plain __DSB + flush_ready. This trades a bit of tearing on
     * changed regions for a display that CANNOT go blank from a mistimed ping-pong swap. */
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

void lvgl_port_n6_display_freeze(int freeze)
{
    /* Remove the LTDC-vs-Neural-ART AXI contention window (root cause per AN4861/RM0486, confirmed by
     * ST: LTDC scanout and the NPU share the AXI/memory system; while the NPU floods the bus the LTDC's
     * PSRAM fetch is disturbed and the LIVE scanout corrupts even though the framebuffer bytes stay
     * correct — QoS is already maxed, so the only lever is to stop the LTDC fetching during inference).
     * Disabling Layer1 makes the LTDC stop fetching the framebuffer, so the NPU gets the bus alone for
     * the ~ms it runs; re-enabling restores the scene. IMMEDIATE reload (IMR) so it lands inside the
     * short inference window (a VBlank reload would take effect a whole frame later). */
    if (freeze) {
        hltdc.Instance->BCCR = 0x000B1E2DU;            /* LTDC background = scene bg (COL_BG 0x0B1E2D) so the layer-off shows the bg tint, not black */
        LTDC_Layer1->CR &= ~(uint32_t)LTDC_LxCR_LEN;   /* layer off -> LTDC stops FB fetch */
    } else {
        LTDC_Layer1->CR |=  (uint32_t)LTDC_LxCR_LEN;   /* layer on -> resume scanout */
    }
    hltdc.Instance->SRCR = LTDC_SRCR_IMR;
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
