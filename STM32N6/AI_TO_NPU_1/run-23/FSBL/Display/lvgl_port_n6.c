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

/* Pending FB address handed from lvgl_flush_cb (super-loop) to the LTDC line-event ISR, which swaps
 * CFBAR at the next vblank. 0 = nothing pending. Atomic 32-bit r/w on Cortex-M55. (Reference pattern.) */
volatile uint32_t s_ltdc_pending_fb = 0;

/* SWD-readable LTDC state trace ring — catch the transient inference glitch ON THE FLY without halting.
 * Every vblank (line-event ISR) we snapshot {CFBAR, ISR, LayerCR, pending_fb}. After a glitch, read
 * g_ltdc_trace_idx (newest = (idx-1)&63) and g_ltdc_trace[] via SWD to see the degradation history. */
volatile uint32_t g_ltdc_trace[64][4];
volatile uint32_t g_ltdc_trace_idx = 0;

static void lvgl_flush_cb(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map)
{
    /* DOUBLE-BUFFER DIRECT: LVGL rendered a full frame into the INACTIVE buffer (px_map). Hand its base
     * to the LTDC line-event ISR to swap CFBAR at the next vblank — the LTDC never DMA-reads a buffer
     * being drawn, and the CPU never writes the buffer the LTDC is scanning. That removes the
     * render-vs-LTDC-read contention (the "full-screen refresh conflict"): the CPU always writes the
     * BACK buffer while the LTDC reads the FRONT. __DSB drains the pixel writes to the MPU-non-cacheable
     * PSRAM before the swap. */
    (void)area;
    if (lv_display_flush_is_last(disp)) {
        __DSB();
        s_ltdc_pending_fb = (uint32_t)px_map;
    }
    lvgl_port_n6_flush_count++;
    lv_display_flush_ready(disp);
}

/* LTDC line-event callback — fires at the start of vblank (line fb_height+1). Applies the pending swap
 * via a VBlank reload (atomic, tear-free), then re-arms the line event. HAL calls this from
 * HAL_LTDC_IRQHandler on the line interrupt. */
void HAL_LTDC_LineEventCallback(LTDC_HandleTypeDef *h)
{
    (void)h;
    /* On-the-fly trace: snapshot the live LTDC swap state each vblank (see g_ltdc_trace above). */
    uint32_t ti = g_ltdc_trace_idx & 63U;
    g_ltdc_trace[ti][0] = LTDC_Layer1->CFBAR;
    g_ltdc_trace[ti][1] = hltdc.Instance->ISR;
    g_ltdc_trace[ti][2] = LTDC_Layer1->CR;
    g_ltdc_trace[ti][3] = s_ltdc_pending_fb;
    g_ltdc_trace_idx++;

    uint32_t pending = s_ltdc_pending_fb;
    if (pending != 0U) {
        LTDC_Layer1->CFBAR   = pending;
        hltdc.Instance->SRCR = LTDC_SRCR_VBR;
        s_ltdc_pending_fb    = 0U;
    }
    HAL_LTDC_ProgramLineEvent(&hltdc, s_ctx.cfg.fb_height + 1U);
}

/* IRQ handler for the LTDC line interrupt (LTDC_UP_IRQn = 193). Only the LINE interrupt is enabled
 * (NOT the LTDC_UP_ERR error IRQ that hung bring-up). Dispatches to HAL, which invokes the callback. */
void LTDC_UP_IRQHandler(void)
{
    /* Warm-reload guard. hltdc lives in .bss; a GDB-driven SRAM reload (load_and_run) restarts the CPU
     * with PRIMASK cleared while the LTDC peripheral + its NVIC line are LEFT ENABLED from the prior run.
     * The pending line IRQ then fires during early startup — AFTER .bss zeroes hltdc but BEFORE
     * MX_LTDC_Init repopulates hltdc.Instance — so HAL_LTDC_IRQHandler would deref NULL->ISR2 (BFAR=0x68
     * HardFault). If the handle isn't set up yet, mask this IRQ at the NVIC and bail; MX_LTDC_Init
     * force-resets the LTDC and lvgl_port_n6_init re-enables the line IRQ cleanly after setup. On a cold
     * NOR boot the NVIC starts disabled, so this never triggers there — it only hardens SRAM reloads. */
    if (hltdc.Instance == NULL) {
        HAL_NVIC_DisableIRQ(LTDC_UP_IRQn);
        HAL_NVIC_ClearPendingIRQ(LTDC_UP_IRQn);
        return;
    }
    HAL_LTDC_IRQHandler(&hltdc);
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
    /* DOUBLE buffer: front = fb_addr (LTDC scans), back = fb_back_addr (LVGL draws). If fb_back_addr is
     * NULL, default to fb_addr + 1 MB (legacy contiguous layout). The two buffers MAY live in separate
     * memory banks — this is what lets the 100%-SRAM reference layout put front in AXISRAM1 and back in
     * AXISRAM5-6 (no single bank holds 1.5 MB). LVGL renders the back, the line-event ISR swaps CFBAR at
     * vblank — the CPU write and the LTDC read never touch the same buffer, so no render-vs-read
     * contention. __DSB in flush_cb drains pixel writes to the (MPU-non-cacheable) FB before the swap. */
    uint8_t *fb1 = (cfg->fb_back_addr != NULL) ? (uint8_t *)cfg->fb_back_addr
                                               : (uint8_t *)cfg->fb_addr + 0x100000U;
    /* FULL render mode (not DIRECT): on any change LVGL renders the ENTIRE screen into the inactive
     * buffer, then the line-event ISR swaps. This is the correct model for run-23's mostly-STATIC scene:
     * every swap shows a COMPLETE frame, so the two buffers never diverge -> no intermittent blank (the
     * DIRECT-mode failure: a small dirty region lands in one buffer while the rest of that buffer is
     * many seconds stale, so a swap exposes a big out-of-date region as a blank). The reference keeps
     * DIRECT + tolerates a strobe only because its busy scene repaints most of the screen every frame;
     * for a static scene a full repaint is cheap and identical frame-to-frame, so it's glitch-free.
     * This matches flush_cb's own assumption ("LVGL rendered a full frame into the inactive buffer"). */
    lv_display_set_buffers(disp, cfg->fb_addr, fb1, fb_size, LV_DISPLAY_RENDER_MODE_FULL);
    lvgl_port_n6_state = 50;

    lv_display_set_flush_cb(disp, lvgl_flush_cb);
    lvgl_port_n6_state = 60;

    /* Arm the LTDC line-event at vblank + enable ONLY the line IRQ (LTDC_UP_IRQn) — NOT the error IRQ
     * (LTDC_UP_ERR) that had no handler and hung early bring-up. HAL_LTDC_ProgramLineEvent sets LIPCR +
     * the line-interrupt enable; the error interrupts stay masked. The ISR (LTDC_UP_IRQHandler ->
     * HAL_LTDC_IRQHandler -> HAL_LTDC_LineEventCallback) does the tear-free double-buffer swap. */
    HAL_LTDC_ProgramLineEvent(&hltdc, cfg->fb_height + 1U);
    HAL_NVIC_SetPriority(LTDC_UP_IRQn, 0x0FU, 0U);   /* lowest preempt priority — display not time-critical */
    HAL_NVIC_EnableIRQ(LTDC_UP_IRQn);

    if (cfg->build_scene_cb != NULL) {
        cfg->build_scene_cb();
    }
    lvgl_port_n6_state = 70;

    s_ctx.initialized  = true;
    lvgl_port_n6_state = 1;    /* all-done sentinel */
    return LVGL_PORT_N6_OK;
}

/* Dual-buffer DIRECT propagation (ported from the reference N6_EDGEAI_1 lvgl_port.c). A widget update
 * dirties only ONE of the two alternating DIRECT buffers; the LTDC then STROBES the change on/off as it
 * swaps between the updated buffer and the still-stale one (seen as a ~1 s strobe / flicker on the
 * heartbeat + answer). To land an update in BOTH buffers, a widget change requests a full-screen
 * invalidate for the NEXT 2 render passes — NOT every frame (that is the "huge white slow refresh" /
 * big blank that full-invalidate-per-update produced), only the 2 passes needed to propagate. The
 * invalidate MUST happen here in the render pass, never inside flush_cb (LVGL asserts during render). */
static volatile uint8_t s_full_redraw_frames = 0;

void lvgl_port_n6_mark_dirty(void) { s_full_redraw_frames = 2U; }

void lvgl_port_n6_run_once(void)
{
    if (!s_ctx.initialized) return;

    if (s_full_redraw_frames > 0U) {
        lv_obj_invalidate(lv_screen_active());   /* propagate the pending update into this buffer too */
        s_full_redraw_frames--;
    }

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
