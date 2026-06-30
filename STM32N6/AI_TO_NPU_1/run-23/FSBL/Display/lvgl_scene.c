/**
 * lvgl_scene.c — C2 test scene for the run-23 LVGL bring-up.
 *
 * Proves LVGL renders into the PSRAM framebuffer (DIRECT mode) that the LTDC scans: a dark
 * background + a couple of widgets + a live status label. Built once by lvgl_port_n6_init via the
 * build_scene_cb; updated each frame by lvgl_scene_tick() (called from lvgl_port_n6_run_once path).
 *
 * Compiled with -mno-unaligned-access (LVGL Makefile rule) — mandatory on Cortex-M55 + GCC.
 */
#include "lvgl.h"
#include "lvgl_scene.h"

static lv_obj_t      *s_status;
static unsigned long  s_last_hb = ~0UL;

void lvgl_scene_build(void)
{
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x0B1E2D), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);

    /* Title */
    lv_obj_t *title = lv_label_create(scr);
    lv_label_set_text(title, "STM32N6  -  nocode grammar LM  -  LVGL");
    lv_obj_set_style_text_color(title, lv_color_white(), LV_PART_MAIN);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 24);

    /* Three colour swatches — prove widget creation + layout + styling work. */
    static const uint32_t cols[3] = { 0xE03C3C, 0x3CE06E, 0x3C7CE0 };
    for (int i = 0; i < 3; i++) {
        lv_obj_t *box = lv_obj_create(scr);
        lv_obj_remove_style_all(box);
        lv_obj_set_size(box, 180, 140);
        lv_obj_set_pos(box, 70 + i * 230, 130);
        lv_obj_set_style_bg_color(box, lv_color_hex(cols[i]), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(box, LV_OPA_COVER, LV_PART_MAIN);
        lv_obj_set_style_radius(box, 12, LV_PART_MAIN);
        lv_obj_set_style_border_color(box, lv_color_white(), LV_PART_MAIN);
        lv_obj_set_style_border_width(box, 3, LV_PART_MAIN);
    }

    /* Live status label (updated each frame) — proof of the running event loop on-panel. */
    s_status = lv_label_create(scr);
    lv_label_set_text(s_status, "boot");
    lv_obj_set_style_text_color(s_status, lv_color_hex(0x9FE0FF), LV_PART_MAIN);
    lv_obj_align(s_status, LV_ALIGN_BOTTOM_MID, 0, -16);
}

void lvgl_scene_tick(unsigned long frame, unsigned long hb)
{
    if (s_status == NULL) return;

    /* Update the status text only when the heartbeat ticks (~2 Hz) — avoids re-laying-out the label
     * every super-loop iteration. */
    if (hb != s_last_hb) {
        s_last_hb = hb;
        char buf[64];
        lv_snprintf(buf, sizeof(buf), "frame %lu   heartbeat %lu", frame, hb);
        lv_label_set_text(s_status, buf);
    }

    /* DUAL-BUFFER strobe fix (reference lvgl_port.c): a partial invalidate lands in only ONE of the
     * two ping-pong buffers, so the LTDC alternation makes slow-update widgets strobe/degrade. Force
     * a full-screen redraw every frame so the COMPLETE scene is painted into BOTH buffers as LVGL
     * alternates them. Called from the super-loop (NOT flush_cb — LVGL forbids invalidate there). */
    lv_obj_invalidate(lv_screen_active());
}
