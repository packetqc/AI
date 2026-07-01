/**
 * lvgl_scene.c — run-23 application scene (L1 frame + L2 content).
 *
 * Layout mirrors the N6_EDGEAI_1 reference design:
 *   L1 framework — top app-tab bar; bottom global bar (SW-LED on the left, "LVGL x.y.z" on the right)
 *   L2 content   — inference request (top) / inference response (bottom)
 *
 * Display↔NPU coordination lives in main.c: each inference is show-prompt -> lvgl_port_n6_wait_idle()
 * (settle the framebuffer swap) -> run the NPU with NO refresh -> show-answer. That bare-metal gate is
 * the stand-in for the reference's ThreadX display event-flag, and is what stops inference-time
 * corruption. Compiled with -mno-unaligned-access (LVGL Makefile rule) — mandatory on Cortex-M55 + GCC.
 */
#include "lvgl.h"
#include "lvgl_scene.h"

#define COL_BAR     0x16324A   /* L1 top/bottom bars   */
#define COL_BG      0x0B1E2D   /* L2 content background */
#define COL_TAB     0x9FE0FF   /* active app tab        */
#define COL_PROMPT  0xFFD27A   /* inference request     */
#define COL_ANSWER  0x7CFFB0   /* inference response    */
#define COL_DIM     0x6E8AA0   /* muted captions        */
#define COL_LED_ON  0x16FF6E   /* SW-LED bright         */
#define COL_LED_OFF 0x0A6E33   /* SW-LED dim            */

static lv_obj_t      *s_prompt;
static lv_obj_t      *s_answer;
static lv_obj_t      *s_led;
static unsigned long  s_last_hb = ~0UL;

static lv_obj_t *make_bar(lv_obj_t *scr, lv_align_t side)
{
    lv_obj_t *b = lv_obj_create(scr);
    lv_obj_remove_style_all(b);
    lv_obj_set_size(b, 800, 56);   /* taller bars to fit the doubled (~28px) bar text */
    lv_obj_align(b, side, 0, 0);
    lv_obj_set_style_bg_color(b, lv_color_hex(COL_BAR), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(b, LV_OPA_COVER, LV_PART_MAIN);
    return b;
}

static lv_obj_t *make_caption(lv_obj_t *scr, const char *txt, int y)
{
    lv_obj_t *c = lv_label_create(scr);
    lv_label_set_text(c, txt);
    lv_obj_set_style_text_color(c, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_x(c, 0, LV_PART_MAIN);         /* left pivot: grow rightward, left edge fixed at x=24 */
    lv_obj_set_style_transform_pivot_y(c, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(c, 512, LV_PART_MAIN);         /* 512/256 = 2x -> ~28px */
    lv_obj_align(c, LV_ALIGN_TOP_LEFT, 24, y);
    return c;
}

void lvgl_scene_build(void)
{
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_hex(COL_BG), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, LV_PART_MAIN);

    /* ---- L1 top: app-tab bar ---- */
    lv_obj_t *top = make_bar(scr, LV_ALIGN_TOP_MID);
    lv_obj_t *tab = lv_label_create(top);
    lv_label_set_text(tab, "calculator");
    lv_obj_set_style_text_color(tab, lv_color_hex(COL_TAB), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_x(tab, 0, LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_y(tab, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(tab, 512, LV_PART_MAIN);       /* 2x -> ~28px */
    lv_obj_align(tab, LV_ALIGN_LEFT_MID, 16, 0);
    lv_obj_t *appn = lv_label_create(top);
    lv_label_set_text(appn, "STM32N6  nocode LM");   /* shortened so it fits the top bar at 2x */
    lv_obj_set_style_text_color(appn, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_x(appn, lv_pct(100), LV_PART_MAIN);   /* right pivot: grow leftward */
    lv_obj_set_style_transform_pivot_y(appn, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(appn, 512, LV_PART_MAIN);       /* 2x -> ~28px */
    lv_obj_align(appn, LV_ALIGN_RIGHT_MID, -16, 0);

    /* ---- L2 content: inference (top) / response (bottom) ---- */
    make_caption(scr, "inference", 64);
    s_prompt = lv_label_create(scr);
    lv_label_set_text(s_prompt, "-");
    lv_obj_set_style_text_color(s_prompt, lv_color_hex(COL_PROMPT), LV_PART_MAIN);
    lv_obj_set_style_text_align(s_prompt, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    /* Enlarge the 14px default font ~2.6x via transform (reference technique — no bigger font, no ROM).
     * Pivot at the label centre so the scale is symmetric and the text stays centred as it grows. */
    lv_obj_set_style_transform_pivot_x(s_prompt, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_y(s_prompt, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(s_prompt, 1097, LV_PART_MAIN);   /* 1097/256 = 4.29x -> ~60px */
    lv_obj_align(s_prompt, LV_ALIGN_CENTER, 0, -100);   /* centred (H+V) in the upper inference zone */

    make_caption(scr, "response", 244);
    s_answer = lv_label_create(scr);
    lv_label_set_text(s_answer, "-");
    lv_obj_set_style_text_color(s_answer, lv_color_hex(COL_ANSWER), LV_PART_MAIN);
    lv_obj_set_style_text_align(s_answer, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    /* The response is the headline — scale it bigger, ~3.3x → ~46px effective. Centre pivot too. */
    lv_obj_set_style_transform_pivot_x(s_answer, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_y(s_answer, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(s_answer, 1097, LV_PART_MAIN);   /* 1097/256 = 4.29x -> ~60px */
    lv_obj_align(s_answer, LV_ALIGN_CENTER, 0, 100);    /* centred (H+V) in the lower response zone */

    /* ---- L1 bottom: global bar — SW-LED (left), LVGL version (right) ---- */
    lv_obj_t *bot = make_bar(scr, LV_ALIGN_BOTTOM_MID);
    s_led = lv_obj_create(bot);
    lv_obj_remove_style_all(s_led);
    lv_obj_set_size(s_led, 16, 16);
    lv_obj_set_style_radius(s_led, LV_RADIUS_CIRCLE, LV_PART_MAIN);
    lv_obj_set_style_bg_color(s_led, lv_color_hex(COL_LED_ON), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(s_led, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_align(s_led, LV_ALIGN_LEFT_MID, 14, 0);
    lv_obj_t *ledtx = lv_label_create(bot);
    lv_label_set_text(ledtx, "SW");
    lv_obj_set_style_text_color(ledtx, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_x(ledtx, 0, LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_y(ledtx, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(ledtx, 512, LV_PART_MAIN);      /* 2x */
    lv_obj_align(ledtx, LV_ALIGN_LEFT_MID, 44, 0);

    lv_obj_t *ver = lv_label_create(bot);
    char vb[24];
    lv_snprintf(vb, sizeof vb, "LVGL %d.%d.%d",
                LVGL_VERSION_MAJOR, LVGL_VERSION_MINOR, LVGL_VERSION_PATCH);
    lv_label_set_text(ver, vb);
    lv_obj_set_style_text_color(ver, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_transform_pivot_x(ver, lv_pct(100), LV_PART_MAIN);   /* right pivot: grow leftward, right edge fixed */
    lv_obj_set_style_transform_pivot_y(ver, lv_pct(50), LV_PART_MAIN);
    lv_obj_set_style_transform_scale(ver, 512, LV_PART_MAIN);       /* 2x */
    lv_obj_align(ver, LV_ALIGN_RIGHT_MID, -16, 0);
}

void lvgl_scene_set_prompt(const char *s)
{
    /* Full-screen invalidate = force a complete repaint on the next run_once. With a single buffer and
     * dirty-only rendering, a region the NPU may have disturbed during inference would otherwise never
     * be repainted (the stretch "doesn't come back"). Repainting the whole scene around each inference
     * guarantees recovery. Called between inferences (no NPU running), so no contention. */
    if (s_prompt != NULL) { lv_label_set_text(s_prompt, s); lv_obj_invalidate(lv_screen_active()); }
}

void lvgl_scene_set_answer(const char *s)
{
    if (s_answer != NULL) { lv_label_set_text(s_answer, s); lv_obj_invalidate(lv_screen_active()); }
}

void lvgl_scene_tick(unsigned long frame, unsigned long hb)
{
    (void)frame;

    /* SW-LED mirrors the physical heartbeat LED: bright on even ticks, dim on odd. Only the LED and the
     * status label change per tick; LVGL redraws just those small dirty regions into the single
     * framebuffer and leaves the rest untouched. No full-screen invalidate — that existed only to keep
     * two ping-pong buffers in sync; with one buffer it would rewrite all 768 KB every frame (maximum
     * tearing) for no benefit. */
    if (s_led != NULL && hb != s_last_hb) {
        s_last_hb = hb;
        lv_obj_set_style_bg_color(s_led,
            lv_color_hex((hb & 1UL) ? COL_LED_OFF : COL_LED_ON), LV_PART_MAIN);
    }
}
