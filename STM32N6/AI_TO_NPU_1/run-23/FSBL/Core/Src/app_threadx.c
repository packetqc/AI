/**
 * app_threadx.c — run-23 ThreadX application (calculator FSBL).
 *
 * Framework (ported from reference N6_EDGEAI_1): two static threads spawned from a byte pool.
 *
 *   render thread (prio 20) — the SOLE owner of LVGL (LVGL 9.x is single-threaded). Runs ~continuously:
 *       applies any UI change the demo thread published, ticks the heartbeat, and drives one LVGL frame,
 *       then yields. Because it is decoupled from inference, the display + heartbeat stay FLUID.
 *
 *   demo thread (prio 25) — the calculator loop: publish "expr" + "= ?", run the ~1 s NPU inference,
 *       publish the result, sleep ~3 s. The blocking inference stalls ONLY this thread; the render
 *       thread keeps drawing (this is the whole reason for the ThreadX pivot — bare-metal blocked both).
 *
 * Cross-thread rule: the demo thread NEVER calls LVGL. It publishes plain strings (s_ui_expr/s_ui_answer)
 * + a sequence counter; the render thread reads them and does the lv_label_set_text calls.
 */
#include "app_threadx.h"
#include "app_azure_rtos.h"
#include "lvgl.h"            /* lv_mem_monitor — heap telemetry */
#include "lvgl_port_n6.h"
#include "lvgl_scene.h"
#include "stm32n6570_discovery.h"   /* BSP_LED_Toggle */
#include <string.h>

/* ---- shared UI state: demo thread publishes, render thread applies -------------------------------- */
static char s_ui_expr[48]   = "-";
static char s_ui_answer[40]  = "-";
static volatile uint32_t s_ui_seq     = 0;   /* bumped on any publish */
static uint32_t          s_ui_applied = 0;   /* last seq the render thread drew */

void run23_ui_set_prompt(const char *expr)
{
    strncpy(s_ui_expr, expr, sizeof s_ui_expr - 1); s_ui_expr[sizeof s_ui_expr - 1] = '\0';
    strcpy(s_ui_answer, "= ?");                 /* new expression → answer pending */
    s_ui_seq++;
}
void run23_ui_set_answer(const char *ans)
{
    strncpy(s_ui_answer, ans, sizeof s_ui_answer - 1); s_ui_answer[sizeof s_ui_answer - 1] = '\0';
    s_ui_seq++;
}

/* ---- threads ------------------------------------------------------------------------------------- */
#define RENDER_STACK_BYTES  (32U * 1024U)   /* LVGL lv_timer_handler is stack-heavy (bare-metal used the ~50 KB main stack) */
#define DEMO_STACK_BYTES    (16U * 1024U)
#define RENDER_PRIO         20U
#define DEMO_PRIO           25U

static TX_THREAD s_render_thread;
static TX_THREAD s_demo_thread;

extern volatile int           g_lvgl_ok;
extern volatile unsigned long g_heartbeat;

/* SWD-readable LVGL heap telemetry (leak hunt). If g_lv_free_min trends DOWN over minutes, the heap is
 * leaking → after a while glyph-draw allocs fail → text stops rendering while shapes still draw (the
 * observed "blank after ~13 min, firmware alive"). Read these live via read_memory. */
volatile uint32_t g_lv_free_now = 0;   /* current free bytes */
volatile uint32_t g_lv_free_min = 0xFFFFFFFFU; /* lowest free ever seen */
volatile uint8_t  g_lv_used_pct = 0;
volatile uint8_t  g_lv_frag_pct = 0;

/* Provided by main.c — wraps Grammar_Calc + the network/IO buffers (kept in main's scope). */
extern void run23_infer(const char *expr, char *out, int out_sz);

static void render_thread_entry(ULONG arg)
{
    (void)arg;
    unsigned long frame = 0;
    ULONG hb_last = tx_time_get();
    for (;;) {
        if (g_lvgl_ok) {
            if (s_ui_seq != s_ui_applied) {          /* apply pending publish — LVGL calls ONLY here */
                s_ui_applied = s_ui_seq;
                lvgl_scene_set_prompt(s_ui_expr);    /* sets prompt (+ "= ?") */
                lvgl_scene_set_answer(s_ui_answer);  /* overwrite answer with the published string */
            }
            /* Heartbeat on the ThreadX tick (TX_TIMER_TICKS_PER_SECOND=100 → 50 ticks = 500 ms), NOT the
             * frame count: the render thread runs ~100 FPS so frame-gating made the LED ~3x too fast. */
            ULONG now = tx_time_get();
            if (now - hb_last >= 50UL) {
                hb_last = now;
                g_heartbeat++;
                BSP_LED_Toggle(LED_GREEN);
                /* Sample the LVGL heap every 500 ms (this thread is the LVGL owner). */
                lv_mem_monitor_t mon;
                lv_mem_monitor(&mon);
                g_lv_free_now = (uint32_t)mon.free_size;
                if (g_lv_free_now < g_lv_free_min) g_lv_free_min = g_lv_free_now;
                g_lv_used_pct = (uint8_t)mon.used_pct;
                g_lv_frag_pct = (uint8_t)mon.frag_pct;
            }
            lvgl_scene_tick(frame, g_heartbeat);
            lvgl_port_n6_run_once();
        }
        frame++;
        tx_thread_sleep(3);                          /* ~30 FPS (reference LVGL_THREAD_SLEEP_TICKS=3). 100 FPS
                                                      * (sleep 1) out-runs the 60 Hz LTDC swap → the distortion. */
    }
}

static void demo_thread_entry(ULONG arg)
{
    (void)arg;
    /* Grammar supports + - * / , parentheses, multi-digit integers, proper precedence
     * (grammar_runner.cpp fallback_playbook: expr/term/factor). Divisions kept exact and
     * results non-negative for a clean, readable demo. Array count is auto-computed below. */
    static const char *const demo[] = {
        "3 + 4", "12 - 5", "6 * 7", "8 / 2", "9 * 9", "2 * (3 + 4)",
        "10 + 15", "7 * 8", "20 - 6", "18 / 3", "5 + 6 * 2", "(8 - 3) * 4",
        "100 - 45", "11 * 11", "9 + 8 + 7", "(2 + 3) * (4 + 1)", "48 / 6", "25 + 25",
    };
    const int n = (int)(sizeof demo / sizeof demo[0]);
    int i = 0;
    for (;;) {
        run23_ui_set_prompt(demo[i]);                /* publish new expr + "= ?" */
        char ans[40];
        run23_infer(demo[i], ans, (int)sizeof ans);  /* ~1 s NPU — stalls this thread only */
        run23_ui_set_answer(ans);                    /* publish the result */
        i = (i + 1) % n;
        tx_thread_sleep(300);                        /* ~3 s cadence @ 100 Hz tick */
    }
}

UINT App_ThreadX_Init(VOID *memory_ptr)
{
    TX_BYTE_POOL *pool = (TX_BYTE_POOL *)memory_ptr;
    CHAR *stack;
    UINT r;

    r = tx_byte_allocate(pool, (VOID **)&stack, RENDER_STACK_BYTES, TX_NO_WAIT);
    if (r != TX_SUCCESS) return r;
    r = tx_thread_create(&s_render_thread, "render", render_thread_entry, 0,
                         stack, RENDER_STACK_BYTES, RENDER_PRIO, RENDER_PRIO,
                         TX_NO_TIME_SLICE, TX_AUTO_START);
    if (r != TX_SUCCESS) return r;

    r = tx_byte_allocate(pool, (VOID **)&stack, DEMO_STACK_BYTES, TX_NO_WAIT);
    if (r != TX_SUCCESS) return r;
    r = tx_thread_create(&s_demo_thread, "demo", demo_thread_entry, 0,
                         stack, DEMO_STACK_BYTES, DEMO_PRIO, DEMO_PRIO,
                         TX_NO_TIME_SLICE, TX_AUTO_START);
    return r;
}

void MX_ThreadX_Init(void)
{
    tx_kernel_enter();   /* never returns — scheduler owns the CPU */
}
