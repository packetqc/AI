/**
 * lvgl_scene.h — C2 test scene for the run-23 LVGL bring-up.
 */
#ifndef LVGL_SCENE_H
#define LVGL_SCENE_H

#ifdef __cplusplus
extern "C" {
#endif

void lvgl_scene_build(void);                              /* build_scene_cb for lvgl_port_n6_init */
void lvgl_scene_tick(unsigned long frame, unsigned long hb);  /* per-frame: SW-LED blink + full repaint */
void lvgl_scene_set_prompt(const char *s);                /* L2 content top: the inference request  */
void lvgl_scene_set_answer(const char *s);                /* L2 content bottom: the inference response */

#ifdef __cplusplus
}
#endif

#endif /* LVGL_SCENE_H */
