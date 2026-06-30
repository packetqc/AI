/**
 * lcd.h — minimal LTDC + panel bring-up for run-23.
 */
#ifndef LCD_DISPLAY_H
#define LCD_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the panel + LTDC layer 0 (RGB565 framebuffer in PSRAM) and show a test pattern.
 * Call after PSRAM is up (post extmem init). Requires LTDC1/LTDC2 RIF master grants. */
void LCD_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* LCD_DISPLAY_H */
