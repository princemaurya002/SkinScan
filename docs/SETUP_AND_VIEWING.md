# Setup and viewing angle

## Why viewing angle matters

The pipeline assumes the camera is looking **roughly perpendicular** at the specimen (like from directly above). In that case:

- The image is close to an **orthographic** view of the surface.
- Recovered “depth” is height in a consistent direction.
- Calibration (mm per unit) is valid across the image.

If the camera is **at an angle** (oblique view):

- **Perspective** changes scale across the image (foreground vs background).
- The link between image gradients and surface slope is no longer the simple one the code assumes.
- Depth and heights can be **distorted** and calibration can **vary across the image**.

So yes: **errors and odd height/depth can be caused by the camera not being (nearly) directly above the object.** The pipeline does not fully model oblique viewing.

---

## Camera can’t be directly above (e.g. blocks the screen)

If you can’t put the camera straight above because it would block the gradient screen or the light source, you have two types of options.

### 1. Change the setup (recommended if you need best accuracy)

- **Mirror:** Put a **mirror** above the specimen at about 45°. Place the **camera to the side** so it looks at the mirror. The camera then “sees” a top-down view of the specimen without being above it, and the screen can stay in front or to the other side.
- **Beam splitter / semi-transparent mirror:** Similar idea: camera and screen arranged so the camera effectively sees the specimen from above through the optics.
- **Screen above, camera to the side:** If the gradient source is a screen *above* the specimen, you can sometimes tilt the screen slightly and keep the camera as perpendicular as possible to the specimen plane to reduce perspective.

These keep the **view** close to “from above” so the pipeline’s assumptions hold.

### 2. Software correction (perspective warp)

If you must keep an oblique camera position, you can **warp the images** so that the white sheet becomes fronto-parallel. The **entire image** is warped (including the specimen), so you can still examine the specimen in the pipeline.

**What to select:** Click the **4 corners of the white sheet** (the flat base). The quad should **enclose the specimen** so that both the sheet and the specimen appear in the warped image. Do *not* select only the specimen—the sheet defines the reference plane; the specimen is then measured on top of it.

**Steps:**

1. Run:
   ```bash
   python correct_perspective.py
   ```
2. When the first gradient image appears, **click the 4 corners of the white sheet** in order (e.g. top-left → top-right → bottom-right → bottom-left). Ensure the specimen lies **inside** this quad.
3. Close the window. The script warps all four gradient images (full scene: sheet + specimen) and saves them to `test_images_warped/`.
4. Run `test_pipeline.py`; it will use `test_images_warped` automatically if present.

After this, the pipeline runs on the **corrected** full scene. You calibrate and measure the specimen in the query tool as usual (set base on the white sheet with **b**, then measure on the specimen).

---

## Summary

| Situation | What to do |
|-----------|------------|
| Camera can be directly above | Use that; no change needed. |
| Camera would block the screen | Prefer **mirror or beam-splitter** so the camera still “sees” a top-down view. |
| Oblique view is unavoidable | Run **correct_perspective.py** (click 4 corners of reference rectangle), then run the pipeline on `test_images_warped`. |

So: **the problem can be due to viewing at an angle; you can either change the setup (mirror, etc.) or partly correct in software with the perspective warp.**
