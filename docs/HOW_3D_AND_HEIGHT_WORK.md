# How 3D Reconstruction and Height Difference Are Calculated

This document explains how the pipeline builds a 3D height map from the four gradient images and how vertical height (difference between two levels) is computed and converted to mm.

---

## 1. From four gradient images to a depth map (3D reconstruction)

We have **four images** of the same scene under gradient illumination:

- `grad_x_pos`, `grad_x_neg` (gradient in X direction, positive and negative)
- `grad_y_pos`, `grad_y_neg` (gradient in Y direction, positive and negative)

### Step 1: Normalize and form differentials

- Each image is normalized (and optionally divided by the average of all four to reduce shadows).
- **Differential images**:
  - `diff_x = I_x_pos - I_x_neg`
  - `diff_y = I_y_pos - I_y_neg`

Under a **linear shading model**, these differentials are proportional to the **surface slope** in the x and y directions (how much the surface tilts relative to the camera).

### Step 2: Surface normals

From the slopes we get a **normal vector** at each pixel:

- `nx ≈ diff_x`, `ny ≈ diff_y`
- `nz = sqrt(1 - nx² - ny²)` (so the vector has unit length and points “up” out of the surface)

So we get a **normal map**: at each pixel (row, col) we have a 3D vector (nx, ny, nz) describing the local surface orientation.

### Step 3: Slope field (height gradients)

The **height** of the surface at each pixel is some unknown function `z(row, col)`. Its gradients are:

- `p = ∂z/∂x ≈ nx / nz`
- `q = ∂z/∂y ≈ ny / nz`

So we now have a **slope field** (p, q) over the image.

### Step 4: Integration (Frankot–Chellappa)

The slopes (p, q) are **integrated** to recover the height map `z(x,y)`:

- In the **Fourier domain**, integration has a simple form: the height is computed so that its gradients best match (p, q) in a least-squares sense and the result is **integrable** (no conflicting height values along different paths).
- The result is a single **depth map** `depth[row, col]`: each pixel has a value in arbitrary “depth units”. Higher value = surface further from the camera (or higher in the scene, depending on sign convention).

So:

**4 gradient images → differentials → normals → slopes (p,q) → Frankot–Chellappa → depth map**

That depth map **is** the 3D reconstruction: we have one height value per pixel. The “3D” is this 2.5D height field: (col, row, depth).

---

## 2. How height difference between two levels is calculated

Once we have the depth map `depth[row, col]`:

- **Point A:** pixel (r1, c1) → depth value `d1 = depth[r1, c1]` (optionally averaged over a small window for stability).
- **Point B:** pixel (r2, c2) → depth value `d2 = depth[r2, c2]`.

The **vertical height difference** between the two points (in depth units) is:

```
height_diff_units = d2 - d1
```

So we simply **subtract the two depth values**. That is the height of B relative to A in the reconstructed surface. No local plane or extra model: it comes directly from the integrated height map.

### Converting to millimetres (calibration)

The depth map is in arbitrary “units”. We calibrate by measuring a **known height** in the scene:

1. Click **base** and **top** of an object whose height in mm you know (e.g. 2 mm).
2. The script gets `z_base` and `z_top` (median depth in a small window at each click).
3. **Delta in units:** `delta_units = |z_top - z_base|`
4. You enter **known_height_mm** (e.g. 2).
5. **Scale factor:** `mm_per_unit = known_height_mm / delta_units`

Then for **any** two points:

```
height_diff_mm = (d2 - d1) * mm_per_unit
```

So:

- **Height difference in “units”** = `d2 - d1` (from the depth map).
- **Height difference in mm** = `(d2 - d1) * mm_per_unit` (after calibration).

If you set a **base** (e.g. white sheet = 0) in the query tool, depths are first converted to “height above base” (`d - base_reference`), and the same formula applies: height above base in mm = `(d - base_reference) * mm_per_unit`.

---

## 3. Summary

| What | How it’s obtained |
|------|-------------------|
| **3D reconstruction** | 4 gradient images → diff_x, diff_y → normals → slopes p, q → Frankot–Chellappa integration → **depth map** z(row, col). |
| **Height diff (units)** | At two pixels: `d1 = depth[r1,c1]`, `d2 = depth[r2,c2]`. **Height diff = d2 - d1.** |
| **Height diff (mm)** | **Calibrate:** mm_per_unit = known_height_mm / delta_units (from one or more base–top pairs). Then **height_diff_mm = (d2 - d1) * mm_per_unit**. |
| **Lateral distance (mm)** | Calibrate mm_per_pixel (e.g. from a ruler). Then lateral_mm = pixel_distance * mm_per_pixel. |
| **3D distance (mm)** | With both scales: treat (col, row, depth) as (X, Y, Z) in mm using mm_per_pixel and mm_per_unit; then Euclidean distance. |

The **calibrate_scale.py** script lets you add **multiple base–top pairs**; it shows a table of each pair’s depth levels (z_base, z_top, delta) and the inferred mm_per_unit per pair, then combines all pairs into one **mm_per_unit** so you can observe consistency and get a more stable scale.
