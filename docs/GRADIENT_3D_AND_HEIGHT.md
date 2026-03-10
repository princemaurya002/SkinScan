# 3D Reconstruction and Height from 4-Gradient Images

This note summarizes how **distance between two points at different heights** and **point height** are derived from the 4-gradient (photometric-style) pipeline, and what the literature recommends when the current method is not working well.

---

## 1. What the pipeline does

1. **Capture**: Four images under gradient illumination (e.g. grad_x_pos, grad_x_neg, grad_y_pos, grad_y_neg).
2. **Normal map**: From intensity ratios/differences we get per-pixel surface normals (nx, ny, nz).  
   - Differential images: `diff_x = I_x+ - I_x-`, `diff_y = I_y+ - I_y-`.  
   - Under a linear shading model, these relate to gradient components; normals are derived and then slope fields `p = ∂z/∂x`, `q = ∂z/∂y` from `n = (nx, ny, nz)` as `p = nx/nz`, `q = ny/nz`.
3. **Depth (height map)**: The slope field (p, q) is **integrated** to get a height/depth map `z(x,y)`.  
   - **Frankot–Chellappa** (Fourier-domain integration) is used to enforce integrability and get a single height surface.
4. **Calibration**: Using a known-height object, we get `mm_per_unit` (depth units → mm) and optionally `mm_per_pixel` (lateral scale).

So we effectively **reconstruct a 3D surface** (height map) from the four gradients; height of a point = depth at that pixel (optionally relative to a "base" plane).

---

## 2. Why it may not work well

- **Non-integrability**: Estimated (p, q) often have non-zero curl (noise, shadows, non-Lambertian effects). Frankot–Chellappa projects onto an integrable field in Fourier space and **spreads error globally**, which can blur or shift height.
- **Shadows / low signal**: In shadow or very dark regions, ratios are unstable → wrong normals → wrong gradients → wrong height.
- **Scale and bias**: Depth is only defined up to an arbitrary offset (and global scale until calibrated). So we need a **reference (e.g. white sheet = 0)** and **calibration (mm_per_unit, mm_per_pixel)** for absolute heights and distances.

---

## 3. Proper algorithms from the literature

### 3.1 Height of a point

- **From the reconstructed depth map**: Height at pixel (r, c) = `depth[r, c]` in "units".  
- **Relative to base**: If a base (e.g. white sheet) is set at depth `d_base`, then **height above base** = `depth[r,c] - d_base` (in units); in mm: `(depth[r,c] - d_base) * mm_per_unit`.  
- So "height of a point from the image" is exactly this: use the integrated depth map and (optionally) subtract the base and convert to mm.

### 3.2 Distance between two points at different levels

Once we have a depth map and calibration:

- Treat each pixel as a 3D point in a **right-handed image-plane coordinate system**:  
  - **X** = column index (horizontal), **Y** = row index (vertical), **Z** = depth (height) from the map.  
- In **mm** (with calibration):  
  - `X_mm = col * mm_per_pixel`  
  - `Y_mm = row * mm_per_pixel`  
  - `Z_mm = depth[row,col] * mm_per_unit` (or `(depth - base) * mm_per_unit` if using a base plane)

Then the **3D Euclidean distance** between point A and B is:

```
distance_mm = sqrt((X2-X1)² + (Y2-Y1)² + (Z2-Z1)²)
```

This gives the true distance between two points at different heights (e.g. base of lesion to top, or two points on a 3D surface). The query tool reports this when both `mm_per_unit` and `mm_per_pixel` are set.

### 3.3 Improving the depth map (when current method is not working properly)

- **Shadow / outlier handling**:  
  - Mask or downweight low-intensity (shadow) pixels before forming differentials.  
  - Use a **confidence/weight map** (e.g. based on mean intensity or variance) and feed it into a **weighted** integration (see below).

- **Better integration (instead of plain Frankot–Chellappa)**  
  - **Poisson integration**: Solve ∇²z = ∂p/∂x + ∂q/∂y with Dirichlet/Neumann boundary conditions. Same idea as F–C (minimize deviation from gradients) but in spatial domain; can incorporate **weights** (e.g. zero weight in shadow).  
  - **Multi-scale integration**: Integrate at coarse scale first, then refine at finer scales to reduce error propagation (see "A robust multi-scale integration method to obtain the depth from gradient maps", CVPR etc.).  
  - **Curl-based correction**: Use **integrability (curl)** to find bad gradients and correct them before integrating (e.g. "An Algebraic Approach to Surface Reconstruction from Gradient Fields", Agrawal et al., ICCV 2005). This can **confine errors locally** instead of smoothing globally.

- **4-source photometric stereo**: With 4 gradient images we have an over-determined system; use robust estimation (e.g. median or RANSAC over the 4 intensities) to reject highlights/shadows and get more stable normals, then integrate.

- **Regularization**: For industrial/photometric stereo, **Tikhonov regularization** in the integration step can stabilize depth in the presence of noise and non-Lambertian effects.

---

## 4. Summary

| Goal | Method |
|------|--------|
| **Height of a point** | Read `depth[r,c]`; subtract base if used; convert to mm with `mm_per_unit`. |
| **Distance between two points (any levels)** | Convert (col, row, depth) to 3D mm with `mm_per_pixel` and `mm_per_unit`, then Euclidean distance. |
| **More reliable depth** | Shadow masking, weighted or Poisson integration, curl-based gradient correction, multi-scale integration, robust normal estimation from 4 images. |

The pipeline already reconstructs 3D (a height field) from the 4 gradients; the main improvements when depth is unreliable are: (1) robust normal estimation (shadow/outlier handling), (2) better integration (weighted/Poisson or curl-based), and (3) calibration + base reference for correct absolute heights and 3D distances in mm.

---

## 5. What this codebase does

- **`test_pipeline.py`**: Builds normals from the 4 gradient images (with optional shadow reduction and smoothing), then integrates via **Frankot–Chellappa** to get `depth_raw.npy`. Tuning: `reduce_shadow`, `shadow_percentile`, `smooth_diffs`.
- **`calibrate_scale.py`**: Sets `mm_per_unit` (and optionally `mm_per_pixel`) so depths and distances can be reported in mm.
- **`query_depth_point.py`**: Single-point height (and height above base), two-point **vertical height** and **lateral distance**, and when calibrated **3D Euclidean distance** between the two points in mm. Base (e.g. white sheet) can be set with **b** so all heights are relative to it.
