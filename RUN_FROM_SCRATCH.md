# SkinScan – Commands to Run From Scratch

Run these in order from the project root (where `test_pipeline.py` lives).

---

## 1. Environment setup (once)

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Optional: for RANSAC plane fitting in query_depth_point.py
pip install scikit-learn
```

---

## 2. Put your gradient images in place

Place the four gradient images in **`test_images/`** with these names (any of `.png`, `.jpg`, `.jpeg`):

- `grad_x_pos`
- `grad_x_neg`
- `grad_y_pos`
- `grad_y_neg`

Example:

```text
test_images/
  grad_x_pos.png
  grad_x_neg.png
  grad_y_pos.png
  grad_y_neg.png
```

*(Optional: for ambient light subtraction, add `ambient.png` in the same folder and ensure the pipeline is configured to use it.)*

---

## 3. Optional: Perspective correction (if camera was at an angle)

Only if the camera was **not** directly above the scene:

```bash
python correct_perspective.py
```

- Click the **4 corners of the white sheet** (reference plane) in order; keep the specimen inside the quad.
- Close the window when done. Output goes to **`test_images_warped/`**.
- The pipeline will use `test_images_warped` automatically if it exists and contains all four gradient images.

---

## 4. Run the reconstruction pipeline

```bash
python test_pipeline.py
```

- Uses **`test_images/`** or **`test_images_warped/`** (if present and complete).
- To force warped images: `python test_pipeline.py --warped`
- Writes to **`Results_test/`**:
  - `depth_raw.npy`, `normals_raw.npy`
  - `normals_test.png`, `depth_test.png`, `albedo_test.png`

---

## 5. Export mesh and depth metrics

```bash
python make_mesh_from_depth.py
```

- Reads **`Results_test/depth_raw.npy`** (and normals if present).
- Writes **`Results_test/depth_mesh.obj`**, **`depth_mesh_preview.png`**, **`depth_metrics.txt`**.
- If **`config.json`** has `mm_per_pixel` and `mm_per_unit`, the OBJ is in **millimetres**.

---

## 6. Optional: Calibrate scale (for metric height in mm)

Use an object of **known height** (e.g. 2 mm gauge block, coin):

```bash
python calibrate_scale.py
```

- Click **base** then **top** of the object; enter the known height in mm.
- Optionally add more base/top pairs, then finish.
- Saves **`config.json`** with `mm_per_unit` (and optionally `mm_per_pixel`).
- Re-run **`make_mesh_from_depth.py`** afterward so the mesh uses the new calibration.

---

## 7. Optional: Query height and volume interactively

```bash
python query_depth_point.py
```

- Loads depth from **`Results_test/depth_raw.npy`** and albedo from **`Results_test/albedo_test.png`**.
- Draw ROI, then query point height, region height, or volume (uses **`config.json`** for mm if available).

---

## Quick reference (minimal run)

```bash
pip install -r requirements.txt
# Add grad_x_pos, grad_x_neg, grad_y_pos, grad_y_neg to test_images/

python test_pipeline.py
python make_mesh_from_depth.py
```

Optional: `python correct_perspective.py` → then `test_pipeline.py`; then `calibrate_scale.py` → `make_mesh_from_depth.py`; then `query_depth_point.py` for interactive measurement.
