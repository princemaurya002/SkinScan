"""
Calibrate scale so depth/height are in absolute units (mm).

WHERE THE IMAGE COMES FROM:
  The script uses depth from Results_test/depth_raw.npy (from test_pipeline.py).
  For selecting points, the window shows a real image when available:
  Results_test/albedo_test.png, or test_images. If none exist, depth map.

WHAT TO CLICK:
  Use an object of KNOWN HEIGHT (e.g. 2 mm gauge block, coin ~1.5 mm, ruler on edge).
  - 1st click: BASE (bottom) of that object
  - 2nd click: TOP of that object
  Depth at each click is averaged over a small window to reduce noise.

Steps:
  1. Run:  python calibrate_scale.py
  2. Click BASE then TOP of the known-height object (zoom/pan if needed).
  3. Enter the known height in mm (e.g. 2).
  4. Optionally enter known LENGTH (mm) for lateral scale, or Enter to skip.
  5. Optionally add more base/top pairs for a more stable average (then Enter to finish).
  6. Scale is saved to config.json.
"""

import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    cv2 = None

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEPTH_PATH = Path("Results_test") / "depth_raw.npy"
ALBEDO_PATH = Path("Results_test") / "albedo_test.png"
TEST_IMAGES_DIR = Path("test_images")
ZOOM_FACTOR = 1.25  # scroll in/out

# Median depth in a (2*radius+1)^2 window around each click to reduce noise
# Increase to 5 (11x11) if depth is very noisy; decrease to 1 (3x3) for very small objects
CALIBRATION_WINDOW_RADIUS = 3  # -> 7x7 window


def _depth_at_point_local(depth: np.ndarray, row: int, col: int, radius: int) -> Tuple[float, float, int]:
    """
    Return (median_depth, std_depth, n_valid) in a (2*radius+1)^2 window.
    Uses nanmedian so invalid pixels don't dominate.
    """
    h, w = depth.shape
    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)
    patch = depth[r0:r1, c0:c1].astype(np.float64)
    valid = np.isfinite(patch)
    n = int(np.sum(valid))
    if n == 0:
        return float(depth[row, col]), 0.0, 0
    med = float(np.nanmedian(patch))
    std = float(np.nanstd(patch)) if n > 1 else 0.0
    return med, std, n


def _load_display_image(depth_shape, h, w):
    """Load a real (photo) image for clicking; must match depth shape. Fallback to depth map."""
    # Prefer albedo (same crop as depth from test_pipeline)
    if ALBEDO_PATH.exists() and cv2 is not None:
        alb = cv2.imread(str(ALBEDO_PATH))
        if alb is not None and alb.shape[0] == h and alb.shape[1] == w:
            return cv2.cvtColor(alb, cv2.COLOR_BGR2RGB), "albedo (real image)"
        if alb is not None and (alb.shape[0], alb.shape[1]) != (h, w):
            # Resize to match depth so click coords match
            alb = cv2.resize(alb, (w, h))
            return cv2.cvtColor(alb, cv2.COLOR_BGR2RGB), "albedo (resized)"
    # Fallback: first gradient image from test_images (may need resize)
    for name in ("grad_x_pos", "grad_x_neg", "grad_y_pos", "grad_y_neg"):
        for ext in (".png", ".jpg", ".jpeg"):
            p = TEST_IMAGES_DIR / f"{name}{ext}"
            if p.exists() and cv2 is not None:
                im = cv2.imread(str(p))
                if im is not None:
                    if (im.shape[0], im.shape[1]) == (h, w):
                        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "photo (gradient)"
                    im = cv2.resize(im, (w, h))
                    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "photo (resized)"
    return None, None


def main():
    if not DEPTH_PATH.exists():
        print(f"Not found: {DEPTH_PATH}")
        print("Run test_pipeline.py (or the main pipeline) first to generate depth.")
        return

    depth = np.load(DEPTH_PATH)
    h, w = depth.shape[0], depth.shape[1]

    # Prefer real image for selection; fallback to depth map
    img, img_label = _load_display_image(depth.shape, h, w)
    if img is None:
        d_vis = depth - np.nanmin(depth)
        if np.nanmax(d_vis) > 0:
            d_vis = d_vis / np.nanmax(d_vis)
        if cv2 is not None:
            d_img = (d_vis * 255).astype(np.uint8)
            img = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = d_vis
        img_label = "depth map"

    print("Loading depth from:", DEPTH_PATH)
    print("Displaying:", img_label)
    print("Click BASE then TOP of the known-height object (use zoom/pan if needed).")
    print("Depth at each click is averaged over a 7x7 window to reduce noise.")
    print("Tips: use a flat, well-lit area on the object; avoid edges/shadows; you can add more pairs for a stable average.")

    clicks: List[Tuple[int, int]] = []
    pan_state = {"active": False, "x": 0.0, "y": 0.0, "xlim": None, "ylim": None}

    def onclick(event):
        if event.inaxes is None:
            return
        if event.button == 3:  # right: start pan
            pan_state["active"] = True
            pan_state["x"] = event.xdata
            pan_state["y"] = event.ydata
            pan_state["xlim"] = ax.get_xlim()
            pan_state["ylim"] = ax.get_ylim()
            return
        if event.button != 1:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
            clicks.append((row, col))
            med, std, n = _depth_at_point_local(depth, row, col, CALIBRATION_WINDOW_RADIUS)
            label = "BASE" if len(clicks) == 1 else "TOP"
            print(f"  Point {len(clicks)} ({label}): (row={row}, col={col})  depth_median={med:.6f}  std={std:.6f}  n_pixels={n}")
            if std > 0.01 and n > 1:
                print("    ^ Consider re-clicking in a flatter area if calibration looks wrong.")
            xs = [c for (_r, c) in clicks]
            ys = [r for (r, _c) in clicks]
            scatter_pts.set_data(xs, ys)
            fig.canvas.draw_idle()

    def onscroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        scale = 1.0 / ZOOM_FACTOR if event.button == "up" else ZOOM_FACTOR
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[0] - cur_ylim[1]) * scale
        relx = (x - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0] + 1e-9)
        rely = (y - cur_ylim[1]) / (cur_ylim[0] - cur_ylim[1] + 1e-9)
        ax.set_xlim(x - new_w * relx, x + new_w * (1 - relx))
        ax.set_ylim(y + new_h * (1 - rely), y - new_h * rely)
        fig.canvas.draw_idle()

    def onrelease(event):
        pan_state["active"] = False

    def onmotion(event):
        if not pan_state["active"] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan_state["x"]
        dy = event.ydata - pan_state["y"]
        x0, x1 = pan_state["xlim"]
        y0, y1 = pan_state["ylim"]
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    scatter_pts = ax.plot([], [], "o", color="lime", markersize=10, markeredgecolor="white", markeredgewidth=2)[0]
    scatter_pts.set_data([], [])
    ax.set_title("1st click = BASE, 2nd = TOP  |  Scroll = zoom, Right-drag = pan  |  Close when done")
    ax.set_xlabel(f"Showing: {img_label}")
    ax.imshow(img)
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("scroll_event", onscroll)
    fig.canvas.mpl_connect("button_release_event", onrelease)
    fig.canvas.mpl_connect("motion_notify_event", onmotion)
    plt.show()

    if len(clicks) < 2:
        print("Need at least 2 clicks (base then top). Run again.")
        return

    # Each pair: (z_base, z_top, delta_units, known_height_mm, mm_per_unit_from_pair)
    pair_records: List[Tuple[float, float, float, float, float]] = []

    def process_pair(r0: int, c0: int, r1: int, c1: int, known_height_mm: float) -> Optional[Tuple[float, float, float, float, float]]:
        z_base_med, z_base_std, nb = _depth_at_point_local(depth, r0, c0, CALIBRATION_WINDOW_RADIUS)
        z_top_med, z_top_std, nt = _depth_at_point_local(depth, r1, c1, CALIBRATION_WINDOW_RADIUS)
        delta_units = abs(z_top_med - z_base_med)
        if delta_units < 1e-9:
            print("  Depth difference too small; skip this pair or choose points with clear height.")
            return None
        mm_per_unit_pair = known_height_mm / delta_units
        print(f"  Base (median): {z_base_med:.6f}  Top (median): {z_top_med:.6f}  delta={delta_units:.6f}  -> mm_per_unit={mm_per_unit_pair:.6f}")
        return (z_base_med, z_top_med, delta_units, known_height_mm, mm_per_unit_pair)

    def print_pairs_table(records: List[Tuple[float, float, float, float, float]]) -> None:
        """Print a table of all calibration pairs so you can observe levels and consistency."""
        if not records:
            return
        depth_min = float(np.nanmin(depth))
        depth_max = float(np.nanmax(depth))
        print("\n  --- Calibration pairs (observed levels) ---")
        print(f"  Depth range in scene: min={depth_min:.6f}  max={depth_max:.6f}  range={depth_max - depth_min:.6f} units")
        print(f"  {'#':>2}  {'z_base':>10}  {'z_top':>10}  {'delta':>10}  {'known_mm':>8}  {'mm/unit':>10}")
        print("  " + "-" * 58)
        for i, (zb, zt, d, h_mm, mpu) in enumerate(records, 1):
            print(f"  {i:>2}  {zb:10.6f}  {zt:10.6f}  {d:10.6f}  {h_mm:8.4f}  {mpu:10.6f}")
        if len(records) > 1:
            mpu_list = [r[4] for r in records]
            mean_mpu = float(np.mean(mpu_list))
            std_mpu = float(np.std(mpu_list))
            print("  " + "-" * 58)
            print(f"  Per-pair mm_per_unit:  mean={mean_mpu:.6f}  std={std_mpu:.6f}")
            if std_mpu > 1e-6:
                print("  (Large std suggests noisy depth or inconsistent clicks; add more pairs or re-click in flatter areas.)")

    r0, c0 = clicks[0]
    r1, c1 = clicks[1]
    z_base_med, _, _ = _depth_at_point_local(depth, r0, c0, CALIBRATION_WINDOW_RADIUS)
    z_top_med, _, _ = _depth_at_point_local(depth, r1, c1, CALIBRATION_WINDOW_RADIUS)
    delta_units = abs(z_top_med - z_base_med)
    if delta_units < 1e-9:
        print("Depth difference too small; choose two points with clear height difference.")
        return

    try:
        known_height_mm = float(input("Enter known height of object (mm): ").strip())
    except (ValueError, EOFError):
        print("Invalid input. Exiting.")
        return
    if known_height_mm <= 0:
        print("Height must be positive.")
        return

    rec = process_pair(r0, c0, r1, c1, known_height_mm)
    if rec is not None:
        pair_records.append(rec)
    print_pairs_table(pair_records)

    # Optionally add more base/top pairs for a more stable scale
    while True:
        try:
            add = input("\nAdd another base/top pair? (y / Enter=no): ").strip().lower()
        except EOFError:
            add = ""
        if add != "y":
            break
        print("Click BASE then TOP of the same or another known-height object (2 clicks). Close window when done.")
        clicks2: List[Tuple[int, int]] = []
        fig2, ax2 = plt.subplots()
        ax2.set_title("Additional pair: click BASE then TOP (close when done)")
        ax2.imshow(img)
        sc2 = ax2.plot([], [], "o", color="lime", markersize=10, markeredgecolor="white", markeredgewidth=2)[0]

        def on2(event):
            if event.inaxes is None or event.button != 1:
                return
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
                clicks2.append((row, col))
                sc2.set_data([c for (_, c) in clicks2], [r for (r, _) in clicks2])
                fig2.canvas.draw_idle()

        ax2.figure.canvas.mpl_connect("button_press_event", on2)
        plt.show()
        if len(clicks2) < 2:
            print("  Need 2 clicks; skipping extra pair.")
            continue
        r0b, c0b = clicks2[0]
        r1b, c1b = clicks2[1]
        zb, _, _ = _depth_at_point_local(depth, r0b, c0b, CALIBRATION_WINDOW_RADIUS)
        zt, _, _ = _depth_at_point_local(depth, r1b, c1b, CALIBRATION_WINDOW_RADIUS)
        d2 = abs(zt - zb)
        if d2 < 1e-9:
            print("  Depth difference too small; skipping.")
            continue
        try:
            h2 = float(input("  Known height for this pair (mm): ").strip())
        except (ValueError, EOFError):
            continue
        if h2 <= 0:
            continue
        rec2 = process_pair(r0b, c0b, r1b, c1b, h2)
        if rec2 is not None:
            pair_records.append(rec2)
        print_pairs_table(pair_records)
        plt.close(fig2)

    if not pair_records:
        print("No valid pairs. Exiting.")
        return

    total_height_mm = sum(r[3] for r in pair_records)
    total_delta_units = sum(r[2] for r in pair_records)
    mm_per_unit = total_height_mm / total_delta_units

    print("\n  --- Final scale ---")
    print(f"  Pairs used: {len(pair_records)}")
    print(f"  Combined: total_height_mm={total_height_mm:.4f}  total_delta_units={total_delta_units:.6f}")
    print(f"  mm_per_unit = total_height_mm / total_delta_units = {mm_per_unit:.6f}")
    print("  (Height difference in mm = depth_difference_units * mm_per_unit)")

    mm_per_pixel = None
    # Option A: use same two points' pixel distance (assumes height direction in image)
    try:
        known_length_mm = input("Optional: enter known LENGTH (mm) between your two points, or Enter to skip: ").strip()
        if known_length_mm:
            known_length_mm = float(known_length_mm)
            dist_px = np.sqrt((c1 - c0) ** 2 + (r1 - r0) ** 2)
            if dist_px < 1:
                print("  Pixel distance too small; lateral scale not set.")
            else:
                mm_per_pixel = known_length_mm / dist_px
                print(f"  mm_per_pixel = {mm_per_pixel:.6f}  (from same two points)")
    except (ValueError, EOFError):
        pass

    # Option B: ruler-based lateral (two points on a ruler, horizontal distance in mm)
    if mm_per_pixel is None:
        try:
            ruler = input("Or calibrate lateral from a ruler? Enter distance between two marks (mm), then click those 2 points (or Enter to skip): ").strip()
            if ruler:
                dist_mm = float(ruler)
                print("Click two points on the ruler (same line, known distance apart).")
                clicks_r: List[Tuple[int, int]] = []
                fig_r, ax_r = plt.subplots()
                ax_r.set_title("Ruler: click first point then second")
                ax_r.imshow(img)
                sr = ax_r.plot([], [], "o", color="cyan", markersize=10, markeredgecolor="white", markeredgewidth=2)[0]

                def on_r(event):
                    if event.inaxes is None or event.button != 1:
                        return
                    col = int(round(event.xdata))
                    row = int(round(event.ydata))
                    if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
                        clicks_r.append((row, col))
                        sr.set_data([c for (_, c) in clicks_r], [r for (r, _) in clicks_r])
                        fig_r.canvas.draw_idle()

                ax_r.figure.canvas.mpl_connect("button_press_event", on_r)
                plt.show()
                if len(clicks_r) >= 2:
                    r0r, c0r = clicks_r[0]
                    r1r, c1r = clicks_r[1]
                    dist_px_r = np.sqrt((c1r - c0r) ** 2 + (r1r - r0r) ** 2)
                    if dist_px_r >= 1:
                        mm_per_pixel = dist_mm / dist_px_r
                        print(f"  mm_per_pixel = {mm_per_pixel:.6f}  (from ruler)")
                plt.close(fig_r)
        except (ValueError, EOFError):
            pass

    # Load existing config, update scale, save
    config = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    config["mm_per_unit"] = mm_per_unit
    if mm_per_pixel is not None:
        config["mm_per_pixel"] = mm_per_pixel
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved to {CONFIG_PATH}. You now get absolute (mm) values in query_depth_point.py and make_mesh_from_depth.py.")
    print("For how 3D reconstruction and height difference are calculated, see docs/HOW_3D_AND_HEIGHT_WORK.md")


if __name__ == "__main__":
    main()
