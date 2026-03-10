

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("Results_test")
DEPTH_RAW_PATH = BASE_DIR / "depth_raw.npy"
ALBEDO_PATH = BASE_DIR / "albedo_test.png"
NORMALS_PNG_PATH = BASE_DIR / "normals_test.png"

LOCAL_WINDOW_RADIUS = 15
HEIGHT_MODE = "local_plane"

# Calibration: loaded from config.json if present (run calibrate_scale.py to set)
try:
    from config_loader import load_config
    _cfg = load_config()
    MM_PER_UNIT: Optional[float] = _cfg.get("mm_per_unit")
    MM_PER_PIXEL: Optional[float] = _cfg.get("mm_per_pixel")
except Exception:
    MM_PER_UNIT: Optional[float] = None
    MM_PER_PIXEL: Optional[float] = None

DEFAULT_VIEW_MODE = "albedo"
SCROLL_ZOOM_FACTOR = 1.25
MAGNIFIER_RADIUS = 40
SHOW_MAGNIFIER = True
SHOW_CROSSHAIR = True

# Annulus for lesion baseline: ring outside ROI (pixels)
ANNULUS_INNER_MARGIN = 2   # pixels inside ROI boundary (0 = use ROI edge)
ANNULUS_OUTER_MARGIN = 15  # pixels outside ROI for plane-fitting ring

# Two-point mode: only if |depth_diff| <= this do we treat as "same horizontal level".
# Use strict (small) values so we rarely wrongly say "same level" when there is real height difference.
# Vertical height is always reported as raw depth difference (d2-d1), not local-plane height.
SAME_LEVEL_ABSOLUTE_THRESHOLD = 0.001   # in depth units (strict: only tiny diff = same level)
SAME_LEVEL_RELATIVE_FRACTION = 0.003    # 0.3% of depth range
# Median depth in this window at each click for more stable vertical height (0 = single pixel)
TWO_POINT_DEPTH_RADIUS = 2   # 5x5 window; set 0 to use single-pixel depth


def depth_at_point_median(depth: np.ndarray, row: int, col: int, radius: int) -> float:
    """Median depth in a (2*radius+1)^2 window; falls back to single pixel if radius 0."""
    if radius <= 0:
        return float(depth[row, col])
    h, w = depth.shape
    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)
    patch = depth[r0:r1, c0:c1].astype(np.float64)
    valid = np.isfinite(patch)
    if not np.any(valid):
        return float(depth[row, col])
    return float(np.nanmedian(patch))


def compute_local_height(
    depth: np.ndarray, row: int, col: int, radius: int
) -> Tuple[float, float]:
    """
    Compute depth and local relative height at (row, col).

    - depth_at_point: raw depth at (row, col)
    - local_height: depth_at_point - local_min in a neighborhood window
    """
    h, w = depth.shape
    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)

    window = depth[r0:r1, c0:c1]
    depth_at_point = float(depth[row, col])
    local_min = float(np.nanmin(window))
    local_height = depth_at_point - local_min
    return depth_at_point, local_height


def fit_plane_height(
    depth: np.ndarray, row: int, col: int, radius: int
) -> Tuple[float, float]:
    """
    Height above a locally fitted plane z = ax + by + c within a window.

    Returns:
      depth_at_point, height_above_plane
    """
    h, w = depth.shape
    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)

    z = depth[r0:r1, c0:c1]
    # build (x,y) coordinates in image space (cols=x, rows=y)
    ys, xs = np.mgrid[r0:r1, c0:c1]
    xs = xs.astype(np.float64).reshape(-1)
    ys = ys.astype(np.float64).reshape(-1)
    zs = z.astype(np.float64).reshape(-1)

    # Remove NaNs
    m = np.isfinite(zs)
    xs, ys, zs = xs[m], ys[m], zs[m]
    if zs.size < 10:
        # fallback to local_min if not enough data
        return compute_local_height(depth, row, col, radius)

    A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)  # a,b,c
    a, b, c = coeff
    depth_at_point = float(depth[row, col])
    plane_at_point = float(a * col + b * row + c)
    return depth_at_point, float(depth_at_point - plane_at_point)


def mask_from_polygon(verts: np.ndarray, h: int, w: int) -> np.ndarray:
    """verts: (N,2) in (col, row) / (x, y) image coordinates. Returns (h,w) bool mask."""
    if len(verts) < 3:
        return np.zeros((h, w), dtype=bool)
    path = MplPath(verts)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.column_stack([cols.ravel(), rows.ravel()])
    return path.contains_points(pts).reshape(h, w)


def fit_plane_to_region(depth: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """Fit z = a*col + b*row + c to depth where mask is True. Returns (a, b, c)."""
    rows, cols = np.where(mask)
    zs = depth[mask].astype(np.float64)
    if len(zs) < 10:
        return 0.0, 0.0, float(np.nanmean(depth))
    A = np.stack([cols, rows, np.ones_like(cols, dtype=np.float64)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def annulus_ring_mask(roi_mask: np.ndarray, inner_margin: int, outer_margin: int) -> np.ndarray:
    """
    Ring outside ROI: pixels near the lesion but not inside it.
    dilate(roi, outer) - roi gives the ring around the boundary. Requires cv2.
    """
    if cv2 is None:
        return np.zeros_like(roi_mask)
    roi_uint = (roi_mask.astype(np.uint8)) * 255
    k_outer = max(3, 2 * outer_margin + 1)
    kernel_o = np.ones((k_outer, k_outer), np.uint8)
    dilated = cv2.dilate(roi_uint, kernel_o)
    ring = (dilated > 127) & (~roi_mask)
    return ring


def roi_stats(
    depth: np.ndarray,
    roi_mask: np.ndarray,
    baseline: str = "min",
    plane_coeffs: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float, int]:
    """
    baseline: "min" = min depth in ROI; "plane" = use plane_coeffs (a,b,c) for z = a*col + b*row + c.
    Returns: max_height, mean_height, volume (units * px^2), n_pixels.
    """
    z_roi = np.where(roi_mask, depth, np.nan)
    n_pixels = int(np.sum(roi_mask))
    if n_pixels == 0:
        return 0.0, 0.0, 0.0, 0

    if baseline == "min":
        z0 = np.nanmin(z_roi)
        height = np.where(roi_mask, depth - z0, np.nan)
    else:
        a, b, c = plane_coeffs or (0.0, 0.0, float(np.nanmean(depth)))
        rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        plane = a * cols + b * rows + c
        height = np.where(roi_mask, depth - plane, np.nan)
    height_roi = np.nan_to_num(height, nan=0.0) * roi_mask.astype(np.float64)
    max_h = float(np.max(height_roi)) if np.any(roi_mask) else 0.0
    mean_h = float(np.sum(height_roi) / n_pixels) if n_pixels else 0.0
    volume = float(np.sum(height_roi))
    return max_h, mean_h, volume, n_pixels


def main() -> None:
    if not DEPTH_RAW_PATH.exists():
        raise FileNotFoundError(
            f"{DEPTH_RAW_PATH} not found. Run `python test_pipeline.py` first."
        )

    depth = np.load(DEPTH_RAW_PATH)
    depth_f64 = depth.astype(np.float64, copy=False)

    # Optional: smooth subpixel sampling if scipy is present
    try:
        from scipy.ndimage import map_coordinates  # type: ignore
    except Exception:
        map_coordinates = None

    def build_depth_viz() -> Tuple[np.ndarray, str | None]:
        d_vis = depth - np.nanmin(depth)
        if np.nanmax(d_vis) > 0:
            d_vis = d_vis / np.nanmax(d_vis)
        if cv2 is not None:
            d_img = (d_vis * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
            d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
            return d_color, None
        return d_vis, "gray"

    def load_png_rgb(path: Path) -> Tuple[np.ndarray, str | None]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(str(path))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, None
        # fallback: matplotlib can read via plt.imread
        img = plt.imread(str(path))
        return img, None

    def choose_initial_view() -> Tuple[str, np.ndarray, str | None]:
        mode = DEFAULT_VIEW_MODE
        if mode == "albedo" and ALBEDO_PATH.exists():
            img, cmap = load_png_rgb(ALBEDO_PATH)
            return "albedo", img, cmap
        if mode == "normals" and NORMALS_PNG_PATH.exists():
            img, cmap = load_png_rgb(NORMALS_PNG_PATH)
            return "normals", img, cmap
        # fallback
        img, cmap = build_depth_viz()
        return "depth", img, cmap

    mode, img_to_show, cmap = choose_initial_view()

    # Interaction state
    interaction_mode = "point"  # "point" | "two_point" | "polygon" | "set_base" | "line"
    two_point_state: dict = {"first": None}  # (row, col) or None
    line_state: dict = {"first": None}  # (row, col) or None
    polygon_selector: Optional[PolygonSelector] = None
    polygon_patch = None
    polygon_use_annulus = False

    # Base reference (white sheet = 0 height). Set with key "b" + click, clear with "0"
    base_reference: Optional[float] = None
    base_marker = None

    fig, ax = plt.subplots()
    ax.set_title(
        "p=point  t=two-point  l=line profile  r=ROI  a=ROI+annulus  b=set base  0=clear base | 1/2/3=view  h/m/c | scroll=zoom  rmb=pan"
    )
    im = ax.imshow(img_to_show, cmap=cmap)
    marker = ax.plot([], [], "wo", markersize=6, markeredgecolor="k", markeredgewidth=1)[0]
    pt_a_marker = ax.plot([], [], "s", color="lime", markersize=8, markeredgecolor="k", markeredgewidth=1)[0]
    pt_b_marker = ax.plot([], [], "s", color="cyan", markersize=8, markeredgecolor="k", markeredgewidth=1)[0]
    base_marker = ax.plot([], [], "s", color="yellow", markersize=10, markeredgecolor="red", markeredgewidth=2)[0]
    line_a_marker = ax.plot([], [], "x", color="magenta", markersize=10, markeredgecolor="k", markeredgewidth=1)[0]
    line_b_marker = ax.plot([], [], "x", color="magenta", markersize=10, markeredgecolor="k", markeredgewidth=1)[0]
    line_seg = ax.plot([], [], "-", color="magenta", lw=1.5, alpha=0.9)[0]
    pt_a_marker.set_visible(False)
    pt_b_marker.set_visible(False)
    base_marker.set_visible(False)
    line_a_marker.set_visible(False)
    line_b_marker.set_visible(False)
    line_seg.set_visible(False)

    vline = ax.axvline(x=0, color="w", lw=0.8, alpha=0.8, visible=False)
    hline = ax.axhline(y=0, color="w", lw=0.8, alpha=0.8, visible=False)

    mag_ax: Optional[Axes] = None
    mag_im = None
    if SHOW_MAGNIFIER:
        mag_ax = fig.add_axes([0.72, 0.72, 0.25, 0.25])
        mag_ax.set_title("magnifier")
        mag_ax.set_xticks([])
        mag_ax.set_yticks([])
        mag_im = mag_ax.imshow(img_to_show, cmap=cmap)

    def set_mode_status():
        nonlocal base_reference
        if interaction_mode == "set_base":
            s = "Set base: click on white sheet (0 height reference)"
        elif interaction_mode == "two_point":
            s = "Two-point: click A then B" if two_point_state["first"] is None else "Two-point: click B"
        elif interaction_mode == "line":
            s = "Line profile: click A then B" if line_state["first"] is None else "Line profile: click B"
        elif interaction_mode == "polygon":
            s = "Draw polygon (close shape to finish)"
        else:
            s = "Point: click to query"
        if base_reference is not None:
            s += "  |  Base=0 set"
        ax.set_ylabel(s)
        fig.canvas.draw_idle()

    def _sample_depth_profile(
        r1: int, c1: int, r2: int, c2: int, n: int = 300
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (dist, z) sampled along the line from A(r1,c1) to B(r2,c2).

        - dist: in px or mm depending on calibration.
        - z: depth in raw units (not base-subtracted).

        Sampling is dense: ~4 samples per pixel along the line, so fine ridges/valleys
        appear clearly in the profile.
        """
        # Compute geometric length of the line in pixels
        line_len_px = float(np.hypot(c2 - c1, r2 - r1))
        # Target ~4 samples per pixel (min 50, max 5000 for safety)
        n_auto = int(max(50, min(5000, line_len_px * 4.0)))
        n = int(max(n_auto, 2))
        rr = np.linspace(r1, r2, n, dtype=np.float64)
        cc = np.linspace(c1, c2, n, dtype=np.float64)
        if map_coordinates is not None:
            z = map_coordinates(depth_f64, [rr, cc], order=1, mode="nearest")
        else:
            ri = np.clip(np.rint(rr).astype(int), 0, depth.shape[0] - 1)
            ci = np.clip(np.rint(cc).astype(int), 0, depth.shape[1] - 1)
            z = depth_f64[ri, ci]

        dist_px = np.sqrt((cc - cc[0]) ** 2 + (rr - rr[0]) ** 2)
        if MM_PER_PIXEL is not None and MM_PER_PIXEL > 0:
            return dist_px * MM_PER_PIXEL, z
        return dist_px, z

    def onclick(event):
        nonlocal base_reference, interaction_mode
        if not event.inaxes or event.inaxes != ax:
            return
        if event.button != 1:
            return
        # Let PolygonSelector handle clicks when drawing ROI
        if interaction_mode == "polygon" and polygon_selector is not None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if row < 0 or col < 0 or row >= depth.shape[0] or col >= depth.shape[1]:
            return

        # Set base (white sheet = 0 height): one click sets reference and returns to point mode
        if interaction_mode == "set_base":
            base_reference = float(depth[row, col])
            base_marker.set_data([col], [row])
            base_marker.set_visible(True)
            interaction_mode = "point"
            msg = f"Base set at (row={row}, col={col}), depth={base_reference:.6f}. All heights = above this surface (0 base)."
            if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                msg += f"  (depth in calibrated units)"
            print(msg)
            ax.set_xlabel(msg)
            set_mode_status()
            fig.canvas.draw_idle()
            return

        # Two-point mode: first click = A, second = B
        if interaction_mode == "two_point":
            if two_point_state["first"] is None:
                two_point_state["first"] = (row, col)
                pt_a_marker.set_data([col], [row])
                pt_a_marker.set_visible(True)
                pt_b_marker.set_visible(False)
                # Hide line UI
                line_state["first"] = None
                line_a_marker.set_visible(False)
                line_b_marker.set_visible(False)
                line_seg.set_visible(False)
                set_mode_status()
                return
            else:
                r1, c1 = two_point_state["first"]
                r2, c2 = row, col
                two_point_state["first"] = None
                pt_b_marker.set_data([c2], [r2])
                pt_b_marker.set_visible(True)

                # Lateral distance (in image plane) = horizontal spacing between the two points
                dist_px = np.sqrt((c2 - c1) ** 2 + (r2 - r1) ** 2)
                d1 = depth_at_point_median(depth, r1, c1, TWO_POINT_DEPTH_RADIUS)
                d2 = depth_at_point_median(depth, r2, c2, TWO_POINT_DEPTH_RADIUS)
                # Vertical height = raw depth difference (actual height between the two points)
                depth_diff = d2 - d1
                vertical_height_units = depth_diff

                # 3D distance (when calibrated): (col, row, depth) -> (X,Y,Z) in mm, then Euclidean
                z1 = d1 if base_reference is None else d1 - base_reference
                z2 = d2 if base_reference is None else d2 - base_reference
                if MM_PER_PIXEL is not None and MM_PER_PIXEL > 0 and MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                    x1_mm = c1 * MM_PER_PIXEL
                    y1_mm = r1 * MM_PER_PIXEL
                    z1_mm = z1 * MM_PER_UNIT
                    x2_mm = c2 * MM_PER_PIXEL
                    y2_mm = r2 * MM_PER_PIXEL
                    z2_mm = z2 * MM_PER_UNIT
                    dist_3d_mm = np.sqrt((x2_mm - x1_mm) ** 2 + (y2_mm - y1_mm) ** 2 + (z2_mm - z1_mm) ** 2)
                else:
                    dist_3d_mm = None

                # Classify: same level only when depth difference is very small (strict threshold)
                depth_range = float(np.nanmax(depth) - np.nanmin(depth)) or 1.0
                thresh = max(
                    SAME_LEVEL_ABSOLUTE_THRESHOLD,
                    SAME_LEVEL_RELATIVE_FRACTION * depth_range,
                )
                same_level = abs(depth_diff) <= thresh

                if same_level:
                    msg = (
                        f"[Same level] A=({r1},{c1}) B=({r2},{c2})  "
                        f"Lateral distance={dist_px:.1f} px"
                    )
                    if MM_PER_PIXEL is not None and MM_PER_PIXEL > 0:
                        msg += f"  ({dist_px * MM_PER_PIXEL:.3f} mm)"
                    msg += f"  | Vertical height≈0 (depth_diff={depth_diff:.6f})"
                else:
                    msg = (
                        f"[Vertical] A=({r1},{c1}) B=({r2},{c2})  "
                        f"Vertical height={vertical_height_units:.6f} units"
                    )
                    if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                        msg += f"  ({vertical_height_units * MM_PER_UNIT:.4f} mm)"
                    msg += f"  | Lateral distance={dist_px:.1f} px"
                    if MM_PER_PIXEL is not None and MM_PER_PIXEL > 0:
                        msg += f"  ({dist_px * MM_PER_PIXEL:.3f} mm)"
                if dist_3d_mm is not None:
                    msg += f"  |  3D distance={dist_3d_mm:.4f} mm"
                if base_reference is not None:
                    a_above = d1 - base_reference
                    b_above = d2 - base_reference
                    msg += f"  |  A above base={a_above:.6f}  B above base={b_above:.6f}"
                    if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                        msg += f"  ({a_above * MM_PER_UNIT:.4f} / {b_above * MM_PER_UNIT:.4f} mm)"
                print(msg)
                ax.set_xlabel(msg)
                set_mode_status()
                fig.canvas.draw_idle()
                return

        # Line profile mode: click A then B, then show a height profile plot
        if interaction_mode == "line":
            if line_state["first"] is None:
                line_state["first"] = (row, col)
                line_a_marker.set_data([col], [row])
                line_a_marker.set_visible(True)
                line_b_marker.set_visible(False)
                line_seg.set_visible(False)
                # Hide two-point UI
                two_point_state["first"] = None
                pt_a_marker.set_visible(False)
                pt_b_marker.set_visible(False)
                set_mode_status()
                fig.canvas.draw_idle()
                return
            else:
                r1, c1 = line_state["first"]
                r2, c2 = row, col
                line_state["first"] = None

                line_b_marker.set_data([c2], [r2])
                line_b_marker.set_visible(True)
                line_seg.set_data([c1, c2], [r1, r2])
                line_seg.set_visible(True)

                dist, z = _sample_depth_profile(r1, c1, r2, c2, n=400)

                # Relative height: prefer base if set, otherwise relative to min along line
                if base_reference is not None:
                    rel = z - base_reference
                    baseline_label = "base (0 height)"
                else:
                    z0 = float(np.nanmin(z)) if np.isfinite(z).any() else 0.0
                    rel = z - z0
                    baseline_label = "min along line"

                # Convert to mm if calibrated
                y = rel
                y_label = "Relative height (units)"
                if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                    y = rel * MM_PER_UNIT
                    y_label = "Relative height (mm)"

                x = dist
                x_label = "Distance (px)"
                if MM_PER_PIXEL is not None and MM_PER_PIXEL > 0:
                    x_label = "Distance (mm)"

                fig2, ax2 = plt.subplots()
                ax2.plot(x, y, color="tab:blue", lw=2)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel(x_label)
                ax2.set_ylabel(y_label)
                ax2.set_title(f"Line profile A=({r1},{c1}) → B=({r2},{c2}) | baseline: {baseline_label}")
                fig2.tight_layout()
                fig2.show()

                msg = f"Line profile plotted: A=({r1},{c1}) B=({r2},{c2})  baseline={baseline_label}"
                print(msg)
                ax.set_xlabel(msg)
                set_mode_status()
                fig.canvas.draw_idle()
                return

        # Point mode: single-point query
        if HEIGHT_MODE == "local_plane":
            depth_at_point, height_val = fit_plane_height(depth, row, col, LOCAL_WINDOW_RADIUS)
            height_label = "height_above_local_plane"
        else:
            depth_at_point, height_val = compute_local_height(depth, row, col, LOCAL_WINDOW_RADIUS)
            height_label = "height_above_local_min"

        msg = (
            f"(row={row}, col={col})  "
            f"depth={depth_at_point:.6f}  "
            f"{height_label}={height_val:.6f} units (vertical relief at this point)"
        )
        if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
            h_mm = height_val * MM_PER_UNIT
            msg += f"  (~{h_mm:.4f} mm)"
        if base_reference is not None:
            height_above_base = depth_at_point - base_reference
            msg += f"  |  height_above_base={height_above_base:.6f} units"
            if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
                msg += f"  ({height_above_base * MM_PER_UNIT:.4f} mm)"

        print(msg)
        ax.set_xlabel(msg)
        marker.set_data([col], [row])
        if SHOW_CROSSHAIR:
            vline.set_xdata([col])
            hline.set_ydata([row])
            vline.set_visible(True)
            hline.set_visible(True)

        if SHOW_MAGNIFIER and mag_ax is not None and mag_im is not None:
            update_magnifier(col, row)
        fig.canvas.draw_idle()

    def update_magnifier(col: int, row: int) -> None:
        if mag_ax is None or mag_im is None:
            return
        r = MAGNIFIER_RADIUS
        r0 = max(0, row - r)
        r1 = min(depth.shape[0], row + r + 1)
        c0 = max(0, col - r)
        c1 = min(depth.shape[1], col + r + 1)

        # Re-slice the current displayed image
        cur = im.get_array()
        # cur can be float 2D, or RGB uint8 (H,W,3)
        patch = cur[r0:r1, c0:c1]
        mag_im.set_data(patch)
        mag_ax.set_xlim(0, patch.shape[1] - 1)
        mag_ax.set_ylim(patch.shape[0] - 1, 0)

        mag_ax.set_title(f"magnifier (r={MAGNIFIER_RADIUS})")

    def on_polygon_done(verts: np.ndarray):
        nonlocal polygon_patch
        if len(verts) < 3:
            return
        h, w = depth.shape
        roi_mask = mask_from_polygon(verts, h, w)
        n_px = int(np.sum(roi_mask))
        if n_px == 0:
            return

        # If user set base (white sheet = 0), use it as ROI baseline
        if base_reference is not None:
            max_h, mean_h, vol, _ = roi_stats(
                depth, roi_mask, baseline="plane", plane_coeffs=(0.0, 0.0, base_reference)
            )
            baseline_name = "base (0 height)"
        elif polygon_use_annulus and cv2 is not None:
            ring = annulus_ring_mask(roi_mask, ANNULUS_INNER_MARGIN, ANNULUS_OUTER_MARGIN)
            if np.sum(ring) >= 10:
                a, b, c = fit_plane_to_region(depth, ring)
                max_h, mean_h, vol, _ = roi_stats(depth, roi_mask, baseline="plane", plane_coeffs=(a, b, c))
                baseline_name = "annulus plane"
            else:
                max_h, mean_h, vol, _ = roi_stats(depth, roi_mask, baseline="min")
                baseline_name = "min (annulus too small)"
        else:
            max_h, mean_h, vol, _ = roi_stats(depth, roi_mask, baseline="min")
            baseline_name = "min in ROI"

        msg = (
            f"ROI: n={n_px} px  baseline={baseline_name}  "
            f"max_height={max_h:.6f}  mean_height={mean_h:.6f}  volume={vol:.4f} (units·px²)"
        )
        if MM_PER_UNIT is not None and MM_PER_UNIT > 0:
            msg += f"  max_h={max_h * MM_PER_UNIT:.4f} mm  mean_h={mean_h * MM_PER_UNIT:.4f} mm"
        if MM_PER_UNIT is not None and MM_PER_PIXEL is not None and MM_PER_UNIT > 0 and MM_PER_PIXEL > 0:
            vol_mm3 = vol * MM_PER_UNIT * (MM_PER_PIXEL ** 2)
            msg += f"  volume={vol_mm3:.6f} mm³"
        print(msg)
        ax.set_xlabel(msg)

        if polygon_patch is not None:
            polygon_patch.remove()
        from matplotlib.patches import Polygon as MplPolygon
        polygon_patch = ax.add_patch(MplPolygon(verts, fill=True, facecolor="green", edgecolor="yellow", alpha=0.35))
        if polygon_selector is not None:
            polygon_selector.set_active(False)
        set_mode_status()
        fig.canvas.draw_idle()

    def onkey(event):
        nonlocal mode, img_to_show, cmap, interaction_mode, polygon_selector, polygon_use_annulus, base_reference
        global SHOW_MAGNIFIER, SHOW_CROSSHAIR, HEIGHT_MODE
        if event.key == "b":
            interaction_mode = "set_base"
            pt_a_marker.set_visible(False)
            pt_b_marker.set_visible(False)
            two_point_state["first"] = None
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            set_mode_status()
            fig.canvas.draw_idle()
            return
        if event.key == "0":
            base_reference = None
            base_marker.set_visible(False)
            set_mode_status()
            fig.canvas.draw_idle()
            return
        if event.key == "p":
            interaction_mode = "point"
            pt_a_marker.set_visible(False)
            pt_b_marker.set_visible(False)
            two_point_state["first"] = None
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            set_mode_status()
            fig.canvas.draw_idle()
            return
        if event.key == "t":
            interaction_mode = "two_point"
            two_point_state["first"] = None
            pt_a_marker.set_visible(False)
            pt_b_marker.set_visible(False)
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            set_mode_status()
            fig.canvas.draw_idle()
            return
        if event.key == "l":
            interaction_mode = "line"
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            two_point_state["first"] = None
            pt_a_marker.set_visible(False)
            pt_b_marker.set_visible(False)
            set_mode_status()
            fig.canvas.draw_idle()
            return
        if event.key == "r":
            interaction_mode = "polygon"
            polygon_use_annulus = False
            if polygon_selector is not None:
                getattr(polygon_selector, "_disconnect_events", lambda: None)()
                polygon_selector = None
            polygon_selector = PolygonSelector(ax, on_polygon_done, useblit=False)
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            set_mode_status()
            return
        if event.key == "a":
            interaction_mode = "polygon"
            polygon_use_annulus = True
            if polygon_selector is not None:
                getattr(polygon_selector, "_disconnect_events", lambda: None)()
                polygon_selector = None
            polygon_selector = PolygonSelector(ax, on_polygon_done, useblit=False)
            line_state["first"] = None
            line_a_marker.set_visible(False)
            line_b_marker.set_visible(False)
            line_seg.set_visible(False)
            set_mode_status()
            return
        if event.key == "1":
            if ALBEDO_PATH.exists():
                mode = "albedo"
                img_to_show, cmap = load_png_rgb(ALBEDO_PATH)
        elif event.key == "2":
            mode = "depth"
            img_to_show, cmap = build_depth_viz()
        elif event.key == "3":
            if NORMALS_PNG_PATH.exists():
                mode = "normals"
                img_to_show, cmap = load_png_rgb(NORMALS_PNG_PATH)
        elif event.key == "h":
            HEIGHT_MODE = "local_min" if HEIGHT_MODE == "local_plane" else "local_plane"
            ax.set_ylabel(f"view: {mode} | height_mode: {HEIGHT_MODE}")
            fig.canvas.draw_idle()
            return
        elif event.key == "m":
            SHOW_MAGNIFIER = not SHOW_MAGNIFIER
            if mag_ax is not None:
                mag_ax.set_visible(SHOW_MAGNIFIER)
            fig.canvas.draw_idle()
            return
        elif event.key == "c":
            SHOW_CROSSHAIR = not SHOW_CROSSHAIR
            if not SHOW_CROSSHAIR:
                vline.set_visible(False)
                hline.set_visible(False)
            fig.canvas.draw_idle()
            return
        else:
            return
        im.set_data(img_to_show)
        if cmap is not None:
            im.set_cmap(cmap)
        ax.set_ylabel(f"view: {mode} | height_mode: {HEIGHT_MODE}")
        if mag_ax is not None and mag_im is not None:
            mag_im.set_data(img_to_show)
        fig.canvas.draw_idle()

    # Scroll wheel zoom (zoom towards cursor)
    def onscroll(event):
        if event.inaxes != ax:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        if event.button == "up":
            scale_factor = 1.0 / SCROLL_ZOOM_FACTOR
        else:
            scale_factor = SCROLL_ZOOM_FACTOR

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[0] - cur_ylim[1]) * scale_factor  # note inverted y

        relx = (x - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0] + 1e-9)
        rely = (y - cur_ylim[1]) / (cur_ylim[0] - cur_ylim[1] + 1e-9)

        ax.set_xlim([x - new_width * relx, x + new_width * (1 - relx)])
        ax.set_ylim([y + new_height * (1 - rely), y - new_height * rely])
        fig.canvas.draw_idle()

    # Simple pan with right mouse drag
    pan_state = {"pressed": False, "x": 0.0, "y": 0.0, "xlim": None, "ylim": None}

    def onpress(event):
        if event.inaxes != ax:
            return
        if event.button != 3:  # right mouse
            return
        pan_state["pressed"] = True
        pan_state["x"] = event.xdata
        pan_state["y"] = event.ydata
        pan_state["xlim"] = ax.get_xlim()
        pan_state["ylim"] = ax.get_ylim()

    def onrelease(event):
        pan_state["pressed"] = False

    def onmotion(event):
        if not pan_state["pressed"] or event.inaxes != ax:
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

    set_mode_status()
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    kid = fig.canvas.mpl_connect("key_press_event", onkey)
    sid = fig.canvas.mpl_connect("scroll_event", onscroll)
    pid = fig.canvas.mpl_connect("button_press_event", onpress)
    rid = fig.canvas.mpl_connect("button_release_event", onrelease)
    mid = fig.canvas.mpl_connect("motion_notify_event", onmotion)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(kid)
    fig.canvas.mpl_disconnect(sid)
    fig.canvas.mpl_disconnect(pid)
    fig.canvas.mpl_disconnect(rid)
    fig.canvas.mpl_disconnect(mid)


if __name__ == "__main__":
    main()

