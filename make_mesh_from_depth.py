
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None


def save_depth_mesh_as_obj(
    depth: np.ndarray,
    out_path: Path,
    normals: Optional[np.ndarray] = None,
    depth_scale: float = 1.0,
    zero_base: bool = True,
) -> None:
    """
    Create a height-field OBJ mesh from a depth map.
    x, y = pixel indices; z = depth value (scaled).
    """
    depth = np.asarray(depth, dtype=np.float64)
    if zero_base:
        depth = depth - np.nanmin(depth)
    depth = depth * depth_scale
    depth = np.nan_to_num(depth, nan=0.0)

    h, w = depth.shape

    # Vertex grid: x = col, y = row, z = depth
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    zs = depth.astype(np.float32)
    vertices = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)

    def vid(i: int, j: int) -> int:
        return i * w + j + 1  # OBJ is 1-based

    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v1, v2 = vid(i, j), vid(i, j + 1)
            v3, v4 = vid(i + 1, j), vid(i + 1, j + 1)
            faces.append((v1, v2, v3))
            faces.append((v3, v2, v4))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None and normals.shape[:2] == (h, w):
            nflat = normals.reshape(-1, 3)
            for n in nflat:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for (a, b, c) in faces:
            if normals is not None:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")


def compute_depth_height_stats(
    depth: np.ndarray,
    roi: Optional[Tuple[slice, slice]] = None,
    mm_per_unit: Optional[float] = None,
) -> dict:
    """
    Compute min, max, mean, std and max relative height (optionally in mm).
    roi: (row_slice, col_slice) e.g. (slice(100,200), slice(150,250))
    """
    if roi is not None:
        r, c = roi
        d = depth[r, c]
    else:
        d = depth

    d_min = float(np.nanmin(d))
    d_max = float(np.nanmax(d))
    d_mean = float(np.nanmean(d))
    d_std = float(np.nanstd(d))
    rel_height = d - d_min
    h_max_units = float(np.nanmax(rel_height))

    out = {
        "depth_min": d_min,
        "depth_max": d_max,
        "depth_mean": d_mean,
        "depth_std": d_std,
        "height_max_units": h_max_units,
    }
    if mm_per_unit is not None and mm_per_unit > 0:
        out["height_max_mm"] = h_max_units * mm_per_unit
    return out


def write_depth_metrics_file(
    depth: np.ndarray,
    out_path: Union[str, Path],
    roi: Optional[Tuple[slice, slice]] = None,
    mm_per_unit: Optional[float] = None,
) -> None:
    """Write depth/height statistics to a text file."""
    stats = compute_depth_height_stats(depth, roi=roi, mm_per_unit=mm_per_unit)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("SkinScan depth/height metrics\n")
        f.write("----------------------------\n")
        f.write(f"Depth shape: {depth.shape}\n")
        if roi is not None:
            f.write(f"ROI: rows {roi[0]}, cols {roi[1]}\n")
        f.write(f"Depth min (units):     {stats['depth_min']:.6f}\n")
        f.write(f"Depth max (units):     {stats['depth_max']:.6f}\n")
        f.write(f"Depth mean (units):    {stats['depth_mean']:.6f}\n")
        f.write(f"Depth std (units):     {stats['depth_std']:.6f}\n")
        f.write(f"Max height (units):     {stats['height_max_units']:.6f}\n")
        if "height_max_mm" in stats:
            f.write(f"Max height (mm):       {stats['height_max_mm']:.4f}\n")
        if mm_per_unit is None:
            f.write("\n(Set mm_per_unit after calibration for metric height.)\n")


def main():
    base = Path("Results_test")
    depth_path = base / "depth_raw.npy"
    normals_path = base / "normals_raw.npy"

    if not depth_path.exists():
        print("Run test_pipeline.py first to generate Results_test/depth_raw.npy")
        return

    depth = np.load(depth_path)

    # Optional: scale z for nicer mesh proportions
    depth_scale = 80.0
    zero_base = True

    normals = None
    if normals_path.exists():
        normals = np.load(normals_path)

    # Export OBJ mesh
    save_depth_mesh_as_obj(
        depth,
        base / "depth_mesh.obj",
        normals=normals,
        depth_scale=depth_scale,
        zero_base=zero_base,
    )
    print(f"Saved mesh to {base / 'depth_mesh.obj'}")

    # Preview PNG (depth after zero_base, before scale)
    d_vis = depth - np.nanmin(depth)
    if np.nanmax(d_vis) > 0:
        d_vis = d_vis / np.nanmax(d_vis)
    d_vis = np.clip(d_vis * 255, 0, 255).astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(str(base / "depth_mesh_preview.png"), d_vis)
        print(f"Saved preview to {base / 'depth_mesh_preview.png'}")

    # Global depth/height metrics (optional ROI and mm_per_unit from config)
    roi = None
    mm_per_unit = None
    mm_per_pixel = None
    try:
        from config_loader import load_config
        cfg = load_config()
        mm_per_unit = cfg.get("mm_per_unit")
        mm_per_pixel = cfg.get("mm_per_pixel")
    except Exception:
        pass
    write_depth_metrics_file(
        depth,
        base / "depth_metrics.txt",
        roi=roi,
        mm_per_unit=mm_per_unit,
    )
    if mm_per_unit is not None:
        print(f"Saved metrics (with mm) to {base / 'depth_metrics.txt'}")
    else:
        print(f"Saved metrics to {base / 'depth_metrics.txt'}")


if __name__ == "__main__":
    main()
