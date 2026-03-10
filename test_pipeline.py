import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Shadow & noise reduction (tweak here or pass into compute_normals_and_depth)
# ---------------------------------------------------------------------------
DEFAULT_SHADOW_PERCENTILE = 10.0   # Pixels darker than this % of ref -> shadow (flat normal)
DEFAULT_SHADOW_ABS_THRESHOLD = 0.0  # Deprecated: kept for API compatibility, ignored if <= 0
DEFAULT_SMOOTH_DIFFS = 5            # Gaussian blur on differentials (odd)
DEFAULT_CLIP_DIFFS = 0.92           # Clip |diff_x|,|diff_y| to reduce spikes
DEFAULT_SPECULAR_THRESHOLD = 0.95   # ref > this: specular highlight, mask gradient
DEFAULT_INPUT_DENOISE = 3           # Gaussian on each of 4 images (0 = off)
DEFAULT_DEPTH_BILATERAL_D = 5       # Bilateral filter on depth (0 = off)
DEFAULT_DEPTH_BILATERAL_SIGMA = 50.0
DEFAULT_DEPTH_MEDIAN_SIZE = 3       # Median filter on depth after bilateral (0 = off, 3 or 5 = size)
DEFAULT_DEPTH_INPAINT_SHADOW = True # Fill depth at shadow pixels by inpainting
DEFAULT_VALIDATE_GRADIENTS = True   # Print gradient quality stats and warn if poor


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Simple Frankot–Chellappa integration implementation.
    p ~ ∂z/∂x, q ~ ∂z/∂y

    Returns:
        depth (real 2D array)
    """
    # Ensure float
    p = p.astype(np.float64)
    q = q.astype(np.float64)

    rows, cols = p.shape

    # Frequency grids
    wx = np.fft.fftfreq(cols) * 2.0 * np.pi
    wy = np.fft.fftfreq(rows) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    # Fourier transforms of p and q
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    # Denominator: wx^2 + wy^2  (avoid division by 0)
    denom = wx**2 + wy**2
    denom[denom == 0] = 1.0  # arbitrary, DC component handled below

    # Frankot–Chellappa in frequency domain
    Z = (-1j * wx * P - 1j * wy * Q) / denom

    # Set DC to 0 (unknown absolute offset)
    Z[0, 0] = 0.0

    # Back to spatial domain
    z = np.fft.ifft2(Z).real
    return z


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    return img

def center_crop_to_common_size(images):
    """
    Center-crop a list of 2D images to the same (min_height, min_width).
    Returns a list of cropped images.
    """
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    h_min = min(heights)
    w_min = min(widths)

    cropped = []
    for img in images:
        h, w = img.shape[:2]
        top = (h - h_min) // 2
        left = (w - w_min) // 2
        cropped_img = img[top:top + h_min, left:left + w_min]
        cropped.append(cropped_img)
    return cropped


def compute_normals_and_depth(
    ix_pos: np.ndarray,
    ix_neg: np.ndarray,
    iy_pos: np.ndarray,
    iy_neg: np.ndarray,
    gamma: float = 1.0,
    reduce_shadow: bool = True,
    shadow_percentile: float = 10.0,
    shadow_abs_threshold: float = 0.0,
    smooth_diffs: int = 5,
    clip_diffs: float = 0.92,
    specular_threshold: float = 0.95,
    input_denoise: int = 3,
    depth_bilateral_d: int = 5,
    depth_bilateral_sigma: float = 50.0,
    depth_median_size: int = 0,
    depth_inpaint_shadow: bool = True,
    validate_gradients: bool = True,
):
    """
    Uses I_const normalization: diff = (I_pos - I_neg)/(I_const+eps). Shadow and specular masking.
    shadow_percentile: pixels darker than this percentile of ref are shadow.
    shadow_abs_threshold: pixels with ref < this are always shadow (removes deep shadows).
    smooth_diffs: Gaussian kernel size for diff_x/diff_y (0 = no smoothing).
    clip_diffs: clip |diff_x|,|diff_y| to this before normals (reduces shadow-edge spikes).
    input_denoise: Gaussian blur kernel size on each of the 4 images (0 = no denoise).
    depth_bilateral_d: diameter for bilateral filter on depth (0 = skip).
    depth_inpaint_shadow: fill depth at shadow pixels by inpainting to avoid holes.
    """
    eps = 1e-6

    # 0) Optional input denoising to reduce sensor/camera noise
    if input_denoise >= 3 and input_denoise % 2 == 1:
        k = input_denoise
        ix_pos = cv2.GaussianBlur(ix_pos.astype(np.float32), (k, k), 0)
        ix_neg = cv2.GaussianBlur(ix_neg.astype(np.float32), (k, k), 0)
        iy_pos = cv2.GaussianBlur(iy_pos.astype(np.float32), (k, k), 0)
        iy_neg = cv2.GaussianBlur(iy_neg.astype(np.float32), (k, k), 0)

    # 1) Gamma correction (approximate linearization)
    ix_pos_lin = ix_pos.astype(np.float64) ** gamma
    ix_neg_lin = ix_neg.astype(np.float64) ** gamma
    iy_pos_lin = iy_pos.astype(np.float64) ** gamma
    iy_neg_lin = iy_neg.astype(np.float64) ** gamma

    # 2) I_const = constant illumination (avg of 4) – standard photometric stereo normalization
    ref = (ix_pos_lin + ix_neg_lin + iy_pos_lin + iy_neg_lin) / 4.0
    I_const = np.clip(ref, eps, None)
    I_const = np.minimum(I_const, specular_threshold)  # clamp specular for stable denominator
    # diff = (I_pos - I_neg) / (I_const + eps) removes albedo/camera response variation
    diff_x = (ix_pos_lin - ix_neg_lin) / (I_const + eps)
    diff_y = (iy_pos_lin - iy_neg_lin) / (I_const + eps)
    max_val = max(float(np.nanmax(np.abs(diff_x))), float(np.nanmax(np.abs(diff_y))), 1e-8)
    diff_x = diff_x / max_val
    diff_y = diff_y / max_val

    # 5) Shadow mask: low intensity -> bad normals (computed from image statistics)
    shadow_mask = np.zeros(ref.shape, dtype=bool)
    if reduce_shadow:
        # Data-driven shadow threshold: based purely on image statistics (percentile of ref).
        # shadow_abs_threshold is ignored when <= 0 and kept only for backward compatibility.
        if shadow_percentile > 0:
            thresh = float(np.percentile(ref, shadow_percentile))
        else:
            # Fallback to default percentile if user disables it
            thresh = float(np.percentile(ref, DEFAULT_SHADOW_PERCENTILE))
        thresh = max(thresh, eps)
        shadow_mask = ref < thresh
        diff_x = np.where(shadow_mask, 0.0, diff_x)
        diff_y = np.where(shadow_mask, 0.0, diff_y)

    # 5b) Specular mask: very bright pixels break Lambertian assumption
    spec_mask = ref < specular_threshold
    diff_x = np.where(spec_mask, diff_x, 0.0)
    diff_y = np.where(spec_mask, diff_y, 0.0)

    # 5c) Clip extreme differentials (shadow edges / residual speculars)
    if clip_diffs > 0 and clip_diffs < 1:
        diff_x = np.clip(diff_x, -clip_diffs, clip_diffs)
        diff_y = np.clip(diff_y, -clip_diffs, clip_diffs)

    # 6) Smooth differentials to reduce noise and shadow-edge artifacts
    if smooth_diffs >= 3 and smooth_diffs % 2 == 1:
        k = smooth_diffs
        diff_x = cv2.GaussianBlur(diff_x.astype(np.float32), (k, k), 0)
        diff_y = cv2.GaussianBlur(diff_y.astype(np.float32), (k, k), 0)

    # 7) Compute normals
    z_sq = 1.0 - diff_x**2 - diff_y**2
    z_sq = np.clip(z_sq, a_min=1e-8, a_max=None)
    z = np.sqrt(z_sq)

    norm = np.sqrt(diff_x**2 + diff_y**2 + z**2) + 1e-8
    nx = diff_x / norm
    ny = diff_y / norm
    nz = z / norm

    normals = np.stack([nx, ny, nz], axis=2)

    # 8) Depth via Frankot–Chellappa integration
    p = nx / (nz + 1e-8)  # ∂z/∂x
    q = ny / (nz + 1e-8)  # ∂z/∂y
    depth = frankot_chellappa(p, q)

    # 9) Optional bilateral filter on depth (smooths noise, preserves edges)
    if depth_bilateral_d >= 3 and depth_bilateral_d % 2 == 1:
        depth = cv2.bilateralFilter(
            depth.astype(np.float32),
            depth_bilateral_d,
            depth_bilateral_sigma,
            depth_bilateral_sigma,
        )

    # 9b) Optional median filter on depth (further regularization, reduces outliers)
    if depth_median_size >= 3 and depth_median_size % 2 == 1:
        try:
            from scipy.ndimage import median_filter
            depth = median_filter(depth.astype(np.float64), size=depth_median_size, mode="nearest")
        except ImportError:
            pass

    # 10) Gradient quality validation (warn if capture quality is poor)
    if validate_gradients:
        mag = np.sqrt(diff_x.astype(np.float64)**2 + diff_y.astype(np.float64)**2)
        mean_mag = float(np.nanmean(mag))
        shadow_pct = 100.0 * float(np.sum(shadow_mask)) / max(ref.size, 1)
        spec_pct = 100.0 * float(np.sum(~spec_mask)) / max(ref.size, 1)
        print("Gradient quality: mean |grad|={:.4f}  shadow%={:.1f}  specular%={:.1f}".format(mean_mag, shadow_pct, spec_pct))
        if mean_mag < 0.05:
            print("  ^ Low gradient strength – check illumination or use a more textured surface.")
        if shadow_pct > 30:
            print("  ^ High shadow % – consider better lighting or increase shadow_percentile.")

    # 11) Inpaint shadow regions in depth so they don't leave holes or wrong values
    if reduce_shadow and depth_inpaint_shadow and shadow_mask.any():
        # Inpaint only where shadow was: use neighborhood to fill
        mask_u8 = (shadow_mask.astype(np.uint8)) * 255
        depth_inpainted = cv2.inpaint(
            depth.astype(np.float32),
            mask_u8,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA,
        )
        depth = depth_inpainted

    return normals, depth


def save_normals(normals: np.ndarray, out_path: Path):
    # Map from [-1,1] to [0,255] and save as RGB
    normals_vis = (normals + 1.0) / 2.0
    normals_vis = (np.clip(normals_vis, 0, 1) * 255).astype(np.uint8)
    normals_bgr = cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), normals_bgr)


def save_depth(depth: np.ndarray, out_path: Path):
    d = depth - np.nanmin(depth)
    if np.nanmax(d) > 0:
        d /= np.nanmax(d)
    d_vis = (d * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), d_vis)

def save_albedo_from_frames(ix_pos: np.ndarray, ix_neg: np.ndarray, iy_pos: np.ndarray, iy_neg: np.ndarray, out_path: Path):
    """
    Create an albedo-like (normal-looking) image by averaging the 4 input frames.
    Since these are gradient-illuminated, this is not perfect albedo but is a good
    natural background for clicking/querying points.
    """
    alb = (ix_pos + ix_neg + iy_pos + iy_neg) / 4.0
    alb = np.clip(alb, 0.0, 1.0)
    alb_u8 = (alb * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), alb_u8)


def print_depth_stats(depth: np.ndarray):
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    d_mean = float(np.nanmean(depth))
    d_std = float(np.nanstd(depth))
    h_rel = depth - d_min
    h_max = float(np.nanmax(h_rel))
    print("Depth stats (arbitrary units):")
    print(f"  min depth    : {d_min:.6f}")
    print(f"  max depth    : {d_max:.6f}")
    print(f"  mean depth   : {d_mean:.6f}")
    print(f"  std depth    : {d_std:.6f}")
    print(f"  max height (relative to min): {h_max:.6f}")


def main():
    import sys
    WARPED_DIR = Path("test_images_warped")
    SOURCE_DIR = Path("test_images")

    def _has_gradients(d: Path) -> bool:
        for n in ("grad_x_pos", "grad_x_neg", "grad_y_pos", "grad_y_neg"):
            found = any((d / f"{n}{e}").exists() for e in (".png", ".jpg", ".jpeg"))
            if not found:
                found = any((d / f"{n.replace('grad_','gard_')}{e}").exists() for e in (".png", ".jpg", ".jpeg"))
            if not found:
                return False
        return True

    # --warped: force use of perspective-corrected images
    use_warped = "--warped" in sys.argv or "-w" in sys.argv
    if use_warped:
        if not WARPED_DIR.exists():
            raise FileNotFoundError(
                f"Folder {WARPED_DIR} not found. Run correct_perspective.py first."
            )
        if not _has_gradients(WARPED_DIR):
            raise FileNotFoundError(
                f"Folder {WARPED_DIR} exists but does not contain all 4 gradient images "
                "(grad_x_pos, grad_x_neg, grad_y_pos, grad_y_neg). Run correct_perspective.py first."
            )
        img_dir = WARPED_DIR
    else:
        # Auto: use warped if present and complete; otherwise test_images
        if WARPED_DIR.exists() and _has_gradients(WARPED_DIR):
            img_dir = WARPED_DIR
        else:
            img_dir = SOURCE_DIR

    def _find_image(base_name: str) -> Path:
        for name in (base_name, base_name.replace("grad_", "gard_")):
            for ext in (".png", ".jpg", ".jpeg"):
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    return p
        return img_dir / f"{base_name}.png"  # raise with clear path

    ix_pos_path = _find_image("grad_x_pos")
    ix_neg_path = _find_image("grad_x_neg")
    iy_pos_path = _find_image("grad_y_pos")
    iy_neg_path = _find_image("grad_y_neg")

    if not ix_pos_path.exists():
        raise FileNotFoundError(
            f"Could not find gradient images in {img_dir}. "
            "Add grad_x_pos, grad_x_neg, grad_y_pos, grad_y_neg as .png, .jpg, or .jpeg"
        )

    if img_dir == WARPED_DIR:
        print("Using test_images_warped (perspective-corrected images).")
    ix_pos = load_gray(ix_pos_path)
    ix_neg = load_gray(ix_neg_path)
    iy_pos = load_gray(iy_pos_path)
    iy_neg = load_gray(iy_neg_path)

    # Ensure all images have exactly the same size via center-crop
    ix_pos, ix_neg, iy_pos, iy_neg = center_crop_to_common_size(
        [ix_pos, ix_neg, iy_pos, iy_neg]
    )

    # Simple gamma (you can tweak, e.g. 0.57 to approximate camera response)
    gamma = 1.0

    normals, depth = compute_normals_and_depth(
        ix_pos, ix_neg, iy_pos, iy_neg,
        gamma=gamma,
        reduce_shadow=True,
        shadow_percentile=DEFAULT_SHADOW_PERCENTILE,
        shadow_abs_threshold=DEFAULT_SHADOW_ABS_THRESHOLD,
        smooth_diffs=DEFAULT_SMOOTH_DIFFS,
        clip_diffs=DEFAULT_CLIP_DIFFS,
        specular_threshold=DEFAULT_SPECULAR_THRESHOLD,
        input_denoise=DEFAULT_INPUT_DENOISE,
        depth_bilateral_d=DEFAULT_DEPTH_BILATERAL_D,
        depth_bilateral_sigma=DEFAULT_DEPTH_BILATERAL_SIGMA,
        depth_median_size=DEFAULT_DEPTH_MEDIAN_SIZE,
        depth_inpaint_shadow=DEFAULT_DEPTH_INPAINT_SHADOW,
        validate_gradients=DEFAULT_VALIDATE_GRADIENTS,
    )

    out_dir = Path("Results_test")
    out_dir.mkdir(exist_ok=True)

    save_normals(normals, out_dir / "normals_test.png")
    save_depth(depth, out_dir / "depth_test.png")
    # Save a natural-looking background for point selection
    save_albedo_from_frames(ix_pos, ix_neg, iy_pos, iy_neg, out_dir / "albedo_test.png")
    # Save raw depth (before normalization) for mesh export and metric height
    np.save(out_dir / "depth_raw.npy", depth)
    np.save(out_dir / "normals_raw.npy", normals)
    print_depth_stats(depth)
    print(f"Saved raw depth and normals to {out_dir}. Run: python make_mesh_from_depth.py")


if __name__ == "__main__":
    main()