"""
Optional: correct for camera viewing at an angle (not directly above).

When the camera cannot be placed directly above, this script warps the four gradient
images so the white sheet (reference plane) becomes fronto-parallel. The ENTIRE image
is warped—including the specimen—so you can still examine the specimen after.

What to select: Click the 4 corners of the WHITE SHEET (the flat base). The quad
should ENCLOSE the specimen so both sheet and specimen appear in the warped image.
Do NOT select only the specimen (it is not flat; the sheet defines the reference plane).

Usage:
  1. Put your 4 gradient images in test_images/ (grad_x_pos, grad_x_neg, grad_y_pos, grad_y_neg).
  2. Run:  python correct_perspective.py
  3. Click the 4 corners of the white sheet in order (e.g. top-left, top-right, bottom-right, bottom-left).
     Ensure the specimen is INSIDE this quad so it is included in the warped result.
  4. Close the window. Warped images (full scene: sheet + specimen) go to test_images_warped/.
  5. Run test_pipeline.py (it uses test_images_warped automatically if present).

Requires: opencv-python (cv2), numpy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print("This script requires opencv-python. Install with: pip install opencv-python")
    raise

SOURCE_DIR = Path("test_images")
OUTPUT_DIR = Path("test_images_warped")
IMAGE_NAMES = ["grad_x_pos", "grad_x_neg", "grad_y_pos", "grad_y_neg"]


def _find_image(base_name: str, img_dir: Path) -> Path:
    for name in (base_name, base_name.replace("grad_", "gard_")):
        for ext in (".png", ".jpg", ".jpeg"):
            p = img_dir / f"{name}{ext}"
            if p.exists():
                return p
    return img_dir / f"{base_name}.png"


def main():
    # Find first image to show for clicking
    first_path = None
    for name in IMAGE_NAMES:
        p = _find_image(name, SOURCE_DIR)
        if p.exists():
            first_path = p
            break
    if first_path is None:
        print(f"Could not find gradient images in {SOURCE_DIR}. Run after adding grad_* images.")
        return

    img = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {first_path}")
        return
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    clicks = []

    def onclick(event):
        if event.inaxes is None or event.button != 1:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        clicks.append((x, y))
        print(f"  Corner {len(clicks)}: ({x}, {y})")
        if len(clicks) <= 4:
            ax.plot(x, y, "o", color="lime", markersize=8, markeredgecolor="white", markeredgewidth=2)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    ax.set_title("Click 4 corners of the WHITE SHEET (must enclose the specimen), then close")
    ax.imshow(img_display)
    fig.canvas.mpl_connect("button_press_event", onclick)
    print("Select the WHITE SHEET: click its 4 corners in order (e.g. top-left, top-right, bottom-right, bottom-left).")
    print("The quad must ENCLOSE the specimen so both appear in the warped image. Then close the window.")
    plt.show()

    if len(clicks) != 4:
        print("Need exactly 4 clicks. Run again.")
        return

    src_pts = np.array(clicks, dtype=np.float32)
    # Destination: axis-aligned rectangle. Match each clicked point to nearest bbox corner for consistent order.
    x_min, y_min = src_pts.min(axis=0)
    x_max, y_max = src_pts.max(axis=0)
    w = max(x_max - x_min, 50)
    h = max(y_max - y_min, 50)
    dst_corners_xy = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    # Assign each clicked point to the destination corner it is closest to (tl, tr, br, bl)
    src_ordered = np.zeros_like(src_pts)
    used = [False] * 4
    for i in range(4):
        dists = np.array([np.linalg.norm(src_pts[j] - dst_corners_xy[i]) if not used[j] else 1e9 for j in range(4)])
        j = np.argmin(dists)
        used[j] = True
        src_ordered[i] = src_pts[j]
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H, _ = cv2.findHomography(src_ordered, dst_pts, method=cv2.RANSAC)

    if H is None:
        print("Failed to compute homography. Check that the 4 points form a valid quad.")
        return

    out_size = (int(round(w)), int(round(h)))
    OUTPUT_DIR.mkdir(exist_ok=True)

    for name in IMAGE_NAMES:
        path = _find_image(name, SOURCE_DIR)
        if not path.exists():
            continue
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        warped = cv2.warpPerspective(im, H, out_size)
        out_path = OUTPUT_DIR / path.name
        cv2.imwrite(str(out_path), warped)
        print(f"  Wrote {out_path}")

    print(f"\nDone. Warped images (full scene: white sheet + specimen) saved to {OUTPUT_DIR}/")
    print("Next: run test_pipeline.py — it will use test_images_warped automatically.")


if __name__ == "__main__":
    main()
