
import os
from typing import Literal, Optional, Tuple

import numpy as np

from Cameras import Webcam, MachineVision
from Projections import MainScreen
from CaptureSessions import GradientIlluminationCapture
from Calibrations import RadiometricCalibration, IntrinsicCalibration
from CalibrationsSessions import RadiometricCalibSession, IntrinsicCalibSession
from Reconstructions import GradientIlluminationReconstruction
from Visualization import Visualization
import Calibration

try:
    from config_loader import load_config, get_phone_cam_url
except ImportError:
    load_config = lambda: {}
    get_phone_cam_url = lambda: "http://192.168.1.100:8080/video"


# ---------------------------------------------------------------------------
# Configuration (overridden by config.json if present)
# ---------------------------------------------------------------------------

_def = load_config() if callable(load_config) else {}
CAMERA_TYPE: Literal["webcam", "basler", "phone"] = _def.get("camera_type", "webcam")
N_GRADIENT_IMAGES: int = int(_def.get("n_gradient_images", 2))
CAPTURE_NEW_DATA: bool = bool(_def.get("capture_new_data", True))


def _make_results_dir(path: str = "Results") -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _create_camera(camera_type: Literal["webcam", "basler", "phone"]):
    if camera_type == "webcam":
        cam = Webcam.Webcam()
    elif camera_type == "basler":
        cam = MachineVision.Basler()
    elif camera_type == "phone":
        url = get_phone_cam_url()
        print("Using phone camera at:", url)
        cam = Webcam.PhoneCam(url=url)
    else:
        raise ValueError(f"Unsupported CAMERA_TYPE '{camera_type}'. Use webcam, basler, or phone.")
    return cam


def _setup_calibration(cam) -> Calibration.Calibration:
    """
    Create a combined Calibration object.

    If radiometric calibration data is present on disk, it is loaded;
    otherwise the default gamma from RadiometricCalibration is used.
    Intrinsic calibration is created but not automatically captured here.
    """
    # Radiometric calibration (gamma correction, HDR, etc.)
    r_cal = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
    # Try to load existing radiometric calibration if available
    r_cal.load_calibration_data()

    # Intrinsic calibration (camera matrix, distortion). The capture process
    # is not run automatically here – you can run it separately via:
    #   IntrinsicCalibSession.IntrinsicCalibSession(cam, intr_calib).capture()
    intr_calib = IntrinsicCalibration.IntrinsicCalibration()

    calib = Calibration.Calibration(radio_calib=r_cal, intr_calib=intr_calib)
    return calib


def _write_depth_metrics(
    depth: np.ndarray,
    output_path: str = "Results/depth_metrics.txt",
    roi: Optional[Tuple[slice, slice]] = None,
) -> None:
    """
    Compute and save basic statistics of the depth map.

    Parameters
    ----------
    depth : np.ndarray
        2D depth map as returned by the reconstruction (arbitrary scale units).
    output_path : str
        Text file to write statistics into.
    roi : Optional[Tuple[slice, slice]]
        Optional region-of-interest (row_slice, col_slice). If provided,
        statistics are computed only on that region (e.g., around a lesion).
    """
    if depth is None:
        raise ValueError("Depth map is None. Make sure computePointCloud() was called.")

    if roi is not None:
        rows, cols = roi
        depth_roi = depth[rows, cols]
    else:
        depth_roi = depth

    d_min = float(np.nanmin(depth_roi))
    d_max = float(np.nanmax(depth_roi))
    d_mean = float(np.nanmean(depth_roi))
    d_std = float(np.nanstd(depth_roi))

    # Relative height map: subtract the minimum depth within ROI.
    # Positive values correspond to "raised" regions relative to the lowest point.
    relative_height = depth_roi - d_min
    h_max = float(np.nanmax(relative_height))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SkinScan depth/height statistics\n")
        f.write("--------------------------------\n")
        f.write(f"Depth shape: {depth.shape}\n")
        if roi is not None:
            f.write(f"ROI rows: {rows}, cols: {cols}\n")
        f.write(f"Depth min      : {d_min:.6f}\n")
        f.write(f"Depth max      : {d_max:.6f}\n")
        f.write(f"Depth mean     : {d_mean:.6f}\n")
        f.write(f"Depth std-dev  : {d_std:.6f}\n")
        f.write(f"Max height (relative to local min): {h_max:.6f}\n")


def run_skinscan_pipeline(
    capture_new: bool = CAPTURE_NEW_DATA,
    camera_type: Literal["webcam", "basler", "phone"] = CAMERA_TYPE,
) -> None:
    """
    Run the full SkinScan gradient-illumination pipeline:
    capture → normals → depth → metrics.

    Parameters
    ----------
    capture_new : bool
        If True, project gradient patterns and capture a new dataset.
        If False, only process the latest dataset in CapturedNumpyData/.
    camera_type : \"webcam\" or \"basler\"
        Select which camera implementation to use.
    """
    _make_results_dir("Results")

    # ------------------------------------------------------------------
    # 1. Camera and projector setup
    # ------------------------------------------------------------------
    cam = _create_camera(camera_type)
    projection = MainScreen.Screen()

    # ------------------------------------------------------------------
    # 2. Calibration objects (radiometric + intrinsic)
    # ------------------------------------------------------------------
    calib = _setup_calibration(cam)

    # ------------------------------------------------------------------
    # 3. Image processing (gradient-illumination reconstruction)
    # ------------------------------------------------------------------
    image_processing = GradientIlluminationReconstruction.GradientIlluminationReconstruction(
        n=N_GRADIENT_IMAGES
    )

    # ------------------------------------------------------------------
    # 4. Capture session
    # ------------------------------------------------------------------
    session = GradientIlluminationCapture.GradientIlluminationCapture(
        cam, projection, image_processing, n=N_GRADIENT_IMAGES
    )
    session.calibrate(calib)

    if capture_new:
        # Display gradient patterns on the screen and capture corresponding images.
        # Each capture is stored as a NumPy array in CapturedNumpyData/.
        session.capture(red=1.0, green=1.0, blue=1.0)

    # ------------------------------------------------------------------
    # 5. Reconstruction: normals, albedo, depth, mesh
    # ------------------------------------------------------------------
    session.compute()  # loads data, computes normals + albedo

    # Compute depth map and export mesh/texture; this also sets image_processing.depth
    image_processing.computePointCloud()

    # ------------------------------------------------------------------
    # 6. Depth/height statistics
    # ------------------------------------------------------------------
    _write_depth_metrics(image_processing.depth, output_path="Results/depth_metrics.txt")

    # Optional: quick visual inspection of intermediate results
    vis = Visualization(image_processing)
    vis.showAlbedo()
    vis.showNormals()
    # vis.showQuiverNormals(stride=30)  # uncomment for vector field view


if __name__ == "__main__":
    run_skinscan_pipeline()

