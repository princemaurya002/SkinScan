"""
Load SkinScan config from config.json (project root).
Falls back to defaults if file missing or keys absent.
"""
import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_DEFAULTS = {
    "camera_type": "webcam",
    "phone_cam_url": "http://192.168.1.100:8080/video",
    "n_gradient_images": 2,
    "capture_new_data": True,
    "mm_per_unit": None,
    "mm_per_pixel": None,
}


def load_config():
    out = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k in out:
                    out[k] = v
        except (json.JSONDecodeError, IOError) as e:
            print("Warning: could not load config.json:", e)
    return out


def get_phone_cam_url():
    cfg = load_config()
    url = cfg.get("phone_cam_url") or os.environ.get("SKINSCAN_PHONE_CAM_URL")
    return url or _DEFAULTS["phone_cam_url"]
