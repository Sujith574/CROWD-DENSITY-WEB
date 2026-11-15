"""
prepare_data.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Prepares training data for CSRNet crowd counting.

Mode A â€“ ShanghaiTech Part_B (real data):
  Expects the dataset at  data/ShanghaiTech/part_B/
    â”œâ”€â”€ train_data/images/*.jpg
    â”œâ”€â”€ train_data/ground-truth/GT_IMG_*.mat
    â”œâ”€â”€ test_data/images/*.jpg
    â””â”€â”€ test_data/ground-truth/GT_IMG_*.mat

  Converts each .mat annotation file â†’ Gaussian density map â†’ .h5 file
  saved alongside the image (same stem, .h5 extension).

Mode B â€“ Synthetic bootstrap (default if no dataset found):
  Generates N synthetic crowd images + exact ground-truth density maps
  so the full training + ONNX export pipeline runs immediately without
  requiring a Kaggle download.
"""

from __future__ import annotations

import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import h5py
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
SYNTH_TRAIN = DATA_DIR / "synthetic" / "train"
SYNTH_VAL   = DATA_DIR / "synthetic" / "val"
SHANGHAI_TRAIN = DATA_DIR / "ShanghaiTech" / "part_B" / "train_data"
SHANGHAI_VAL   = DATA_DIR / "ShanghaiTech" / "part_B" / "test_data"

IMG_H, IMG_W = 256, 256          # synthetic canvas size
SIGMA = 15                        # Gaussian kernel Ïƒ for density maps
N_TRAIN = 60
N_VAL   = 20

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def gaussian_density(points: np.ndarray, h: int, w: int, sigma: int = SIGMA) -> np.ndarray:
    """Return a normalised Gaussian density map given head-point annotations."""
    dm = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        dm[yi, xi] += 1.0
    dm = gaussian_filter(dm, sigma=sigma)
    return dm


# â”€â”€ Mode A: ShanghaiTech â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_shanghai_split(split_dir: Path) -> int:
    """Convert .mat GT files in a ShanghaiTech split to .h5 density maps."""
    try:
        import scipy.io as sio
    except ImportError:
        print("[ERROR] scipy not installed â€“ run pip install scipy")
        return 0

    img_dir = split_dir / "images"
    gt_dir  = split_dir / "ground-truth"

    if not img_dir.exists():
        print(f"[WARN] {img_dir} not found â€“ skipping ShanghaiTech processing.")
        return 0

    count = 0
    for mat_path in sorted(gt_dir.glob("GT_IMG_*.mat")):
        stem  = mat_path.stem.replace("GT_", "")        # e.g. IMG_1
        img_path = img_dir / f"{stem}.jpg"
        h5_path  = img_dir / f"{stem}.h5"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        mat   = sio.loadmat(str(mat_path))
        pts   = mat["image_info"][0, 0][0, 0][0]       # shape (N, 2)
        pts   = pts.astype(np.float32)

        dm = gaussian_density(pts, h, w, sigma=SIGMA)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("density", data=dm)
            f.create_dataset("count",   data=np.array([pts.shape[0]], dtype=np.float32))

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} images â€¦")

    print(f"  â†’ {count} .h5 density maps written.")
    return count


# â”€â”€ Mode B: Synthetic dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _crowd_scene(n_people: int, h: int = IMG_H, w: int = IMG_W,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a synthetic crowd image:
      - Coloured background (sky gradient)
      - n_people small ellipses (heads/bodies) placed in Gaussian clusters
    Returns (BGR image uint8, density map float32).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Background: sky-to-ground gradient
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        ratio = row / h
        r = int(100 + 155 * ratio)
        g = int(149 + 100 * ratio)
        b = int(237 - 100 * ratio)
        img[row, :] = (b, g, r)      # BGR

    # Crowd clusters
    n_clusters = max(1, rng.integers(2, 6))
    cx = rng.uniform(0.1, 0.9, n_clusters) * w
    cy = rng.uniform(0.4, 0.9, n_clusters) * h   # lower half = ground

    heads = []
    per_cluster = max(1, n_people // n_clusters)
    for i in range(n_clusters):
        for _ in range(per_cluster):
            px = int(np.clip(rng.normal(cx[i], w * 0.08), 2, w - 2))
            py = int(np.clip(rng.normal(cy[i], h * 0.08), 2, h - 2))
            heads.append((px, py))

            # Draw a person silhouette
            head_r = max(2, int(rng.integers(3, 7)))
            colour = tuple(int(c) for c in rng.integers(30, 220, 3))
            cv2.ellipse(img, (px, py), (head_r, int(head_r * 1.6)),
                        0, 0, 360, colour, -1)
            # Outline
            cv2.ellipse(img, (px, py), (head_r, int(head_r * 1.6)),
                        0, 0, 360, (20, 20, 20), 1)

    pts = np.array(heads, dtype=np.float32)
    dm  = gaussian_density(pts, h, w, sigma=SIGMA)
    return img, dm


def generate_synthetic_split(out_dir: Path, n: int, label: str) -> None:
    """Generate n synthetic crowd scenes and save image + .h5 density map."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42 if label == "train" else 7)

    for i in range(1, n + 1):
        n_people = int(rng.integers(10, 300))
        img, dm  = _crowd_scene(n_people, rng=rng)

        img_path = out_dir / "images" / f"IMG_{i}.jpg"
        h5_path  = out_dir / "images" / f"IMG_{i}.h5"

        cv2.imwrite(str(img_path), img)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("density", data=dm)
            f.create_dataset("count",   data=np.array([n_people], dtype=np.float32))

    print(f"  [Synthetic {label}] {n} scenes written to {out_dir}")


# â”€â”€ Entry-point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="Prepare crowd density data")
    parser.add_argument("--mode", choices=["auto", "synthetic", "shanghai"],
                        default="auto",
                        help="Data mode: auto=try shanghai then fallback to synthetic")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val",   type=int, default=N_VAL)
    args = parser.parse_args()

    mode = args.mode

    if mode == "auto":
        mode = "shanghai" if SHANGHAI_TRAIN.exists() else "synthetic"
        print(f"[prepare_data] Auto-detected mode: {mode}")

    if mode == "shanghai":
        print("Processing ShanghaiTech Part_B â€¦")
        t = process_shanghai_split(SHANGHAI_TRAIN)
        v = process_shanghai_split(SHANGHAI_VAL)
        if t == 0 and v == 0:
            print("[WARN] No ShanghaiTech images processed; falling back to synthetic.")
            mode = "synthetic"

    if mode == "synthetic":
        print(f"Generating synthetic dataset ({args.n_train} train / {args.n_val} val) â€¦")
        generate_synthetic_split(SYNTH_TRAIN, args.n_train, "train")
        generate_synthetic_split(SYNTH_VAL,   args.n_val,   "val")

    print("[prepare_data] Done âœ“")


if __name__ == "__main__":
    main()

