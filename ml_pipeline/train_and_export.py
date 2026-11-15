"""
train_and_export.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
1. Trains CSRNet on the prepared dataset (synthetic or ShanghaiTech).
2. Evaluates MAE / MSE on the validation split.
3. Exports the trained model to ONNX (opset 14, dynamic H/W axes).
4. Saves training metrics to public/metrics.json for the web dashboard.

Usage:
    python ml_pipeline/train_and_export.py [--epochs N] [--lr LR] [--mode auto|synthetic|shanghai]
"""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import cv2
import h5py

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml_pipeline"))
from model import CSRNet

DATA_DIR    = ROOT / "data"
PUBLIC_DIR  = ROOT / "public"
MODEL_DIR   = PUBLIC_DIR / "models"
ONNX_PATH   = MODEL_DIR / "csrnet.onnx"
METRICS_PATH = PUBLIC_DIR / "metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class CrowdDataset(Dataset):
    """Loads image + pre-computed .h5 density map pairs."""

    IMG_SIZE = (256, 256)   # resize target

    def __init__(self, img_dir: Path):
        self.samples = sorted(img_dir.glob("*.jpg"))
        if not self.samples:
            raise FileNotFoundError(f"No .jpg images in {img_dir}")
        print(f"  Dataset: {len(self.samples)} images in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        h5_path  = img_path.with_suffix(".h5")

        # â”€â”€ Image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))  # (W, H)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        for c in range(3):
            img_t[c] = (img_t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # â”€â”€ Density map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with h5py.File(h5_path, "r") as f:
            dm = f["density"][:]          # (H_orig, W_orig)

        # Resize to 1/8 of model input (32Ã—32 for 256Ã—256 input)
        dm_h = self.IMG_SIZE[0] // 8
        dm_w = self.IMG_SIZE[1] // 8
        dm_small = cv2.resize(dm, (dm_w, dm_h), interpolation=cv2.INTER_LINEAR)
        # Preserve total count: sum(dm_small) Ã— 64 â‰ˆ sum(dm)
        dm_small = dm_small * (dm.sum() / (dm_small.sum() + 1e-8)) / 64.0 * dm_small.sum() / (dm_small.sum() + 1e-8)
        # Simpler: scale factor
        scale = (dm.sum() + 1e-8) / (dm_small.sum() * 64 + 1e-8)
        dm_small = dm_small * scale

        dm_t = torch.from_numpy(dm_small).float().unsqueeze(0)  # (1, 32, 32)
        return img_t, dm_t


def find_img_dir(mode: str, split: str) -> Path:
    if mode == "synthetic":
        base = DATA_DIR / "synthetic" / ("train" if split == "train" else "val")
    else:
        part = "train_data" if split == "train" else "test_data"
        base = DATA_DIR / "ShanghaiTech" / "part_B" / part
    return base / "images"


# â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, dms in loader:
        imgs, dms = imgs.to(device), dms.to(device)
        optimiser.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, dms)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_total, mse_total, n = 0.0, 0.0, 0
    for imgs, dms in loader:
        imgs = imgs.to(device)
        pred = model(imgs)                                   # BÃ—1Ã—H/8Ã—W/8
        pred_count = pred.sum(dim=(1, 2, 3)).cpu()           # sum â†’ headcount
        gt_count   = dms.sum(dim=(1, 2, 3)) * 64            # restore count
        diff = torch.abs(pred_count - gt_count)
        mae_total += diff.sum().item()
        mse_total += (diff ** 2).sum().item()
        n += imgs.size(0)
    mae = mae_total / max(n, 1)
    mse = (mse_total / max(n, 1)) ** 0.5
    return mae, mse


# â”€â”€ ONNX Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def export_onnx(model: nn.Module, out_path: Path, img_size: tuple[int, int] = (256, 256)):
    model.eval()
    dummy = torch.randn(1, 3, img_size[0], img_size[1])
    torch.onnx.export(
        model.cpu(),
        dummy,
        str(out_path),
        opset_version=14,
        input_names=["input"],
        output_names=["density_map"],
        dynamic_axes={
            "input":       {0: "batch", 2: "height", 3: "width"},
            "density_map": {0: "batch", 2: "out_h",  3: "out_w"},
        },
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ONNX] Exported â†’ {out_path}  ({size_mb:.1f} MB)")
    return size_mb


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=1e-5)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--mode",     choices=["auto", "synthetic", "shanghai"], default="auto")
    parser.add_argument("--workers",  type=int,   default=0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    args = parser.parse_args()

    # â”€â”€ Resolve data mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mode = args.mode
    if mode == "auto":
        shanghai_ok = (DATA_DIR / "ShanghaiTech" / "part_B" / "train_data" / "images").exists()
        mode = "shanghai" if shanghai_ok else "synthetic"
        print(f"[Mode] Auto-detected: {mode}")

    # Ensure data exists
    synth_train_dir = DATA_DIR / "synthetic" / "train" / "images"
    if mode == "synthetic" and not synth_train_dir.exists():
        print("[Auto] Running prepare_data.py first â€¦")
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "ml_pipeline" / "prepare_data.py"), "--mode", "synthetic"],
            check=True
        )

    # â”€â”€ Build datasets / loaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_dir = find_img_dir(mode, "train")
    val_dir   = find_img_dir(mode, "val")

    train_ds = CrowdDataset(train_dir)
    val_ds   = CrowdDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)

    # â”€â”€ Model / optimiser / loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    device    = get_device()
    model     = CSRNet(pretrained=args.pretrained).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="sum")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CSRNet â€” {n_params:,} parameters")

    # â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    history  = []
    best_mae = float("inf")
    best_state = None

    print(f"\nTraining for {args.epochs} epochs â€¦\n{'â”€'*50}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        mae, mse = evaluate(model, val_loader, device)

        history.append({"epoch": epoch, "loss": round(loss, 6),
                         "mae": round(mae, 4), "mse": round(mse, 4)})

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        flag = " â˜…" if mae == best_mae else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
              f"MAE={mae:.2f}  RMSE={mse:.2f}{flag}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  best MAE={best_mae:.2f}")

    # â”€â”€ Restore best checkpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to("cpu")

    # Final evaluation
    model_cpu = model.to("cpu")
    # Create a small cpu loader for final eval
    val_ds_cpu = CrowdDataset(val_dir)
    val_loader_cpu = DataLoader(val_ds_cpu, batch_size=1, shuffle=False, num_workers=0)
    final_mae, final_mse = evaluate(model_cpu, val_loader_cpu, torch.device("cpu"))
    print(f"[Final] MAE={final_mae:.2f}  RMSE={final_mse:.2f}")

    # â”€â”€ ONNX Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    onnx_mb = export_onnx(model_cpu, ONNX_PATH)

    # â”€â”€ Save metrics.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics = {
        "model": "CSRNet",
        "architecture": {
            "frontend": "VGG-16 (first 10 conv layers, up to pool3, 1/8 spatial resolution)",
            "backend": "6Ã— Dilated Conv2d(3Ã—3, dilation=2) â€” 512â†’512â†’512â†’256â†’128â†’64",
            "head": "Conv2d(64, 1, kernel=1Ã—1) â€” continuous density map",
            "total_parameters": n_params,
        },
        "training": {
            "dataset": mode,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch,
            "loss_fn": "MSELoss(reduction='sum')",
            "optimiser": "Adam",
            "training_time_s": round(elapsed, 1),
        },
        "evaluation": {
            "mae": round(final_mae, 4),
            "rmse": round(final_mse, 4),
        },
        "export": {
            "format": "ONNX",
            "opset": 14,
            "size_mb": round(onnx_mb, 1),
            "dynamic_axes": ["batch", "height", "width"],
        },
        "loss_history": history,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[Metrics] Saved â†’ {METRICS_PATH}")
    print("\nâœ“ All done! Files ready:")
    print(f"  {ONNX_PATH}")
    print(f"  {METRICS_PATH}")


if __name__ == "__main__":
    main()

