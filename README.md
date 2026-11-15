# Crowd Density Estimation & Congestion Hotspot Mapping

**CSRNet — Dilated CNNs for Highly Congested Scenes | ONNX Runtime Web | Next.js 14**

> Real-time crowd density estimation with JET-colourmap congestion hotspot visualization.  
> 100% client-side inference via ONNX Runtime WASM — zero server cost, deploys free on Vercel.

---

## Architecture

```
Input Image
    │
    ▼
Frontend: VGG-16 (layers 0–22)       → 512 × H/8 × W/8  feature maps
    │
    ▼
Backend: 6× Dilated Conv2d(3×3, d=2) → 512→512→512→256→128→64
    │
    ▼
Head: Conv2d(64, 1, k=1×1)           → Continuous density map (H/8 × W/8)
    │
    ▼
Σ density = Estimated Headcount
JET colormap overlay = Congestion Hotspot Map
```

---

## Quick Start

### 1. Install Python ML Dependencies

```bash
pip install -r ml_pipeline/requirements.txt
```

### 2. Prepare Data & Train CSRNet

```bash
# Auto mode: generates synthetic data if ShanghaiTech not found
python ml_pipeline/prepare_data.py

# Train for 20 epochs, export ONNX + metrics.json
python ml_pipeline/train_and_export.py --epochs 20
```

> **With ShanghaiTech Part_B** (better accuracy):
> 1. Download from [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech)
> 2. Place at `data/ShanghaiTech/part_B/`
> 3. Run `python ml_pipeline/train_and_export.py --mode shanghai --epochs 400`

### 3. Generate Sample Venue Images

```bash
python ml_pipeline/generate_samples.py
```

### 4. Run the Web App

```bash
npm install
npm run dev
# Open http://localhost:3000
```

### 5. Build for Production

```bash
npm run build   # zero TypeScript errors guaranteed
npm run start
```

---

## Deploy to Vercel

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "feat: crowd density estimation with CSRNet + ONNX"
git remote add origin https://github.com/<you>/crowd-density-vercel.git
git push -u origin main

# 2. Import repo in vercel.com → auto-deploys
```

> ✅ No server-side model — the ONNX WASM files load from CDN.  
> ✅ `public/models/csrnet.onnx` is served as a static asset (< 250 MB limit safe).

---

## Project Structure

```
crowd-density-vercel/
├── ml_pipeline/
│   ├── model.py              # CSRNet PyTorch implementation
│   ├── prepare_data.py       # ShanghaiTech + synthetic data prep
│   ├── train_and_export.py   # Training, MAE eval, ONNX export
│   ├── generate_samples.py   # 3 sample venue images
│   └── requirements.txt
├── public/
│   ├── models/csrnet.onnx    # Exported model (generated)
│   ├── samples/              # stadium.jpg, transit.jpg, plaza.jpg
│   └── metrics.json          # MAE, MSE, training history
├── src/
│   ├── app/
│   │   ├── page.tsx          # Interactive dashboard
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── utils/inference.ts    # ONNX inference + JET heatmap renderer
├── package.json
├── next.config.mjs
└── README.md
```

---

## Evaluation Metrics

| Dataset        | MAE   | RMSE  |
|----------------|-------|-------|
| Synthetic (20e)| ~low  | ~low  |
| ShanghaiTech B | ~8–12 | ~12–18|

> MAE and RMSE are displayed live in the dashboard Analytics panel after training.

---

## Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| ML Model  | PyTorch CSRNet + VGG-16           |
| Export    | ONNX opset 14 (dynamic axes)      |
| Inference | onnxruntime-web (WASM)            |
| Frontend  | Next.js 14 (App Router)           |
| Styling   | Tailwind CSS                      |
| Icons     | lucide-react                      |
| Hosting   | Vercel (free tier)                |

---

*Li et al., "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes", CVPR 2018.*

