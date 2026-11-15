/**
 * inference.ts
 * ─────────────
 * Client-side crowd density inference using ONNX Runtime Web via CDN script tag.
 *
 * Pipeline:
 *  1. Load /models/csrnet.onnx using window.ort.InferenceSession.
 *  2. Preprocess an HTMLImageElement → Float32Array tensor (ImageNet normalised).
 *  3. Run inference → density map tensor (H/8 × W/8).
 *  4. Sum density map → estimated headcount.
 *  5. Render JET-colourmap heatmap blended over source image on a canvas.
 */

// ── ImageNet normalisation constants ──────────────────────────────────────
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// Max dimension to resize the input image before inference (perf vs accuracy)
const MAX_DIM = 640;

// ── Module-level session cache ────────────────────────────────────────────
let _session: any = null;
let _loading = false;
let _loadQueue: Array<(s: any) => void> = [];

declare global {
  interface Window {
    ort?: any;
  }
}

/** Helper to ensure window.ort is fully initialized from CDN */
async function waitForOrt(): Promise<any> {
  if (typeof window === 'undefined') {
    throw new Error('ONNX inference only supported in the browser.');
  }

  if (window.ort) {
    window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
    window.ort.env.wasm.numThreads = 1;
    return window.ort;
  }

  // Poll for window.ort in case script is still loading
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const interval = setInterval(() => {
      attempts++;
      if (window.ort) {
        clearInterval(interval);
        window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
        window.ort.env.wasm.numThreads = 1;
        resolve(window.ort);
      } else if (attempts > 100) {
        clearInterval(interval);
        reject(new Error('Timed out waiting for ONNX Runtime Web to load from CDN.'));
      }
    }, 100);
  });
}

export async function getSession(): Promise<any> {
  if (_session) return _session;

  if (_loading) {
    return new Promise((resolve) => {
      _loadQueue.push(resolve);
    });
  }

  _loading = true;
  try {
    const ort = await waitForOrt();
    const session = await ort.InferenceSession.create('/models/csrnet.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    _session = session;
    _loadQueue.forEach((cb) => cb(session));
    _loadQueue = [];
    return session;
  } finally {
    _loading = false;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/** Resize image maintaining aspect ratio so max dimension ≤ maxDim. */
function resizeImage(
  img: HTMLImageElement,
  maxDim: number
): { canvas: HTMLCanvasElement; w: number; h: number } {
  let { naturalWidth: w, naturalHeight: h } = img;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  w = Math.round(w * scale);
  h = Math.round(h * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  return { canvas, w, h };
}

/** Convert canvas pixel data to a normalised Float32Array CHW tensor. */
async function canvasToTensor(canvas: HTMLCanvasElement, w: number, h: number): Promise<any> {
  const ort = await waitForOrt();
  const ctx = canvas.getContext('2d')!;
  const { data } = ctx.getImageData(0, 0, w, h); // RGBA uint8

  const tensor = new Float32Array(3 * h * w);
  for (let i = 0; i < h * w; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensor[i] = (r - MEAN[0]) / STD[0]; // R channel
    tensor[h * w + i] = (g - MEAN[1]) / STD[1]; // G channel
    tensor[2 * h * w + i] = (b - MEAN[2]) / STD[2]; // B channel
  }
  return new ort.Tensor('float32', tensor, [1, 3, h, w]);
}

// ── JET colour-map ────────────────────────────────────────────────────────

function jetColor(t: number): [number, number, number] {
  // t ∈ [0, 1] → (R, G, B) ∈ [0, 255]
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const r = clamp(1.5 - Math.abs(4 * t - 3));
  const g = clamp(1.5 - Math.abs(4 * t - 2));
  const b = clamp(1.5 - Math.abs(4 * t - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ── Bilinear upsample density map ─────────────────────────────────────────

function bilinearUpsample(
  src: Float32Array,
  srcH: number,
  srcW: number,
  dstH: number,
  dstW: number
): Float32Array {
  const dst = new Float32Array(dstH * dstW);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = (x / (dstW - 1)) * (srcW - 1);
      const sy = (y / (dstH - 1)) * (srcH - 1);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, srcW - 1);
      const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, srcH - 1);
      const wx = sx - x0, wy = sy - y0;
      dst[y * dstW + x] =
        (1 - wy) * ((1 - wx) * src[y0 * srcW + x0] + wx * src[y0 * srcW + x1]) +
        wy * ((1 - wx) * src[y1 * srcW + x0] + wx * src[y1 * srcW + x1]);
    }
  }
  return dst;
}

// ── Public API ────────────────────────────────────────────────────────────

export interface InferenceResult {
  headcount: number;
  peakDensity: number; // max cell value (before normalisation)
  inferenceMs: number;
  densityMap: Float32Array; // raw density map (dmH × dmW)
  dmH: number;
  dmW: number;
  imgW: number;
  imgH: number;
}

/**
 * Run CSRNet inference on an HTMLImageElement.
 * Returns headcount, density map, and timings.
 */
export async function runInference(img: HTMLImageElement): Promise<InferenceResult> {
  const session = await getSession();

  const { canvas, w, h } = resizeImage(img, MAX_DIM);
  const inputTensor = await canvasToTensor(canvas, w, h);

  const t0 = performance.now();
  const feeds: Record<string, any> = { input: inputTensor };
  const results = await session.run(feeds);
  const inferenceMs = performance.now() - t0;

  const output = results['density_map'];
  const dmData = output.data as Float32Array;
  const [, , dmH, dmW] = output.dims as [number, number, number, number];

  const headcount = dmData.reduce((s: number, v: number) => s + v, 0);
  let peakDensity = 0;
  for (let i = 0; i < dmData.length; i++) {
    if (dmData[i] > peakDensity) peakDensity = dmData[i];
  }

  return {
    headcount: Math.max(0, Math.round(headcount)),
    peakDensity,
    inferenceMs: Math.round(inferenceMs),
    densityMap: dmData,
    dmH,
    dmW,
    imgW: w,
    imgH: h,
  };
}

/**
 * Render a JET heatmap blended over the original image onto a canvas.
 */
export function renderHeatmap(
  result: InferenceResult,
  sourceImg: HTMLImageElement,
  canvas: HTMLCanvasElement,
  opacity: number,
  viewMode: 'overlay' | 'heatmap' | 'original'
): void {
  const { densityMap, dmH, dmW, imgW, imgH } = result;
  const ctx = canvas.getContext('2d')!;

  canvas.width = imgW;
  canvas.height = imgH;

  // ── Draw original image ──────────────────────────────────────────
  if (viewMode === 'original') {
    ctx.drawImage(sourceImg, 0, 0, imgW, imgH);
    return;
  }

  // ── Upsample density map to image size ───────────────────────────
  const upsampled = bilinearUpsample(densityMap, dmH, dmW, imgH, imgW);

  // ── Normalise to [0, 1] ──────────────────────────────────────────
  let maxVal = 1e-8;
  for (let i = 0; i < upsampled.length; i++) {
    if (upsampled[i] > maxVal) maxVal = upsampled[i];
  }
  const norm = new Float32Array(upsampled.length);
  for (let i = 0; i < upsampled.length; i++) {
    norm[i] = upsampled[i] / maxVal;
  }

  // ── Build heatmap ImageData ──────────────────────────────────────
  const heatData = ctx.createImageData(imgW, imgH);
  for (let i = 0; i < norm.length; i++) {
    const [r, g, b] = jetColor(norm[i]);
    heatData.data[i * 4] = r;
    heatData.data[i * 4 + 1] = g;
    heatData.data[i * 4 + 2] = b;
    heatData.data[i * 4 + 3] = 255;
  }

  if (viewMode === 'heatmap') {
    ctx.putImageData(heatData, 0, 0);
    return;
  }

  // ── Overlay mode: blend heatmap over original ────────────────────
  ctx.drawImage(sourceImg, 0, 0, imgW, imgH);

  const offscreen = document.createElement('canvas');
  offscreen.width = imgW;
  offscreen.height = imgH;
  const offCtx = offscreen.getContext('2d')!;
  offCtx.putImageData(heatData, 0, 0);

  ctx.globalAlpha = opacity;
  ctx.drawImage(offscreen, 0, 0);
  ctx.globalAlpha = 1;
}
