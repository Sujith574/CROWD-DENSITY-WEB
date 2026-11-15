'use client';

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  Download,
  Layers,
  MapPin,
  Shield,
  Sliders,
  Upload,
  Users,
  Zap,
} from 'lucide-react';
import {
  runInference,
  renderHeatmap,
  InferenceResult,
} from '@/utils/inference';

// ── Types ─────────────────────────────────────────────────────────────────

type ViewMode = 'overlay' | 'heatmap' | 'original';

interface Metrics {
  model: string;
  architecture: {
    frontend: string;
    backend: string;
    head: string;
    total_parameters: number;
  };
  training: {
    dataset: string;
    epochs: number;
    learning_rate: number;
    batch_size: number;
    training_time_s: number;
  };
  evaluation: {
    mae: number;
    rmse: number;
  };
  export: {
    format: string;
    opset: number;
    size_mb: number;
  };
  loss_history: Array<{ epoch: number; loss: number; mae: number; mse: number }>;
}

// ── Sample Venue Images ───────────────────────────────────────────────────

const SAMPLE_VENUES = [
  { id: 'stadium',  label: 'Stadium Crowd',      src: '/samples/stadium.jpg',  icon: '🏟️' },
  { id: 'transit',  label: 'Metro Transit Hub',   src: '/samples/transit.jpg',  icon: '🚇' },
  { id: 'plaza',    label: 'City Plaza',           src: '/samples/plaza.jpg',    icon: '🏙️' },
];

// ── Helpers ───────────────────────────────────────────────────────────────

function congestionStatus(headcount: number, threshold: number) {
  const ratio = headcount / threshold;
  if (ratio < 0.6)  return { label: 'NORMAL FLOW',           color: 'emerald', icon: CheckCircle2  };
  if (ratio < 0.9)  return { label: 'MODERATE CROWD',        color: 'yellow',  icon: AlertTriangle };
  return               { label: 'HIGH CONGESTION ALERT',  color: 'red',     icon: Shield        };
}

function formatNum(n: number) {
  return n.toLocaleString('en-US');
}

// ── Main Component ────────────────────────────────────────────────────────

export default function CrowdDashboard() {
  // ── State ──────────────────────────────────────────────────────────────
  const [imageEl,     setImageEl]     = useState<HTMLImageElement | null>(null);
  const [imageSrc,    setImageSrc]    = useState<string | null>(null);
  const [result,      setResult]      = useState<InferenceResult | null>(null);
  const [metrics,     setMetrics]     = useState<Metrics | null>(null);
  const [threshold,   setThreshold]   = useState(150);
  const [opacity,     setOpacity]     = useState(0.5);
  const [viewMode,    setViewMode]    = useState<ViewMode>('overlay');
  const [isRunning,   setIsRunning]   = useState(false);
  const [isDragging,  setIsDragging]  = useState(false);
  const [modelReady,  setModelReady]  = useState(false);
  const [loadError,   setLoadError]   = useState<string | null>(null);

  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Load model + metrics on mount ─────────────────────────────────────
  useEffect(() => {
    // Pre-warm the ONNX session
    import('@/utils/inference').then(({ getSession }) => {
      getSession()
        .then(() => setModelReady(true))
        .catch((e) => setLoadError(String(e)));
    });

    // Load metrics.json
    fetch('/metrics.json')
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, []);

  // ── Re-render heatmap when opacity / viewMode changes ─────────────────
  useEffect(() => {
    if (result && imageEl && canvasRef.current) {
      renderHeatmap(result, imageEl, canvasRef.current, opacity, viewMode);
    }
  }, [result, imageEl, opacity, viewMode]);

  // ── Image loading ──────────────────────────────────────────────────────
  const loadAndInfer = useCallback(
    async (src: string) => {
      setIsRunning(true);
      setResult(null);
      setImageSrc(src);

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = src;
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('Failed to load image'));
      });

      setImageEl(img);

      try {
        const r = await runInference(img);
        setResult(r);
        if (canvasRef.current) {
          renderHeatmap(r, img, canvasRef.current, opacity, viewMode);
        }
      } catch (e) {
        console.error('Inference error:', e);
        setLoadError(`Inference failed: ${e}`);
      } finally {
        setIsRunning(false);
      }
    },
    [opacity, viewMode]
  );

  // ── File upload handler ────────────────────────────────────────────────
  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      const url = URL.createObjectURL(file);
      loadAndInfer(url);
    },
    [loadAndInfer]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  // ── Download canvas ────────────────────────────────────────────────────
  const handleDownload = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    link.download = 'crowd-hotspot-report.png';
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  // ── Derived state ──────────────────────────────────────────────────────
  const status = result
    ? congestionStatus(result.headcount, threshold)
    : null;

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-200">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Users size={18} className="text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white leading-tight">
                Crowd Density Monitor
              </h1>
              <p className="text-xs text-slate-400">CSRNet · Dilated CNN · ONNX Runtime Web</p>
            </div>
          </div>

          {/* Model status pill */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border
            ${modelReady
              ? 'bg-emerald-950 border-emerald-700 text-emerald-400'
              : 'bg-slate-800 border-slate-600 text-slate-400 animate-pulse'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${modelReady ? 'bg-emerald-400' : 'bg-slate-500'}`} />
            {modelReady ? 'Model Ready' : 'Loading ONNX…'}
          </div>
        </div>
      </header>

      <div className="max-w-screen-2xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr_280px] gap-6">

          {/* ═══════════════════════════════════════════════════════════════
              LEFT PANEL — Controls & Input
          ═══════════════════════════════════════════════════════════════ */}
          <aside className="space-y-5">

            {/* ── Upload Zone ─────────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <Upload size={13} /> Upload Image
              </h2>

              <div
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all
                  ${isDragging ? 'border-blue-500 bg-blue-950/30' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/50'}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <Camera size={28} className="mx-auto text-slate-500 mb-2" />
                <p className="text-sm text-slate-400">Drop an image or click to browse</p>
                <p className="text-xs text-slate-600 mt-1">JPG, PNG, WebP</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
              />
            </div>

            {/* ── Sample Venues ────────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <MapPin size={13} /> Sample Venues
              </h2>
              <div className="space-y-2">
                {SAMPLE_VENUES.map((v) => (
                  <button
                    key={v.id}
                    onClick={() => loadAndInfer(v.src)}
                    disabled={isRunning}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700
                               border border-slate-700 hover:border-blue-600 transition-all text-left
                               disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="text-xl">{v.icon}</span>
                    <span className="text-sm text-slate-300">{v.label}</span>
                    <ChevronRight size={14} className="ml-auto text-slate-500" />
                  </button>
                ))}
              </div>
            </div>

            {/* ── Controls ─────────────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-5">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                <Sliders size={13} /> Controls
              </h2>

              {/* Safe Capacity Threshold */}
              <div>
                <div className="flex justify-between mb-1.5">
                  <label className="text-xs text-slate-400">Safe Capacity Threshold</label>
                  <span className="text-xs font-mono text-blue-400">{threshold} people</span>
                </div>
                <input
                  type="range" min={10} max={1000} step={10}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                  <span>10</span><span>1 000</span>
                </div>
              </div>

              {/* Heatmap Opacity */}
              <div>
                <div className="flex justify-between mb-1.5">
                  <label className="text-xs text-slate-400">Heatmap Opacity</label>
                  <span className="text-xs font-mono text-cyan-400">{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range" min={10} max={90} step={5}
                  value={Math.round(opacity * 100)}
                  onChange={(e) => setOpacity(Number(e.target.value) / 100)}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
              </div>

              {/* View Mode Toggle */}
              <div>
                <label className="text-xs text-slate-400 block mb-2">View Mode</label>
                <div className="grid grid-cols-3 gap-1 bg-slate-800 p-1 rounded-lg">
                  {(['overlay', 'heatmap', 'original'] as ViewMode[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => setViewMode(m)}
                      className={`py-1.5 rounded-md text-xs font-medium capitalize transition-all
                        ${viewMode === m
                          ? 'bg-blue-600 text-white shadow'
                          : 'text-slate-400 hover:text-slate-200'}`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </aside>

          {/* ═══════════════════════════════════════════════════════════════
              CENTER PANEL — Visual Canvas
          ═══════════════════════════════════════════════════════════════ */}
          <main className="flex flex-col gap-4">

            {/* ── Canvas area ──────────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden relative">
              {/* Title bar */}
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800">
                <div className="flex items-center gap-2 text-sm text-slate-300 font-medium">
                  <Layers size={15} className="text-blue-400" />
                  Congestion Hotspot Map
                </div>
                {result && (
                  <button
                    onClick={handleDownload}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-slate-800 hover:bg-slate-700
                               text-xs text-slate-300 border border-slate-700 hover:border-blue-500 transition-all"
                  >
                    <Download size={12} /> Export PNG
                  </button>
                )}
              </div>

              {/* Canvas / placeholder */}
              <div className="relative min-h-[420px] flex items-center justify-center bg-slate-950/40">
                {isRunning && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/70 z-10">
                    <div className="text-center">
                      <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                      <p className="text-sm text-slate-300">Running CSRNet inference…</p>
                    </div>
                  </div>
                )}

                {!imageSrc && !isRunning && (
                  <div className="text-center py-16">
                    <div className="w-20 h-20 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4">
                      <Activity size={36} className="text-slate-600" />
                    </div>
                    <p className="text-slate-500 text-sm">Upload an image or select a sample venue</p>
                    <p className="text-slate-600 text-xs mt-1">The heatmap will appear here</p>
                  </div>
                )}

                {imageSrc && (
                  <canvas
                    ref={canvasRef}
                    className="max-w-full max-h-[600px] object-contain mx-auto block"
                    style={{ display: result ? 'block' : 'none' }}
                  />
                )}

                {/* Heatmap showing but no result yet */}
                {imageSrc && !result && !isRunning && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imageSrc} alt="source" className="max-w-full max-h-[600px] object-contain opacity-30" />
                  </div>
                )}
              </div>

              {/* Colour legend bar */}
              <div className="px-4 py-3 border-t border-slate-800 flex items-center gap-3">
                <span className="text-xs text-slate-500">Low Density</span>
                <div className="flex-1 h-3 rounded-full"
                  style={{
                    background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)',
                  }}
                />
                <span className="text-xs text-slate-500">Severe Hotspot</span>
              </div>
            </div>

            {/* ── Error banner ─────────────────────────────────────────── */}
            {loadError && (
              <div className="bg-red-950/50 border border-red-800 rounded-xl px-4 py-3 text-sm text-red-300 flex items-start gap-2">
                <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                <span>{loadError}</span>
              </div>
            )}
          </main>

          {/* ═══════════════════════════════════════════════════════════════
              RIGHT PANEL — Telemetry & Model Info
          ═══════════════════════════════════════════════════════════════ */}
          <aside className="space-y-5">

            {/* ── KPI Cards ────────────────────────────────────────────── */}
            <div className="grid grid-cols-2 gap-3">
              {/* Headcount */}
              <div className="col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
                  <Users size={12} /> Estimated Headcount
                </div>
                <div className="text-4xl font-black text-white tracking-tight">
                  {result ? formatNum(result.headcount) : '—'}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">people detected</div>
              </div>

              {/* Peak Density */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                <div className="text-xs text-slate-400 mb-1">Peak Density</div>
                <div className="text-xl font-bold text-orange-400">
                  {result ? result.peakDensity.toFixed(4) : '—'}
                </div>
                <div className="text-xs text-slate-600">hotspot index</div>
              </div>

              {/* Inference time */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                <div className="text-xs text-slate-400 mb-1 flex items-center gap-1">
                  <Zap size={10} /> Latency
                </div>
                <div className="text-xl font-bold text-cyan-400">
                  {result ? `${result.inferenceMs} ms` : '—'}
                </div>
                <div className="text-xs text-slate-600">ONNX inference</div>
              </div>
            </div>

            {/* ── Congestion Status Banner ─────────────────────────────── */}
            {status && (
              <div className={`rounded-xl p-4 border
                ${status.color === 'emerald'
                  ? 'bg-emerald-950/60 border-emerald-700'
                  : status.color === 'yellow'
                  ? 'bg-yellow-950/60 border-yellow-700'
                  : 'bg-red-950/60 border-red-700 animate-pulse-slow'}`}>
                <div className={`flex items-center gap-2 text-sm font-bold mb-1
                  ${status.color === 'emerald' ? 'text-emerald-400'
                    : status.color === 'yellow' ? 'text-yellow-400'
                    : 'text-red-400'}`}>
                  <status.icon size={15} />
                  {status.label}
                </div>
                <div className="text-xs text-slate-400">
                  {result!.headcount} / {threshold} capacity threshold
                </div>
                {/* Progress bar */}
                <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all
                      ${status.color === 'emerald' ? 'bg-emerald-500'
                        : status.color === 'yellow' ? 'bg-yellow-500'
                        : 'bg-red-500'}`}
                    style={{ width: `${Math.min(100, (result!.headcount / threshold) * 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* ── Model Evaluation ─────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <BarChart3 size={13} /> Model Evaluation
              </h2>

              {metrics ? (
                <div className="space-y-3">
                  {/* MAE / RMSE */}
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-slate-800/60 rounded-lg p-2.5">
                      <div className="text-xs text-slate-500 mb-0.5">MAE</div>
                      <div className="text-lg font-bold text-blue-400">{metrics.evaluation.mae.toFixed(2)}</div>
                    </div>
                    <div className="bg-slate-800/60 rounded-lg p-2.5">
                      <div className="text-xs text-slate-500 mb-0.5">RMSE</div>
                      <div className="text-lg font-bold text-purple-400">{metrics.evaluation.rmse.toFixed(2)}</div>
                    </div>
                  </div>

                  {/* Architecture summary */}
                  <div className="space-y-1.5 text-xs text-slate-400">
                    <div className="flex gap-2">
                      <span className="text-blue-400 font-medium shrink-0">Frontend:</span>
                      <span className="text-slate-500">{metrics.architecture.frontend}</span>
                    </div>
                    <div className="flex gap-2">
                      <span className="text-cyan-400 font-medium shrink-0">Backend:</span>
                      <span className="text-slate-500">{metrics.architecture.backend}</span>
                    </div>
                    <div className="flex gap-2">
                      <span className="text-purple-400 font-medium shrink-0">Head:</span>
                      <span className="text-slate-500">{metrics.architecture.head}</span>
                    </div>
                  </div>

                  {/* Training details */}
                  <div className="border-t border-slate-800 pt-2 space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Dataset</span>
                      <span className="text-slate-300 font-medium capitalize">{metrics.training.dataset}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Epochs</span>
                      <span className="text-slate-300 font-medium">{metrics.training.epochs}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Parameters</span>
                      <span className="text-slate-300 font-medium">{(metrics.architecture.total_parameters / 1e6).toFixed(1)}M</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">ONNX Size</span>
                      <span className="text-slate-300 font-medium">{metrics.export.size_mb} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Training Time</span>
                      <span className="text-slate-300 font-medium">{metrics.training.training_time_s}s</span>
                    </div>
                  </div>

                  {/* Loss mini-chart (ASCII sparkline) */}
                  {metrics.loss_history.length > 0 && (
                    <div>
                      <div className="text-xs text-slate-500 mb-1">MAE History</div>
                      <div className="h-12 relative bg-slate-800/40 rounded-md overflow-hidden">
                        {(() => {
                          const maes = metrics.loss_history.map((h) => h.mae);
                          const min = Math.min(...maes);
                          const max = Math.max(...maes) || 1;
                          return maes.map((mae, i) => {
                            const pct = 1 - (mae - min) / (max - min + 1e-8);
                            return (
                              <div
                                key={i}
                                className="absolute bottom-0 bg-blue-500/60 rounded-t-sm"
                                style={{
                                  left:   `${(i / maes.length) * 100}%`,
                                  width:  `${90 / maes.length}%`,
                                  height: `${Math.max(4, pct * 100)}%`,
                                }}
                              />
                            );
                          });
                        })()}
                      </div>
                      <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                        <span>Epoch 1</span>
                        <span>Epoch {metrics.loss_history.length}</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-xs text-slate-500 text-center py-4">
                  <BarChart3 size={24} className="mx-auto mb-2 text-slate-700" />
                  <p>Run training to see model metrics</p>
                  <p className="text-slate-600 mt-0.5">python ml_pipeline/train_and_export.py</p>
                </div>
              )}
            </div>

            {/* ── About ────────────────────────────────────────────────── */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-500 space-y-1">
                <p className="text-slate-400 font-medium text-xs mb-2">How it works</p>
                <p>1. VGG-16 frontend extracts 512-channel feature maps at 1/8× resolution.</p>
                <p>2. Six dilated conv layers (d=2) capture multi-scale crowd context.</p>
                <p>3. 1×1 conv head generates a continuous density map.</p>
                <p>4. Sum of density = estimated headcount. JET colourmap highlights hotspots.</p>
                <p className="text-slate-600 mt-2">Inference runs 100% in-browser via ONNX Runtime WASM — no server needed.</p>
              </div>
            </div>

          </aside>
        </div>
      </div>
    </div>
  );
}

