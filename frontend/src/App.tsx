import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import {
  MAX_SIDE,
  breakEvenRank,
  compressionRatio,
  mse,
  psnr,
  quantize,
  reconstruct,
  storageSavedPercent,
  type SvdResult,
  type WorkerOut,
} from './svd';

interface GrayImage {
  name: string;
  w: number;
  h: number;
  pixels: Uint8Array;
}

const QUICK_KS = [5, 20, 50, 100]; // K_VALUES from main.py
const REPO_URL = 'https://github.com/chitniskedar/SVD-Image-Compression';

function toGray(source: CanvasImageSource, sw: number, sh: number, name: string): GrayImage {
  const scale = Math.min(1, MAX_SIDE / Math.max(sw, sh));
  const w = Math.max(2, Math.round(sw * scale));
  const h = Math.max(2, Math.round(sh * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const pixels = new Uint8Array(w * h);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
  }
  return { name, w, h, pixels };
}

function makeSample(): GrayImage {
  const c = document.createElement('canvas');
  c.width = 320;
  c.height = 214;
  const g = c.getContext('2d')!;
  const bg = g.createLinearGradient(0, 0, 320, 214);
  bg.addColorStop(0, '#1b1f2a');
  bg.addColorStop(1, '#d0d6e0');
  g.fillStyle = bg;
  g.fillRect(0, 0, 320, 214);
  g.fillStyle = '#ffffff';
  g.beginPath();
  g.arc(112, 96, 52, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = '#08090a';
  g.fillRect(190, 40, 90, 130);
  g.strokeStyle = '#8a8f98';
  g.lineWidth = 2;
  for (let x = -214; x < 320; x += 9) {
    g.beginPath();
    g.moveTo(x, 214);
    g.lineTo(x + 214, 0);
    g.stroke();
  }
  g.fillStyle = '#ffffff';
  g.font = '600 44px sans-serif';
  g.fillText('SVD', 150, 200);
  return toGray(c, 320, 214, 'sample.png');
}

function GrayCanvas({ pixels, w, h, label }: { pixels: Uint8Array; w: number; h: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current!.getContext('2d')!;
    const data = ctx.createImageData(w, h);
    for (let i = 0; i < pixels.length; i++) {
      const v = pixels[i];
      data.data[i * 4] = v;
      data.data[i * 4 + 1] = v;
      data.data[i * 4 + 2] = v;
      data.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, w, h]);
  return <canvas ref={ref} width={w} height={h} className="pane-canvas" role="img" aria-label={label} />;
}

function Spectrum({ S, k }: { S: Float64Array; k: number }) {
  const W = 600;
  const H = 140;
  const logs = Array.from(S, (v) => Math.log10(Math.max(v, 1e-6)));
  const hi = Math.max(...logs);
  const lo = Math.min(...logs);
  const x = (i: number) => 20 + (i / Math.max(1, S.length - 1)) * (W - 40);
  const y = (l: number) => 12 + (1 - (l - lo) / Math.max(1e-9, hi - lo)) * (H - 28);
  const line = logs.map((l, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(l).toFixed(1)}`).join('');
  const kx = x(Math.min(k, S.length) - 1);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="spectrum" role="img" aria-label="Singular values on a log scale">
      <rect x={20} y={0} width={Math.max(0, kx - 20)} height={H - 8} fill="rgba(255,255,255,0.05)" />
      <path d={line} fill="none" stroke="#d0d6e0" strokeWidth={1.5} />
      <line x1={kx} x2={kx} y1={0} y2={H - 8} stroke="#ffffff" strokeWidth={1} />
    </svg>
  );
}

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '∞');

export default function App() {
  const [img, setImg] = useState<GrayImage | null>(null);
  const [svd, setSvd] = useState<SvdResult | null>(null);
  const [sweep, setSweep] = useState(0);
  const [k, setK] = useState(20);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!img) return;
    setSvd(null);
    setSweep(0);
    const worker = new Worker(new URL('./svd.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<WorkerOut>) => {
      if (e.data.type === 'progress') setSweep(e.data.sweep);
      else {
        const result = e.data.result;
        setSvd(result);
        setK((prev) => Math.min(prev, result.S.length));
      }
    };
    worker.onerror = () => setError('The SVD failed for this image. Try a different file.');
    worker.postMessage({ pixels: img.pixels, width: img.w, height: img.h });
    return () => worker.terminate();
  }, [img]);

  const deferredK = useDeferredValue(k);
  const recon = useMemo(() => (svd ? quantize(reconstruct(svd, deferredK)) : null), [svd, deferredK]);

  const stats = useMemo(() => {
    if (!svd || !img || !recon) return null;
    const { m, n, S } = svd;
    let total = 0;
    let kept = 0;
    S.forEach((s, i) => {
      total += s * s;
      if (i < deferredK) kept += s * s;
    });
    const err = mse(img.pixels, recon);
    return {
      ratio: compressionRatio(m, n, deferredK),
      saved: storageSavedPercent(m, n, deferredK),
      err,
      psnr: psnr(err),
      energy: (kept / total) * 100,
      breakEven: breakEvenRank(m, n),
    };
  }, [svd, img, recon, deferredK]);

  async function loadFile(file: File | undefined) {
    if (!file) return;
    setError('');
    try {
      const bmp = await createImageBitmap(file);
      setImg(toGray(bmp, bmp.width, bmp.height, file.name));
    } catch {
      setError('Could not read that file. Use a PNG, JPEG or WebP image.');
    }
  }

  function download() {
    if (!img || !recon) return;
    const c = document.createElement('canvas');
    c.width = img.w;
    c.height = img.h;
    const ctx = c.getContext('2d')!;
    const data = ctx.createImageData(img.w, img.h);
    recon.forEach((v, i) => {
      data.data.set([v, v, v, 255], i * 4);
    });
    ctx.putImageData(data, 0, 0);
    c.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `compressed_k${deferredK}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }

  const maxK = svd ? svd.S.length : 1;

  return (
    <>
      <header className="nav">
        <div className="wrap nav-inner">
          <span className="logo">SVD Compression</span>
          <nav>
            <a className="pill-white" href={`${REPO_URL}#readme`} target="_blank" rel="noreferrer">
              Source
            </a>
            <a className="pill-white" href={`${REPO_URL}#readme`} target="_blank" rel="noreferrer">
              README
            </a>
          </nav>
        </div>
      </header>

      <main className="wrap">
        <section className="hero">
          <h1>Singular Value Decomposition.</h1>
          <p>
            Pick a rank k and see what a low-rank approximation of your image stores and what it loses.
          </p>
        </section>

        <section className="card" aria-live="polite">
          {!img ? (
            <div
              className={`drop${dragging ? ' drop-on' : ''}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                void loadFile(e.dataTransfer.files[0]);
              }}
            >
              <p className="body-emph">Drop an image here</p>
              <p className="muted">PNG, JPEG or WebP. It is converted to grayscale and scaled to fit {MAX_SIDE}px.</p>
              <div className="row">
                <button className="btn-primary" onClick={() => fileRef.current?.click()}>
                  Choose image
                </button>
                <button className="btn-ghost" onClick={() => setImg(makeSample())}>
                  Use sample
                </button>
              </div>
              {error && <p className="err">{error}</p>}
            </div>
          ) : (
            <>
              <div className="card-head">
                <div>
                  <span className="mono">{img.name}</span>
                  <span className="badge">
                    {img.w} × {img.h}
                  </span>
                </div>
                <div className="row">
                  <button className="btn-ghost" onClick={() => fileRef.current?.click()}>
                    Replace
                  </button>
                  <button className="btn-primary" onClick={download} disabled={!recon}>
                    Download PNG
                  </button>
                </div>
              </div>

              <div className="panes">
                <figure>
                  <GrayCanvas pixels={img.pixels} w={img.w} h={img.h} label="Original grayscale image" />
                  <figcaption>Original</figcaption>
                </figure>
                <figure>
                  {recon ? (
                    <GrayCanvas pixels={recon} w={img.w} h={img.h} label={`Reconstruction with k = ${deferredK}`} />
                  ) : (
                    <div className="pane-canvas pending" style={{ aspectRatio: `${img.w} / ${img.h}` }}>
                      Computing SVD{sweep ? `, sweep ${sweep}` : ''}
                    </div>
                  )}
                  <figcaption>
                    Rank <span className="mono">{deferredK}</span>
                  </figcaption>
                </figure>
              </div>

              <div className="control">
                <label htmlFor="k">Singular values kept (k)</label>
                <input
                  id="k"
                  type="range"
                  min={1}
                  max={maxK}
                  value={k}
                  disabled={!svd}
                  onChange={(e) => setK(Number(e.target.value))}
                  style={{ ['--fill' as string]: `${((k - 1) / Math.max(1, maxK - 1)) * 100}%` }}
                />
                <span className="mono k-out">
                  {k} / {maxK}
                </span>
              </div>
              <div className="row pills">
                {QUICK_KS.filter((q) => q <= maxK).map((q) => (
                  <button key={q} className={`pill${k === q ? ' pill-on' : ''}`} onClick={() => setK(q)} disabled={!svd}>
                    k = {q}
                  </button>
                ))}
              </div>

              {stats && (
                <dl className="stats">
                  <div>
                    <dt>Compression ratio</dt>
                    <dd className="mono">{fmt(stats.ratio)}×</dd>
                  </div>
                  <div>
                    <dt>Storage saved</dt>
                    <dd className="mono">{fmt(stats.saved, 1)}%</dd>
                  </div>
                  <div>
                    <dt>MSE</dt>
                    <dd className="mono">{fmt(stats.err)}</dd>
                  </div>
                  <div>
                    <dt>PSNR</dt>
                    <dd className="mono">{fmt(stats.psnr)} dB</dd>
                  </div>
                  <div>
                    <dt>Energy kept</dt>
                    <dd className="mono">{fmt(stats.energy, 1)}%</dd>
                  </div>
                </dl>
              )}
              {stats && (
                <p className="muted note">
                  <span className={stats.ratio >= 1 ? 'badge badge-ok' : 'badge badge-bad'}>
                    {stats.ratio >= 1 ? 'Smaller than raw' : 'Larger than raw'}
                  </span>{' '}
                  Storing k(m+n+1) values beats the raw m×n matrix only while k &lt; {fmt(stats.breakEven, 1)}.
                </p>
              )}
              {error && <p className="err">{error}</p>}
            </>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              void loadFile(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
        </section>

        {svd && (
          <section className="card spectrum-card">
            <h2>Singular values</h2>
            <p className="muted">
              Log scale. The shaded region is what rank {deferredK} keeps. A steep drop means the image compresses well.
            </p>
            <Spectrum S={svd.S} k={deferredK} />
          </section>
        )}
      </main>
    </>
  );
}
