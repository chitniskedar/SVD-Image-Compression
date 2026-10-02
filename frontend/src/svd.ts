/**
 * Browser port of the Python pipeline (svd_compression.py + metrics.py).
 * A = U Σ Vᵀ  →  A_k = U_k Σ_k V_kᵀ, computed with a one-sided Jacobi SVD.
 */

/** Longest image side used for the in-browser SVD (keeps the decomposition fast). */
export const MAX_SIDE = 320;

export interface SvdResult {
  m: number; // rows (height)
  n: number; // cols (width)
  S: Float64Array; // singular values, descending, length min(m, n)
  left: Float64Array; // r-th left vector (length m) at [r*m, (r+1)*m)
  right: Float64Array; // r-th right vector (length n) at [r*n, (r+1)*n)
}

export type WorkerIn = { pixels: Uint8Array; width: number; height: number };
export type WorkerOut =
  | { type: 'progress'; sweep: number }
  | { type: 'done'; result: SvdResult };

export function computeSvd(
  pixels: Uint8Array,
  m: number,
  n: number,
  onSweep?: (sweep: number) => void,
): SvdResult {
  // Jacobi needs rows >= cols, so decompose Aᵀ when the image is wider than tall.
  const transposed = m < n;
  const M = transposed ? n : m;
  const N = transposed ? m : n;

  const W = new Float64Array(M * N); // column-major working matrix
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = pixels[i * n + j];
      if (transposed) W[i * M + j] = v;
      else W[j * M + i] = v;
    }
  }
  const V = new Float64Array(N * N);
  for (let c = 0; c < N; c++) V[c * N + c] = 1;

  const EPS = 1e-10;
  for (let sweep = 1; sweep <= 30; sweep++) {
    let rotations = 0;
    for (let p = 0; p < N - 1; p++) {
      for (let q = p + 1; q < N; q++) {
        const po = p * M;
        const qo = q * M;
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let i = 0; i < M; i++) {
          const x = W[po + i];
          const y = W[qo + i];
          alpha += x * x;
          beta += y * y;
          gamma += x * y;
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        rotations++;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let i = 0; i < M; i++) {
          const x = W[po + i];
          const y = W[qo + i];
          W[po + i] = c * x - s * y;
          W[qo + i] = s * x + c * y;
        }
        const pv = p * N;
        const qv = q * N;
        for (let i = 0; i < N; i++) {
          const x = V[pv + i];
          const y = V[qv + i];
          V[pv + i] = c * x - s * y;
          V[qv + i] = s * x + c * y;
        }
      }
    }
    onSweep?.(sweep);
    if (rotations === 0) break;
  }

  const norms = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    let sum = 0;
    for (let i = 0; i < M; i++) sum += W[j * M + i] ** 2;
    norms[j] = Math.sqrt(sum);
  }
  const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => norms[b] - norms[a]);

  const S = new Float64Array(N);
  const U = new Float64Array(M * N);
  const Vs = new Float64Array(N * N);
  order.forEach((src, dst) => {
    S[dst] = norms[src];
    const inv = norms[src] > 1e-12 ? 1 / norms[src] : 0;
    for (let i = 0; i < M; i++) U[dst * M + i] = W[src * M + i] * inv;
    for (let i = 0; i < N; i++) Vs[dst * N + i] = V[src * N + i];
  });

  return transposed
    ? { m, n, S, left: Vs, right: U }
    : { m, n, S, left: U, right: Vs };
}

/** A_k = Σ_{r<k} σ_r u_r v_rᵀ, as a flat m×n array. */
export function reconstruct({ m, n, S, left, right }: SvdResult, k: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let r = 0; r < k; r++) {
    const lo = r * m;
    const ro = r * n;
    for (let i = 0; i < m; i++) {
      const coef = S[r] * left[lo + i];
      const row = i * n;
      for (let j = 0; j < n; j++) out[row + j] += coef * right[ro + j];
    }
  }
  return out;
}

/** Same as image_utils.save_image: clip to [0, 255], then truncate to uint8. */
export function quantize(a: Float32Array): Uint8Array {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.min(255, Math.max(0, a[i]));
  return out;
}

// ---- metrics.py ----
export const originalSize = (m: number, n: number) => m * n;
export const compressedSize = (m: number, n: number, k: number) => k * (m + n + 1);
export const compressionRatio = (m: number, n: number, k: number) =>
  originalSize(m, n) / compressedSize(m, n, k);
export const storageSavedPercent = (m: number, n: number, k: number) =>
  (1 - compressedSize(m, n, k) / originalSize(m, n)) * 100;
export const breakEvenRank = (m: number, n: number) => (m * n) / (m + n + 1);

export function mse(a: Uint8Array, b: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum / a.length;
}

export function psnr(err: number, maxVal = 255): number {
  return err === 0 ? Infinity : 10 * Math.log10((maxVal * maxVal) / err);
}
