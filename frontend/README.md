# SVD Image Compression — frontend

React + TypeScript + Vite. Styled to the Linear-style DESIGN.md.

The Python pipeline is ported to TypeScript (`src/svd.ts`), so no backend is needed:
- `computeSvd` — one-sided Jacobi SVD, runs in a Web Worker (`src/svd.worker.ts`)
- `reconstruct` / `quantize` — A_k = U_k Σ_k V_kᵀ, then clip + truncate to uint8 like `save_image`
- metrics match `metrics.py` (compression ratio, storage saved, MSE, PSNR, break-even rank)

Images are converted to grayscale and scaled to 320px max side (`MAX_SIDE`) so the SVD stays fast in the browser.

```bash
npm install
npm run dev
```

To keep it in your repo, copy this folder in as `frontend/`.
