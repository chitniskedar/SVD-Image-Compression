import { computeSvd, type WorkerIn, type WorkerOut } from './svd';

const ctx = self as unknown as Worker;

ctx.onmessage = (e: MessageEvent<WorkerIn>) => {
  const { pixels, width, height } = e.data;
  const result = computeSvd(pixels, height, width, (sweep) => {
    const msg: WorkerOut = { type: 'progress', sweep };
    ctx.postMessage(msg);
  });
  const done: WorkerOut = { type: 'done', result };
  ctx.postMessage(done, [result.S.buffer, result.left.buffer, result.right.buffer]);
};
