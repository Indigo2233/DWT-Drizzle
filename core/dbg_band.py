"""诊断：CFA drizzle 按带累积 vs 整幅一次累积，找出棋盘状明暗的来源。

用法：python dbg_band.py <帧目录> [帧数]
做法：同一帧、同一几何，分别用
  (a) 一条带覆盖整幅（Y0=0, Y1=H）
  (b) 按 256 行分带
累加 G 平面，直接比两张图；再用 2×2 分块均值看差异的空间周期。
"""
import sys
from pathlib import Path

import numpy as np

import cfa
import pws
from drizzle import plan_output_rot, _xy_rows, _scatter_band, measure_shifts


def main():
    src = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    frames = sorted(src.glob('*.xisf'))[:limit]
    crop = 1600
    center = pws.resolve_center(frames, crop, None)
    fs = pws.prepare_frames(frames, 'dbg', crop, center, limit, '', 0.55, False, 0, 0,
                            star_n=60)
    try:
        n = int(fs.n)
        h, w = fs.shape
        g0 = cfa.plane(np.asarray(fs.frame(0), np.float32), 'RGGB', 'G')
        stars = pws.build_star_table(g0, n_star=60)
        sh = measure_shifts(fs, stars, ref=0, log=lambda *a, **k: None, guess=None)
        dx = np.asarray(sh['dx'], np.float64)[:n]
        dy = np.asarray(sh['dy'], np.float64)[:n]
        th = np.asarray(sh['theta'], np.float64)[:n]
        kk = np.asarray(sh.get('scale', np.ones(n)), np.float64)[:n]
        s = 2
        plan = plan_output_rot(fs.shape, dx, dy, th, float(s), k=kk)
        H, W = int(plan['H']), int(plan['W'])
        print(f'canvas {W}x{H}  n={n}')

        def accum(Y0, Y1):
            bh = Y1 - Y0
            num = np.zeros((n, bh, W), np.float32)
            wts = np.zeros((n, bh, W), np.float32)
            for i in range(n):
                k_i = float(kk[i]) if np.isfinite(kk[i]) else 1.0
                P = 1.0 * s * k_i
                ux, uy = -float(dx[i]), -float(dy[i])
                X, Y = _xy_rows(0, h, w, plan, float(th[i]), ux, uy, k_i, 0.0, h)
                keep = (Y > Y0 - P) & (Y < Y1 + P)
                vals = np.asarray(fs.strip(i, 0, h), np.float32)
                yy = np.arange(h)[:, None] % 2
                xx = np.arange(w)[None, :] % 2
                col = np.where((yy == 0) & (xx == 1), 1,
                               np.where((yy == 1) & (xx == 0), 1, -1))  # RGGB: G at (0,1),(1,0)
                m = (col == 1) & keep
                nb, cb = _scatter_band(vals[m], X[m], Y[m], P, W, Y0, Y1, with_cov=True)
                num[i], wts[i] = nb, cb
            tot = num.sum(axis=0)
            wt = wts.sum(axis=0)
            return np.where(wt > 1e-9, tot / np.maximum(wt, 1e-9), np.nan)

        full = accum(0, H)
        banded = np.full((H, W), np.nan, np.float32)
        for Y0 in range(0, H, 256):
            Y1 = min(H, Y0 + 256)
            banded[Y0:Y1] = accum(Y0, Y1)
        d = np.abs(np.nan_to_num(full) - np.nan_to_num(banded))
        print('full vs banded: max %.3e  mean %.3e  frac>1e-6 %.3f%%'
              % (d.max(), d.mean(), 100 * np.mean(d > 1e-6)))

        # 2×2 与 8×8 分块均值，看棋盘周期
        for k in (2, 4, 8, 16):
            hh, ww = H // k * k, W // k * k
            bl = np.nanmean(np.nan_to_num(full[:hh, :ww]).reshape(hh // k, k, ww // k, k),
                            axis=(1, 3))
            print(f'  block {k}: mean {bl.mean():.4e} std {bl.std():.3e} '
                  f'(std/mean {bl.std() / max(bl.mean(), 1e-30):.2e})')
        np.save(r'C:\Windows\TEMP\dbg_full.npy', full)
        np.save(r'C:\Windows\TEMP\dbg_banded.npy', banded)
        print('saved C:\\Windows\\TEMP\\dbg_full.npy / dbg_banded.npy')
    finally:
        fs.close(keep=False)


if __name__ == '__main__':
    main()
