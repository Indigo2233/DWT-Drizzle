# -*- coding: utf-8 -*-
"""同一片天空的对比：把两版成品粗配准后，在同一位置各裁 700² 并排。

用法：cmp_sky.py A.xisf B.xisf out.npy [x0] [y0] [size]
不给 x0/y0 时自动选「两图都有覆盖、且离画幅中心最远」的亮星位置。
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
sys.path.insert(0, 'core')
import xisf_io


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def coarse_shift(a, b, ds=4):
    A = gaussian_filter(np.maximum(a[::ds, ::ds], 0), 1.0)
    B = gaussian_filter(np.maximum(b[::ds, ::ds], 0), 1.0)
    A -= A.mean()
    B -= B.mean()
    c = fftconvolve(A, B[::-1, ::-1], mode='full')
    py, px = np.unravel_index(np.argmax(c), c.shape)
    return (py - (B.shape[0] - 1)) * ds, (px - (B.shape[1] - 1)) * ds


def stretch(t, lo, hi):
    d = np.arcsinh(np.clip((t - lo) / max(hi - lo, 1e-9) * 6, 0, None)) / np.arcsinh(6.0)
    return (np.clip(d, 0, 1) * 255).astype(np.uint8)


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    out = sys.argv[3]
    ia, ib = load(pa), load(pb)
    dy, dx = coarse_shift(ia, ib)
    print('粗配准 dx=%d dy=%d' % (dx, dy))
    S = 700
    h = min(ia.shape[0], ib.shape[0] + dy)
    w = min(ia.shape[1], ib.shape[1] + dx)
    # 在 A 里找一颗靠外的亮星（避开黑边），同时保证 B 里对应位置有覆盖
    hh, ww = ia.shape
    yy, xx = np.mgrid[0:hh:4, 0:ww:4]
    sub = ia[::4, ::4]
    cy, cx = hh / 2.0 / 4, ww / 2.0 / 4
    r = np.hypot(yy - cy, xx - cx) / max(cy, cx)
    bye = (yy * 4 - dy >= S) & (yy * 4 - dy < ib.shape[0] - S) & \
          (xx * 4 - dx >= S) & (xx * 4 - dx < ib.shape[1] - S) & (r > 0.45) & (r < 0.75)
    lo = np.percentile(sub[bye], 99.5)
    cand = np.argwhere(bye & (sub > lo))
    y, x = cand[len(cand) // 2] * 4
    y = int(np.clip(y, S, ia.shape[0] - S - 1))
    x = int(np.clip(x, S, ia.shape[1] - S - 1))
    print('取样点 A=(%d,%d) 半径比 %.2f' % (x, y, np.hypot(y - hh / 2, x - ww / 2) / max(hh, ww) * 2))
    ta = ia[y - S // 2:y + S // 2, x - S // 2:x + S // 2]
    y2, x2 = y - dy, x - dx
    tb = ib[y2 - S // 2:y2 + S // 2, x2 - S // 2:x2 + S // 2]
    lo = float(np.median(ta))
    hi = float(np.percentile(ta, 99.5))
    sep = np.full((S, 8), 255, np.uint8)
    np.save(out, np.hstack([stretch(ta, lo, hi), sep, stretch(tb, lo, hi)]))
    print('ok')


if __name__ == '__main__':
    main()
