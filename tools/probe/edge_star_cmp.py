# -*- coding: utf-8 -*-
"""对比用户版与新版在「视场边缘亮星」上的形态：各取视场外侧 20% 环带里
最亮的星，4× 放大后并排。"""
import sys
import numpy as np
sys.path.insert(0, 'core')
import xisf_io

A = sys.argv[1]
B = sys.argv[2]
OUT = sys.argv[3]


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def edge_star(img, half=24):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.hypot((yy - cy) / cy, (xx - cx) / cx)
    band = r > 0.72
    lo = np.percentile(img[band], 99.0)
    cand = np.argwhere(band & (img > lo))
    if len(cand) == 0:
        return None
    # 取离中心最远、且不在画布黑边上的那颗
    d = np.hypot(cand[:, 0] - cy, cand[:, 1] - cx)
    y, x = cand[int(np.argmax(d))]
    y = int(np.clip(y, half, h - half - 1))
    x = int(np.clip(x, half, w - half - 1))
    return y, x


def cut(img, y, x, half=24, zoom=4):
    t = img[y - half:y + half, x - half:x + half]
    lo = float(np.median(t))
    hi = float(t.max())
    d = np.arcsinh(np.clip((t - lo) / max(hi - lo, 1e-9) * 6, 0, None)) / np.arcsinh(6.0)
    im = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    return np.repeat(np.repeat(im, zoom, 0), zoom, 1)


def main():
    ia, ib = load(A), load(B)
    pa = edge_star(ia)
    pb = edge_star(ib)
    print('A 边缘亮星', pa, 'B', pb)
    ta = cut(ia, *pa)
    tb = cut(ib, *pb)
    sep = np.full((max(ta.shape[0], tb.shape[0]), 8), 255, np.uint8)
    np.save(OUT, np.hstack([ta, sep, tb]))
    print('ok')


if __name__ == '__main__':
    main()
