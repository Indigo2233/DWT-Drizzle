# -*- coding: utf-8 -*-
"""对比 bgdeg=0 / bgdeg=1 两版成品的背景平坦度与噪声。

用法: python probe_bg_compare.py <a.xisf> <b.xisf>
输出: 各自 128px 块中位网格拟合平面的峰谷幅度、块中位残差 RMS（背景不平行），
      以及天空区域逐像素噪声比、亮星区（最亮 0.1%）像素差统计。
"""
import sys
import numpy as np
import xisf_io


def load(p):
    img = xisf_io.read_xisf(p)[0]
    a = np.asarray(img.data, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (1, 3) and a.shape[0] > 8 else a[0]
    return a


def block_med(a, b=128):
    h, w = a.shape
    gh, gw = h // b, w // b
    x = a[:gh * b, :gw * b].reshape(gh, b, gw, b).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    return np.median(x, axis=2)


def plane_amp(g):
    gh, gw = g.shape
    yy = (np.arange(gh) + 0.5) / gh
    xx = (np.arange(gw) + 0.5) / gw
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    A = np.stack([np.ones_like(Y), Y, X], axis=-1).reshape(-1, 3)
    b = g.reshape(-1)
    ok = np.isfinite(b)
    c, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    p = (A @ c).reshape(g.shape)
    r = (b - A @ c)[ok]
    return float(p.max() - p.min()), float(1.4826 * np.median(np.abs(r - np.median(r))))


def main(pa, pb):
    A = load(pa)
    B = load(pb)
    print('shape', A.shape, B.shape)
    if A.shape != B.shape:
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        y0a, x0a = (A.shape[0] - h) // 2, (A.shape[1] - w) // 2
        y0b, x0b = (B.shape[0] - h) // 2, (B.shape[1] - w) // 2
        A = A[y0a:y0a + h, x0a:x0a + w]
        B = B[y0b:y0b + h, x0b:x0b + w]
        print('crop to', A.shape)
    # 去掉 drizzle 边缘覆盖不均的边带
    m = max(64, min(A.shape) // 50)
    A = A[m:-m, m:-m]
    B = B[m:-m, m:-m]
    for name, im in (('bg0', A), ('bg1', B)):
        g = block_med(im)
        amp, rms = plane_amp(g)
        print(f'{name}: 背景平面峰谷 {amp:.4e} 块中位残差RMS {rms:.4e}  全幅中位 {np.median(im):.4e}  P1 {np.percentile(im,1):.4e}  P99 {np.percentile(im,99):.4e}')
    d = B - A
    print(f'diff: rms {np.sqrt(np.mean(d.astype(np.float64)**2)):.4e}  max|d| {np.max(np.abs(d)):.4e}')
    # 天空（暗半区）噪声：取全幅中位数以下的像素块
    sky = A < np.percentile(A, 60)
    print('sky pixels frac', float(sky.mean()))
    # 亮核区（顶部 0.1%）上的差异：证明星点/星云没被削
    hi = A > np.percentile(A, 99.9)
    print(f'亮核像素(diff)/中位 = {np.median(np.abs(d[hi])) / np.median(np.abs(A[hi])) * 100:.3f}%')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
