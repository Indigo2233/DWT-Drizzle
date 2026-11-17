# -*- coding: utf-8 -*-
"""两版成品的对齐差异分析：先做整数像素配准，再看"被减掉的低频面"有多大。

用法: python probe_bg_diff.py a.xisf b.xisf   （a=bgdeg0, b=bgdeg1）
输出: 相对平移、对齐后差异的背景面峰谷（= 校正掉的梯度量级）、
      天空区差异 RMS、亮核区相对差（对齐后应很小）。
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


def coarse_shift(A, B, box=1400, rad=8):
    cy, cx = A.shape[0] // 2, A.shape[1] // 2
    pa = A[cy - box:cy + box, cx - box:cx + box].astype(np.float32)
    pb = B[cy - box:cy + box, cx - box:cx + box].astype(np.float32)
    pa -= pa.mean()
    pb -= pb.mean()
    fa = np.fft.rfft2(pa)
    fb = np.fft.rfft2(pb)
    cc = np.fft.irfft2(fb * np.conj(fa), s=pa.shape)
    # 只搜 ±rad
    cc = np.fft.fftshift(cc)
    m = cc.shape[0] // 2
    win = cc[m - rad:m + rad + 1, m - rad:m + rad + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return iy - rad, ix - rad


def block_med(a, b=128):
    h, w = a.shape
    gh, gw = h // b, w // b
    x = a[:gh * b, :gw * b].reshape(gh, b, gw, b).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    return np.median(x, axis=2)


def plane_amp(g):
    gh, gw = g.shape
    Y, X = np.meshgrid((np.arange(gh) + .5) / gh, (np.arange(gw) + .5) / gw, indexing='ij')
    A = np.stack([np.ones_like(Y), Y, X], -1).reshape(-1, 3)
    b = g.ravel()
    ok = np.isfinite(b)
    c, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    p = (A @ c).reshape(g.shape)
    r = (b - A @ c)[ok]
    return float(p.max() - p.min()), float(1.4826 * np.median(np.abs(r - np.median(r))))


def main(pa, pb):
    A = load(pa)
    B = load(pb)
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A = A[:h, :w]
    B = B[:h, :w]
    dy, dx = coarse_shift(A, B)
    print(f'粗配准 平移 (dy,dx) = ({dy},{dx})')
    # 用平移后的公共区域
    ay0, ay1 = max(0, -dy), min(h, h - dy)
    ax0, ax1 = max(0, -dx), min(w, w - dx)
    Asub = A[ay0:ay1, ax0:ax1]
    Bsub = B[ay0 + dy:ay1 + dy, ax0 + dx:ax1 + dx]
    e = max(64, min(Asub.shape) // 50)
    Asub = Asub[e:-e, e:-e]
    Bsub = Bsub[e:-e, e:-e]
    D = Bsub - Asub
    g = block_med(D)
    amp, rms = plane_amp(g)
    print(f'差异背景面 峰谷 {amp:.4e}  残差RMS {rms:.4e}   差异全图 RMS {np.sqrt(np.mean(D.astype(np.float64)**2)):.4e}')
    hi = Asub > np.percentile(Asub, 99.9)
    print(f'亮核 0.1% 像素相对差 {np.median(np.abs(D[hi])) / max(np.median(np.abs(Asub[hi])), 1e-30) * 100:.2f}%')
    sky = Asub < np.percentile(Asub, 60)
    print(f'天空像素差 RMS {np.sqrt(np.mean(D[sky].astype(np.float64)**2)):.4e}  天空电平 {np.median(Asub[sky]):.4e}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
