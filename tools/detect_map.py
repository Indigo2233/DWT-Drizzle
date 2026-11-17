# -*- coding: utf-8 -*-
"""模板法检测图：D = (成品低通 − 大尺度天空) / 局部噪声

用途：把「这个尺度上哪里显著」画出来，帮助判断暗弱结构是否真的高过本地噪声。
不参与叠加、不改权重，纯诊断。

用法：
    tools/detect_map.py 成品.xisf [输出前缀] [模板尺度...]
输出：
    <前缀>_D<尺度>.npy   显著性图（float32）
    <前缀>_D<尺度>.png   伪彩图（蓝=低于噪声，青/绿/黄=高，红=很高）
"""
import os
import struct
import sys
import zlib

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, 'core'), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import xisf_io


def load(path):
    return np.asarray(xisf_io.read_xisf(path)[0].data, np.float64)


def star_mask(a, thr_pct=99.7, grow=9):
    lo = np.percentile(a, 50)
    hi = np.percentile(a, thr_pct)
    m = a > max(hi, lo * 1.5)
    return ndimage.gaussian_filter(m.astype(np.float32), grow) > 1e-3


def local_sigma(a, smooth, block=64):
    d = a - smooth
    h, w = a.shape
    gh, gw = h // block, w // block
    if gh < 1 or gw < 1:
        return np.full_like(a, max(float(np.std(d)), 1e-12))
    g = d[:gh * block, :gw * block].reshape(gh, block, gw, block)
    med = np.median(g, axis=(1, 3), keepdims=True)
    mad = 1.4826 * np.median(np.abs(g - med), axis=(1, 3))
    mad = np.maximum(mad, 1e-12)
    sig = ndimage.zoom(mad, (h / gh, w / gw), order=1)
    return sig[:h, :w]


def _png(rgb, path):
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6))
           + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)


def colorize(d, lo=-3.0, hi=6.0):
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    r = np.clip(1.5 * x - 0.25, 0, 1)
    g = np.clip(1.4 - np.abs(x - 0.62) * 3.0, 0, 1)
    b = np.clip(1.1 - x * 1.4, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    stem = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0]
    sigs = [int(v) for v in sys.argv[3:]] or [16, 32, 64]

    a = load(src)
    mk = star_mask(a)
    f = np.where(mk, ndimage.median_filter(a, 31), a)
    sky = ndimage.gaussian_filter(f, 128)
    print('画幅 %dx%d  星掩膜 %.1f%%  天空 %.5g'
          % (a.shape[1], a.shape[0], 100 * mk.mean(), float(np.median(a))))
    for s in sigs:
        t = ndimage.gaussian_filter(f, s)
        sig = local_sigma(a, ndimage.gaussian_filter(f, max(s * 4, 96)))
        D = (t - sky) / sig
        D[mk] = np.nan
        v = D[np.isfinite(D)]
        print('σ=%2dpx  中位 %.2f  天空 P99 %.2f  最强 %.2f'
              % (s, float(np.median(v)), float(np.percentile(v, 99)), float(np.nanmax(v))))
        np.save('%s_D%d.npy' % (stem, s), D.astype(np.float32))
        img = colorize(np.nan_to_num(D, nan=float(np.nanmedian(D))))
        step = max(1, img.shape[1] // 1400)
        _png(img[::step, ::step], '%s_D%d.png' % (stem, s))
    return 0


if __name__ == '__main__':
    sys.exit(main())
