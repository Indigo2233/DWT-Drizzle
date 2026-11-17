# -*- coding: utf-8 -*-
"""WBPP master vs 我们成品：同一把尺子量「星点信噪比」与「天空噪声」。

做法（各自在自己的网格上，量完折成原生像素）：
  * 天空噪声 σ：中值滤波去星 → 32px 块内 1.4826×MAD → 取中位，块按亮度分位取纯天空
  * 星点 SNR：最亮 N 颗未饱和星，3×3 峰值 − 天空，除以同一像素的环带噪声
"""
import sys
import numpy as np
from scipy import ndimage

sys.path[:0] = ['core', '.']
import xisf_io
import matchref as M


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def prep(a, bin_to_native, native_hint):
    if bin_to_native:
        h, w = a.shape
        a = a[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return a


def sky_sigma(a):
    """高频残差上的稳健 σ：先扣 >8px 的大尺度结构，再挖掉星点，取 1.4826×MAD。
    只在有效像素上算（drizzle 并集画布的黑边会被排除）。"""
    valid = a > 0.5 * np.median(a[a > 0])
    f = a - ndimage.gaussian_filter(a, 8.0)
    v = f[valid]
    m = np.median(v)
    s0 = 1.4826 * np.median(np.abs(v - m))
    keep = valid & (np.abs(f - m) < 6 * max(s0, 1e-12))
    v = f[keep]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def star_snr(a, sky, n=120):
    pts = M.find_stars(a, 99.6, 400)
    pts = [(float(p[0]), float(p[1])) for p in pts] if len(pts) else []
    vals = []
    for (x, y) in pts:
        x, y = int(round(x)), int(round(y))
        if y < 12 or x < 12 or y > a.shape[0] - 13 or x > a.shape[1] - 13:
            continue
        p = a[y - 1:y + 2, x - 1:x + 2].max()
        ann = np.concatenate([a[y - 9:y - 4, x - 9:x + 10].ravel(),
                              a[y + 5:y + 10, x - 9:x + 10].ravel()])
        bg = np.median(ann)
        noi = 1.4826 * np.median(np.abs(ann - bg))
        if noi <= 0:
            continue
        vals.append((p - bg) / noi)
        if len(vals) >= n:
            break
    return float(np.median(vals)), len(vals)


def run(name, path, binned):
    a = load(path)
    a = prep(a, binned, None)
    s = sky_sigma(a)
    snr, k = star_snr(a, s)
    print('%-22s %-12s 天空σ %.4e  星点 SNR 中位 %6.2f（%d 颗）'
          % (name, '%dx%d' % (a.shape[1], a.shape[0]), s, snr, k))
    return s, snr


D = r'G:\HFG1-3\master2\master'
OUR = r'G:\HFG1-3\master2\pws26'

print('== O 通道 ==')
s_w, snr_w = run('WBPP master (native)', D + r'\masterLight_BIN-1_2758x2204_EXPOSURE-1200.00s_FILTER-O_mono_autocrop.xisf', False)
s_o, snr_o = run('ours o26 (2x→native)', OUR + r'\PWS_o26.xisf', True)
print('  噪声比 ours/WBPP = %.3f   星点 SNR 比 = %.2f×' % (s_o / s_w, snr_o / snr_w))
print('  相对单帧（8.206e-05）：WBPP %.1f×  我们 %.1f×' % (8.206e-05 / s_w, 8.206e-05 / s_o))

print('== H 通道 ==')
s_w2, snr_w2 = run('WBPP master (native)', D + r'\masterLight_BIN-1_2758x2204_EXPOSURE-1200.00s_FILTER-H_mono_autocrop.xisf', False)
s_o2, snr_o2 = run('ours h26 (2x→native)', OUR + r'\PWS_h26.xisf', True)
print('  噪声比 ours/WBPP = %.3f   星点 SNR 比 = %.2f×' % (s_o2 / s_w2, snr_o2 / snr_w2))
print('  相对单帧（8.593e-05）：WBPP %.1f×  我们 %.1f×' % (8.593e-05 / s_w2, 8.593e-05 / s_o2))
