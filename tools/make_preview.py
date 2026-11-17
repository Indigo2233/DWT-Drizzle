# -*- coding: utf-8 -*-
"""把成品转成可看的小图（.npy，供 DSH python 存 PNG）。

用法: python make_preview.py out.npy in.xisf [stretch_lo] [stretch_hi]
下采样 4× 后做 asinh 拉伸（lo/hi 用全图分位数，缺省 30%/99.9%）。
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


def main(out, src, lo=None, hi=None):
    a = load(src)
    h, w = (a.shape[0] // 4) * 4, (a.shape[1] // 4) * 4
    a = a[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    lo = float(np.percentile(a, 30) if lo is None else lo)
    hi = float(np.percentile(a, 99.9) if hi is None else hi)
    d = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    d = np.arcsinh(20 * d) / np.arcsinh(20)
    np.save(out, d.astype(np.float32))
    print(out, d.shape, 'lo %.4e hi %.4e' % (lo, hi))


if __name__ == '__main__':
    main(*sys.argv[1:])
