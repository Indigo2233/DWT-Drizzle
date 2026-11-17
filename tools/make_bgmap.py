# -*- coding: utf-8 -*-
"""把成品的低频背景单独成像（32px 块中位网格），用来看"坡"还在不在。

用法: python make_bgmap.py out.npy in.xisf
"""
import sys
import numpy as np
import xisf_io


def main(out, src, b=32):
    img = xisf_io.read_xisf(src)[0]
    a = np.asarray(img.data, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (1, 3) and a.shape[0] > 8 else a[0]
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    m = a[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, -1)
    g = np.median(m, axis=2)
    np.save(out, g.astype(np.float32))
    print(out, g.shape, 'p1 %.4e p50 %.4e p99 %.4e' % (
        np.percentile(g, 1), np.median(g), np.percentile(g, 99)))


if __name__ == '__main__':
    main(*sys.argv[1:])
