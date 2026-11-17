# -*- coding: utf-8 -*-
"""把任意 xisf 渲染成 PNG 缩略图，快速目视检查（asinh 拉伸，可按百分位裁剪）。

用法（在 DWT-Drizzle 目录下）：
  py\\python.exe tools\\look.py "G:\\HFG1\\lab\\X\\PWS_X.xisf" [输出png] [中心裁剪边长] [拉仲倍数]
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import xisf_io  # noqa: E402


def main(argv):
    src = Path(argv[1])
    dst = Path(argv[2]) if len(argv) > 2 and argv[2].endswith('.png') \
        else src.with_name(src.stem + '_look.png')
    side = int(argv[3]) if len(argv) > 3 else 0
    stretch = float(argv[4]) if len(argv) > 4 else 12.0

    a = np.asarray(xisf_io.read_xisf(str(src))[0].data, np.float64)
    if side and side < min(a.shape):
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        h = side // 2
        a = a[cy - h:cy + h, cx - h:cx + h]
    lo = float(np.median(a))
    hi = float(np.percentile(a, 99.5))
    d = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1e-12) * stretch, 0, None))
    d /= np.arcsinh(stretch)
    img = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img).save(str(dst))
    except Exception:
        np.save(str(dst.with_suffix('.npy')), img)
        print('没有 PIL，已存 npy:', dst.with_suffix('.npy'))
        return 0
    print('写出', dst, img.shape, 'lo %.5g hi %.5g' % (lo, hi))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
