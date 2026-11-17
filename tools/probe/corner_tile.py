# -*- coding: utf-8 -*-
"""把成品的四角（各 600²）拼成一张图，看边缘拖影。"""
import sys
import numpy as np
from PIL import Image
import xisf_io

src = sys.argv[1]
out = sys.argv[2]
S = int(sys.argv[3]) if len(sys.argv) > 3 else 600

img = np.asarray(xisf_io.load_xisf(src)[0], np.float64)
h, w = img.shape
lo = float(np.median(img))
hi = float(np.percentile(img, 99.5))


def stretch(a):
    d = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1e-9) * 8, 0, None)) / np.arcsinh(8.0)
    return (np.clip(d, 0, 1) * 255).astype(np.uint8)


tiles = [stretch(img[0:S, 0:S]), stretch(img[0:S, w - S:w]),
         stretch(img[h - S:h, 0:S]), stretch(img[h - S:h, w - S:w])]
sep = np.full((S, 6), 255, np.uint8)
row1 = np.hstack([tiles[0], sep, tiles[1]])
row2 = np.hstack([tiles[2], sep, tiles[3]])
gap = np.full((6, row1.shape[1]), 255, np.uint8)
Image.fromarray(np.vstack([row1, gap, row2])).save(out)
print('saved', out, img.shape)
