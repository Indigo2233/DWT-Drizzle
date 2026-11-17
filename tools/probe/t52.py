# -*- coding: utf-8 -*-
"""逐帧「星点总流量」vs「天空噪声」：看我们漏掉了哪个量。"""
import sys
import glob
import os

import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import matchref as M

CAL = r'G:\HFG1-3\calibrated'


def frame_stats(img):
    im = np.asarray(img, np.float64)
    sky = float(np.median(im))
    sig = 1.4826 * float(np.median(np.abs(im - sky)))
    mx = ndimage.maximum_filter(im, 5)
    pts = np.argwhere((im == mx) & (im > np.percentile(im, 99.5)))
    psf = ndimage.gaussian_filter(im - sky, 1.2)
    px, py = pts[:, 1], pts[:, 0]
    ok = (px > 8) & (py > 8) & (px < im.shape[1] - 8) & (py < im.shape[0] - 8)
    px, py = px[ok], py[ok]
    vals = psf[py, px]
    keep = vals > 5 * sig
    px, py = px[keep], py[keep]
    if len(px) < 20:
        return sky, sig, 0.0, 0
    s = 0.0
    for x, y in zip(px, py):
        t = psf[y - 6:y + 7, x - 6:x + 7]
        s += float(t.sum())
    return sky, sig, s / len(px), len(px)


def main():
    subs = sorted(glob.glob(os.path.join(CAL, 'Light_*O*')))
    rows = []
    for sub in subs[:4]:
        fs = sorted(glob.glob(os.path.join(sub, '*.xisf')))[::max(1, len(glob.glob(os.path.join(sub, '*.xisf'))) // 3)][:3]
        for f in fs:
            sky, sig, flux, n = frame_stats(M._load(f))
            rows.append((os.path.basename(os.path.dirname(f))[-24:], os.path.basename(f)[:19], sky, sig, flux, n))
    print('%-26s %-19s %10s %10s %10s %5s' % ('folder', 'frame', 'sky', 'sigma', 'starFlux', 'n'))
    for r in rows:
        print('%-26s %-19s %10.4g %10.4g %10.4g %5d' % r)
    if rows:
        sig = np.array([r[3] for r in rows])
        flux = np.array([r[4] for r in rows])
        print('\nsigma 相对散布 %.1f%%   星点流量相对散布 %.1f%%'
              % (100 * sig.std() / sig.mean(), 100 * flux.std() / flux.mean()))


if __name__ == '__main__':
    main()
