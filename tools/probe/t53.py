# -*- coding: utf-8 -*-
"""逐帧吞吐（星点流量）与噪声：口径修正版（扣天空环、取最亮 300 颗中位）。"""
import sys, glob, os
import numpy as np
from scipy import ndimage
sys.path.insert(0, 'core'); sys.path.insert(0, '.')
import matchref as M

CAL = r'G:\HFG1-3\calibrated'

def stats(img):
    im = np.asarray(img, np.float64)
    sky = float(np.median(im))
    sig = 1.4826 * float(np.median(np.abs(im - sky)))
    sm = ndimage.gaussian_filter(im - sky, 1.2)
    mx = ndimage.maximum_filter(sm, 5)
    pk = np.argwhere((sm == mx) & (sm > 20 * sig))
    if len(pk) < 300:
        return sky, sig, np.nan, len(pk)
    vals = sm[pk[:, 0], pk[:, 1]]
    order = np.argsort(vals)[::-1][:300]
    pk = pk[order]
    yy, xx = np.mgrid[-6:7, -6:7]
    ap = (yy * yy + xx * xx) <= 36
    fl = []
    for y, x in pk:
        if y < 8 or x < 8 or y >= im.shape[0] - 8 or x >= im.shape[1] - 8:
            continue
        t = im[y - 6:y + 7, x - 6:x + 7] - sky
        fl.append(float(t[ap].sum()))
    return sky, sig, float(np.median(fl)), len(fl)

def main():
    rows = []
    for sub in sorted(glob.glob(os.path.join(CAL, 'Light_*')))[:6]:
        fs = sorted(glob.glob(os.path.join(sub, '*.xisf')))
        if not fs:
            continue
        for f in [fs[0], fs[len(fs) // 2], fs[-1]]:
            sky, sig, flux, n = stats(M._load(f))
            rows.append((os.path.basename(sub)[-22:], os.path.basename(f)[:19], sig, flux, n))
    print('%-24s %-19s %9s %9s %5s' % ('chan/night', 'frame', 'sigma', 'flux300', 'n'))
    for r in rows:
        print('%-24s %-19s %9.4g %9.4g %5d' % r)
    sg = np.array([r[2] for r in rows]); fx = np.array([r[3] for r in rows])
    ok = np.isfinite(fx)
    print('\nsigma 散布 %.1f%%   flux 散布 %.1f%%' % (100*sg.std()/sg.mean(), 100*fx[ok].std()/fx[ok].mean()))
    print('两者相关系数 %.2f' % np.corrcoef(sg[ok], fx[ok])[0,1])
    print('flux/sigma 比 散布 %.1f%%' % (100*(fx[ok]/sg[ok]).std()/ (fx[ok]/sg[ok]).mean()))

if __name__ == '__main__':
    main()
