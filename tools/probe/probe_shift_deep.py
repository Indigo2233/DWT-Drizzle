# -*- coding: utf-8 -*-
"""按真实条件（跨夜混合、单一参考帧）逐帧计时，找出慢帧的共性。"""
import sys
import time
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle

if __name__ == '__main__':
    files, _ = pws.collect_frames(r'G:\HFG1-3\calibrated')
    g = pws.group_frames_by_filter(files)
    o = g['O'][::7][:30]
    fs = pws.prepare_frames(o, 'pt', 0, None)
    S = np.median(fs.sigma) / np.maximum(fs.sigma, 1e-30)
    seed = int(np.argmax(S))
    stars = pws.build_star_table(fs.frame(seed), 120)
    s_ref = drizzle._detect_stars(fs.frame(seed))
    ref_img = fs.frame(seed)
    refc = (ref_img.shape[0] / 2.0, ref_img.shape[1] / 2.0)
    print('参考帧 #%d  共 %d 帧' % (seed, len(fs)))
    tot = 0.0
    for i in range(len(fs)):
        img = fs.frame(i)
        s_i = drizzle._detect_stars(img)
        t0 = time.time()
        info = drizzle._rot_hist(s_ref, s_i) if len(s_i) >= 8 else None
        t1 = time.time()
        kind = 'rot_hist ok' if info is not None else 'rot_hist 失败'
        if info is None and len(s_i) >= 8:
            best = drizzle._scale_scan(s_ref, img)
            t2 = time.time()
            kind = 'scale_scan %s (%.1fs)' % ('ok' if best else '无解', t2 - t1)
            if best is not None:
                sc = best[0]
                info = drizzle._rot_hist(s_ref, s_i * sc)
                kind += ' sc=%.4f' % sc
        tot += time.time() - t0
        print('  #%3d 星 %2d  %.2fs  %s' % (i, len(s_i), time.time() - t0, kind))
    print('合计 %.1fs  %.2fs/帧' % (tot, tot / len(fs)))
