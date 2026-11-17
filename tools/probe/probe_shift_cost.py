# -*- coding: utf-8 -*-
"""分夜量 measure_shifts 的成本：难帧到底贵在哪一步。"""
import sys
import time
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle


def frames(pat, n=12, tag='p'):
    files, _ = pws.collect_frames(r'G:\HFG1-3\calibrated')
    g = pws.group_frames_by_filter(files)
    sel = [f for f in g['O'] if pat in str(f)][:n]
    return pws.prepare_frames(sel, tag, 0, None)


def timeit(fs, tag):
    S = np.median(fs.sigma) / np.maximum(fs.sigma, 1e-30)
    seed = int(np.argmax(S))
    stars = pws.build_star_table(fs.frame(seed), 120)
    t0 = time.time()
    sh = drizzle.measure_shifts(fs, stars, ref=seed)
    dt = time.time() - t0
    n = len(fs)
    print('%-26s %2d 帧 %6.1fs  %.2fs/帧  翻转 %2d  尺度 %.3f~%.3f'
          % (tag, n, dt, dt / n,
             int(np.sum(np.abs(np.degrees(sh['theta'])) > 90)),
             np.min(sh['scale']), np.max(sh['scale'])))
    return dt / n


if __name__ == '__main__':
    import sys as _s
    pat = _s.argv[1] if len(_s.argv) > 1 else '2026-10-07'
    fs = frames(pat, 12, 'p')
    timeit(fs, pat + ' 批')
