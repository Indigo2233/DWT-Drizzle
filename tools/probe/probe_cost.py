# -*- coding: utf-8 -*-
"""量 measure_shifts 的每帧成本，以及逐帧畸变拟合占多少。"""
import sys
import time
import glob
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle


def build(n=20):
    files, _ = pws.collect_frames(r'G:\HFG1-3\calibrated')
    g = pws.group_frames_by_filter(files)
    h = g['O'][:n]
    fs = pws.prepare_frames(h, 'probe', 0, None)
    return fs


def run(fs, tag):
    S = np.median(fs.sigma) / np.maximum(fs.sigma, 1e-30)
    seed = int(np.argmax(S))
    stars = pws.build_star_table(fs.frame(seed), 120)
    t0 = time.time()
    sh = drizzle.measure_shifts(fs, stars, ref=seed)
    dt = time.time() - t0
    print('%-14s %d 帧  %.1fs  每帧 %.2fs  残差 %.3fpx'
          % (tag, len(fs), dt, dt / len(fs), np.nanmedian(sh['scatter'])))
    return dt


if __name__ == '__main__':
    fs = build(20)
    t1 = run(fs, '带畸变拟合')
    base = drizzle._fit_sim_dist
    def fast(p, q, c, rc, **kw):
        th, k, ux, uy, rms = drizzle._fit_sim(p, q)
        return th, k, ux, uy, rms, 0.0
    drizzle._fit_sim_dist = fast
    t2 = run(fs, '跳过畸变拟合')
    drizzle._fit_sim_dist = base
    print('跳过可省 %.1f%%' % (100 * (t1 - t2) / t1))
