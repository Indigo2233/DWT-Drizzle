# -*- coding: utf-8 -*-
"""诊断：为什么这些帧配准失败。逐帧跑 trireg，打印星数/内点/尺度/角度。"""
import sys
import os
import glob
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import matchref as M
import trireg

CAL = r'G:\HFG1-3\calibrated'


def find(pat):
    hits = glob.glob(os.path.join(CAL, '*', pat + '*.xisf'))
    return hits[0] if hits else None


def probe(ref_img, img, tag):
    sr = M.find_stars(ref_img)
    ss = M.find_stars(img)
    print('%-8s 星点 ref=%d  src=%d' % (tag, len(sr), len(ss)), end='  ')
    best = None
    for mp, ip, tol in ((90, 2.5, 0.03), (180, 4.0, 0.05), (250, 6.0, 0.08)):
        try:
            tr = trireg.solve(ss, sr, max_pts=mp, tol=tol, inlier_px=ip)
        except Exception as e:
            print('tri err %s' % e)
            return
        if tr:
            line = 'mp%d: n=%d k=%.4f th=%.1f rms=%.2f' % (
                mp, tr['n_in'], tr['scale'], tr['theta'], tr['rms'])
            print(line, end='; ')
            if best is None or tr['n_in'] > best['n_in']:
                best = tr
        else:
            print('mp%d: 无解' % mp, end='; ')
    print()


def main():
    ref = M._load(find('2026-11-03_22-16-43_O'))
    good = M._load(find('2026-11-03_22-16-43_O'))
    print('参考帧 = 自己（应 100% 命中）')
    probe(ref, good, 'self')
    for f in ['2026-11-04_23-12-38_O', '2026-11-05_22-25-27_O',
              '2026-11-06_22-51-42_O', '2026-11-11_00-40-37_O',
              '2026-11-12_00-12-53_O', '2026-11-01_23-26-22_H',
              '2026-11-06_23-57-31_H']:
        p = find(f)
        if not p:
            print(f, '找不到文件')
            continue
        probe(ref, M._load(p), f.split('_')[0])


if __name__ == '__main__':
    main()
