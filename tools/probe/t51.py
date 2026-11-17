# -*- coding: utf-8 -*-
"""按半径量星点轴比：单帧 vs 最新全量叠加成品（看边缘恶化到底多大）。"""
import sys
import glob

import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
sys.path.insert(0, 'tools/probe')
import matchref as M
import probe_pa


def report(name, img, bins=((0, 500), (500, 1000), (1000, 1500), (1500, 2500))):
    c = np.array(img.shape) / 2.0
    st = probe_pa.stars(img)
    if len(st) < 20:
        print(name, '星点太少', len(st))
        return
    r = np.linalg.norm(st[:, :2] - c, axis=1)
    print('%s  星点数 %d' % (name, len(st)))
    for lo, hi in bins:
        m = (r >= lo) & (r < hi)
        if m.sum() < 10:
            continue
        ba = st[m, 3] / np.maximum(st[m, 2], 1e-6)
        print('    r %5d-%5d  n=%4d  轴比中位 %.3f' % (lo, hi, m.sum(), float(np.median(ba))))


def main():
    raw = M._load(glob.glob(r'G:\HFG1-2\calibrated\Light_BIN-1_3072x2240_EXPOSURE-1200.00s_FILTER-H_mono_DATE-2026-10-05\*.xisf')[0])
    report('单帧 H(10-18)', raw)
    report('H 全量成品 (H_all_v2)', M._load(r'G:\HFG1\lab\H_all_v2\PWS_hall2.xisf'))
    report('drz3 (O 全量, 新)', M._load(r'G:\HFG1\drz3\PWS_drz.xisf'))


if __name__ == '__main__':
    main()
