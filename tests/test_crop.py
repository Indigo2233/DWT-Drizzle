# -*- coding: utf-8 -*-
"""自动裁边的自检：合成一张「中间被盖满、四角是空的」覆盖图，看能不能裁回来。

    python tests/test_crop.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'core'))
sys.path.insert(0, str(HERE))

import cropcover


def main():
    h, w = 400, 500
    cov = np.zeros((h, w), np.float32)
    # 真正的完全覆盖区：y 60..340, x 80..420
    cov[60:340, 80:420] = 20.0
    # 边缘薄覆盖带（只被少数帧盖到）
    cov[40:60, 60:440] = 4.0
    cov[340:370, 60:440] = 6.0
    # 撒一些针孔（真实 neff 图里到处都是）
    rng = np.random.default_rng(0)
    yy = rng.integers(60, 340, 400)
    xx = rng.integers(80, 420, 400)
    cov[yy, xx] = 0.0

    box = cropcover.crop_box(cov, 0.9)
    assert box is not None, '裁边没找到框'
    y0, y1, x0, x1 = box
    print('框 y %d..%d  x %d..%d  → %d×%d' % (y0, y1, x0, x1, y1 - y0, x1 - x0))
    # 不该越界到薄覆盖带
    assert y0 >= 60 and y1 <= 340, '裁到了薄覆盖带（上/下）'
    assert x0 >= 80 and x1 <= 420, '裁到了薄覆盖带（左/右）'
    # 也不该缩得太小
    assert (y1 - y0) >= 200 and (x1 - x0) >= 250, '裁得太狠'

    # 框内不该留下成片的空洞
    sub = cov[y0:y1, x0:x1]
    assert (sub >= 18).mean() > 0.95, '框内空洞太多（%.1f%%）' % (
        100 * (sub < 18).mean())

    # 全空 / 全满 的边界情况
    assert cropcover.crop_box(np.zeros((10, 10), np.float32)) is None
    full = np.full((40, 60), 12.0, np.float32)
    b2 = cropcover.crop_box(full, 0.9)
    assert b2 == (0, 40, 0, 60), b2
    print('全满图 →', b2)
    print('crop selftest 通过')


if __name__ == '__main__':
    main()
