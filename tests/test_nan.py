# -*- coding: utf-8 -*-
"""NaN 变换的回归测试：有帧位移/旋转没测出来（NaN）时，叠加不能崩。

这不是假想场景：跨夜素材里总有几帧星点对不上，measure_shifts 会把它们标成
NaN。以前 `_rot_view` / `_native_band` 里 `int(round(float(rot)))` 遇到 NaN 直接
抛 `ValueError: cannot convert float NaN to integer`。

    python tests/test_nan.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'core'))
sys.path.insert(0, str(HERE))

import drizzle


class FakeFS:
    """只实现 stack_drizzle 用到的接口。"""

    def __init__(self, frames, sky0=1e-2):
        self.frames = frames
        self.shape = frames.shape[1:]
        self.sky0 = sky0
        self.off = np.zeros(len(frames), np.float64)
        self.arrays = frames
        self.parallelable = False

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def strip(self, i, j0, j1):
        return self.frames[i][j0:j1]


def main():
    rng = np.random.default_rng(0)
    h, w, n = 120, 160, 4
    frames = (rng.normal(0, 1e-3, (n, h, w)) + 1e-2).astype(np.float32)
    for k in range(n):
        frames[k][40 + 3 * k:48 + 3 * k, 60:68] += 0.5

    dx = np.array([0.0, 1.5, np.nan, -2.0])
    dy = np.array([0.0, -1.0, np.nan, 0.5])
    th = np.array([0.0, 0.2, np.nan, 180.0])
    kk = np.array([1.0, 1.0, np.nan, 1.0])
    plan = drizzle.plan_output((h, w), dx, dy, 1, theta=th, k=kk)
    H, W = plan['H'], plan['W']
    R = np.zeros((H, W), np.float32)
    R[H // 3:2 * H // 3, W // 3:2 * W // 3] = 1.0
    C = np.array([1.0, 1.0, 1.0, 1.0])
    S = np.array([1.0, 1.0, 1.0, 1.0])
    T = np.array([1.0, np.nan, 0.9, 1.0])
    out = drizzle.stack_drizzle(
        FakeFS(frames), R, C, S, dx, dy, plan, 20.0, 2.0, 3.5, 0.0,
        1e-2, 1.0, 32, theta=th, scale=kk, T=T, fp=2.0, native_shape=(h, w))
    st = np.asarray(out['stack'])
    assert st.shape == (H, W), st.shape
    assert np.isfinite(np.median(st)), '成品里有 NaN'
    assert float(np.nanmax(st)) > 0.0, '成品整幅是 0，说明压根没叠进去'
    print('NaN 帧存在时叠加成功  %d×%d  中位 %.4g' % (W, H, float(np.median(st))))

    # 直接测翻转判据
    assert drizzle._is_rot180(np.nan) is False
    assert drizzle._is_rot180(180.0) is True
    assert drizzle._is_rot180(0.0) is False
    assert drizzle._is_rot180(-0.3) is False
    print('翻转判据对 NaN 安全')
    print('nan selftest 通过')


if __name__ == '__main__':
    main()
