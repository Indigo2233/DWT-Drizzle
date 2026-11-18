# -*- coding: utf-8 -*-
"""回归：位移表比帧数短时不能崩（用户现场 IndexError: index 129 ... size 129）

构造 6 帧、dx/dy 只给 5 个，stack_drizzle 应当：
  · 不抛异常
  · 打一行「位移表长度与帧数不一致」
  · 用前 5 帧出结果（第 6 帧当作不参与）
"""
import sys
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import drizzle

rng = np.random.default_rng(11)


class FakeFS:
    def __init__(self, n=6, h=48, w=40, sky=0.01):
        self.n = n
        self.h, self.w = h, w
        self.shape = (h, w)
        self.mm = None
        self.path = ''
        self.sky0 = sky
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.full((h, w), sky, np.float32)
        for _ in range(12):
            cy, cx = rng.integers(6, h - 6), rng.integers(6, w - 6)
            amp = rng.uniform(0.2, 1.0)
            base += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)).astype(np.float32)
        self.arrays = [base + rng.normal(0, 3e-4, (h, w)).astype(np.float32) for _ in range(n)]

    def strip(self, i, j0, j1):
        return self.arrays[i][j0:j1]

    def __len__(self):
        return self.n


def main():
    fs = FakeFS()
    n, h, w = fs.n, fs.h, fs.w
    plan = {'s': 1.0, 'ox': 0.0, 'oy': 0.0, 'W': w, 'H': h}
    R = np.full((h, w), 0.6, np.float32)
    C = np.linspace(0.9, 1.1, n)
    S = np.ones(n)
    # 少给一个位移：n-1
    dx = np.zeros(n - 1)
    dy = np.zeros(n - 1)
    dx[1:] = 0.5
    logs = []
    out = drizzle.stack_drizzle(fs, R, C, S, dx, dy, plan, 14.0, 2.0, 3.5, 10.0,
                               0.01, 1.0, 32, theta=np.zeros(n - 1),
                               scale=np.ones(n - 1), native_shape=(h, w),
                               log=logs.append)
    txt = '\n'.join(logs)
    assert '位移表长度与帧数不一致' in txt, txt
    assert out['stack'].shape == (h, w)
    assert np.isfinite(out['stack']).all()
    print('LOGFIX-OK', [s for s in logs if '位移表' in s][0])


if __name__ == '__main__':
    main()
