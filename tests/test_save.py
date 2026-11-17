# -*- coding: utf-8 -*-
"""成品导出自检：save_linear_xisf 必须真的写出文件。

这条测试是被一次真实事故换来的：`import matchref` 被插进函数体却按模块级
缩进写，函数在 from-import 之后就结束了，导出代码变成永不执行的模块级
except 分支 —— 结果叠加完只有 _R.npy / _neff.npy，没有 .xisf，而且不报错。

    python tests/test_save.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'core'))
sys.path.insert(0, str(HERE))

import pws


def main():
    a = np.full((48, 64), 0.01, np.float32)
    a[24, 32] = 1.0
    metas = [{'exptime': 1200.0, 'filter': 'H'}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'PWS_test.xisf'
        pws.save_linear_xisf(p, a, metas, 'PWS A=20 B=2', diag={'A': 20, 'B': 2})
        assert p.exists(), 'save_linear_xisf 没写出文件'
        assert p.stat().st_size > 1000, '写出的文件太小'
        print('写出 %s  %d 字节' % (p.name, p.stat().st_size))

        # 读回来确认数据没坏
        import xisf_io
        back = np.asarray(xisf_io.read_xisf(str(p))[0].data, np.float64)
        assert back.shape == a.shape, '尺寸不对 %s' % (back.shape,)
        assert abs(float(back.max()) - 1.0) < 1e-3, '峰值丢了'
        print('回读 %d×%d  峰值 %.4f' % (back.shape[0], back.shape[1], back.max()))
    print('save selftest 通过')


if __name__ == '__main__':
    main()
