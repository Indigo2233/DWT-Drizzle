# -*- coding: utf-8 -*-
"""把已有 XISF 的 bounds 改成真实数据范围（旧版成品在 PI 里显示全白时用）。

用法：
    py fix_bounds.py 输入.xisf [输出.xisf]

不给输出路径时写成 `<名字>_bounds.xisf`，不动原文件。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xisf_io import read_xisf, write_xisf  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + '_bounds.xisf')

    im = read_xisf(str(src))[0]
    a = np.asarray(im.data)
    v = a[np.isfinite(a)]
    if not v.size:
        print('没有有限像素，放弃')
        return 1
    lo, hi = float(v.min()), float(v.max())
    print('原 bounds %s  → 新 bounds (%.6g, %.6g)  shape %s'
          % (im.bounds, lo, hi, a.shape))
    im.bounds = (lo, hi) if hi > lo else (0.0, 1.0)
    write_xisf(str(out), im, compression=None)
    chk = read_xisf(str(out))[0]
    print('写出', out.name, ' 回读 bounds', chk.bounds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
