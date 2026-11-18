# -*- coding: utf-8 -*-
"""GPU 自检：只验证真正用到的那条路径（预编译内核），不需要 CUDA Toolkit。

历史坑：旧版用 cupyx.scatter_add（RawKernel → 运行时 JIT → 需要
nvrtc64_112_0.dll），于是「探测通过、跑到一半炸」。现在热路径是
cp.bincount，预编译内核，只要驱动够新就能跑。
"""
import sys

sys.path.insert(0, 'core')

import cupy
import gpu_dz

p = cupy.cuda.runtime.getDeviceProperties(0)
name = p.get("name") or p.get(b"name") or "?"
if isinstance(name, bytes):
    name = name.decode("utf-8", "replace")

free = gpu_dz.free_mem()
print("    CuPy", cupy.__version__, "设备", name)
if free:
    print("    可用显存 %.0f MB（本软件按其中 55%% 定分带预算，最多 3 GB）" % (free / 2**20))

idx = cupy.array([0, 1, 1, 3], dtype=cupy.int64)
w = cupy.array([1.0, 2.0, 3.0, 4.0], dtype=cupy.float64)
got = [float(x) for x in cupy.bincount(idx, weights=w, minlength=5).astype(cupy.float32)]
ok = got == [1.0, 5.0, 0.0, 4.0, 0.0]
print("    散射累加探针:", got, "->", "通过" if ok else "结果不对")
if not ok:
    raise SystemExit(2)
print("    GPU 运算正常（不需要 CUDA Toolkit / nvrtc）")
