# -*- coding: utf-8 -*-
import sys
import cupy
d = cupy.cuda.runtime.getDeviceProperties(0)
name = d.get("name") or d.get(b"name") or "?"
if isinstance(name, bytes):
    name = name.decode("utf-8", "replace")
print("    CuPy", cupy.__version__, "设备", name)
a = cupy.arange(6, dtype="f4")
print("    GPU 运算正常，求和 =", float(a.sum()))
