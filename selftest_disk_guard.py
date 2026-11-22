#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""磁盘预检自测（随包发布）：不需要素材，几秒钟跑完。

用法：  python selftest_disk_guard.py
"""
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "pydeps"), str(HERE), str(HERE / "core"),
                str(HERE / "gpudeps")]

import pws  # noqa: E402


class _Log:
    def __init__(self):
        self.msgs = []

    def __call__(self, m):
        self.msgs.append(str(m))


def _with_free(bytes_free, fn):
    real = shutil.disk_usage
    shutil.disk_usage = lambda p: shutil._ntuple_diskusage(  # noqa: SLF001
        9 * 2 ** 30, 9 * 2 ** 30 - bytes_free, bytes_free)
    try:
        return fn()
    finally:
        shutil.disk_usage = real


def main() -> int:
    ok = True
    frames = [str(pathlib.Path(__file__).resolve())]   # 任意存在的文件
    out = HERE

    # 1) 空间充足：不报错，且日志里带「个平面」
    log = _Log()
    try:
        _with_free(50 * 2 ** 30, lambda: pws._check_disk_space(
            frames, out, 1.0, log, n_ch=3, px=26_000_000))
        line = log.msgs[-1]
        print("  空间充足 ->", line)
        if "个平面" not in line or "可用" not in line:
            print("  [FAIL] 日志格式不符合预期")
            ok = False
    except SystemExit as e:
        print("  [FAIL] 空间充足却报错：", e)
        ok = False

    # 2) 空间不足：必须拦截，且提示里带需要的 GB 与平面数
    log = _Log()
    try:
        _with_free(1 * 2 ** 30, lambda: pws._check_disk_space(
            frames, out, 2.0, log, n_ch=3, px=26_000_000))
        print("  [FAIL] 空间不足没有拦截")
        ok = False
    except SystemExit as e:
        msg = str(e)
        print("  空间不足 ->", msg.splitlines()[0])
        if "磁盘空间不足" not in msg or "个平面" not in msg:
            print("  [FAIL] 提示内容不符合预期")
            ok = False

    # 3) 通道数确实参与估算：1 个平面 vs 6 个平面，需求应差约 4 倍
    def need(n_ch):
        log = _Log()
        try:
            _with_free(500 * 2 ** 30, lambda: pws._check_disk_space(
                frames, out, 2.0, log, n_ch=n_ch, px=26_000_000,
                rgb=(n_ch > 1)))
        except SystemExit:
            pass
        txt = log.msgs[-1]
        return float(txt.split("预计需要 ")[1].split(" GB")[0])

    n1, n6 = need(1), need(3)
    print("  估算：1 平面 %.2f GB，6 平面 %.2f GB" % (n1, n6))
    if not (n6 > n1 * 2.5):
        print("  [FAIL] 平面数没有体现在估算里")
        ok = False

    print("DISK-GUARD-OK" if ok else "DISK-GUARD-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
