#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""彩机 ↔ 黑白隔离自测（随包发布）：不需要素材，几秒钟跑完。

检查三件事：
  1. 黑白流程里「满月帧补偿」(detail_snr) 一律为 0；
  2. 彩机流程里它照常生效；
  3. 离开彩机流程会清掉 CFA 相关环境变量。

用法：  python selftest_mono_isolation.py
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "pydeps"), str(HERE), str(HERE / "core"),
                str(HERE / "gpudeps")]

import pws  # noqa: E402


class _P:
    detail_snr = 1.0


def main() -> int:
    ok = True
    saved = dict(pws._CFA)

    # 1) 黑白：pattern 为空 -> beta 必须为 0
    pws._CFA['pattern'] = None
    b_mono = pws._detail_beta(_P())
    print("  黑白 detail_snr=1 -> beta = %.3f" % b_mono)
    if b_mono != 0.0:
        print("  [FAIL] 黑白流程仍然吃到了 beta")
        ok = False

    # 2) 彩机：pattern 非空 -> 照常生效
    pws._CFA['pattern'] = 'RGGB'
    b_cfa = pws._detail_beta(_P())
    print("  彩机 detail_snr=1 -> beta = %.3f" % b_cfa)
    if abs(b_cfa - 1.0) > 1e-12:
        print("  [FAIL] 彩机流程里 beta 没生效")
        ok = False

    # 3) 环境变量清理
    keys = ("DWT_DETAIL_SNR", "DWT_CFA_CHANNELS", "DWT_CFA_K", "DWT_CFA_RAW")
    for k in keys:
        os.environ[k] = "1"
    pws._clear_cfa_env()
    left = [k for k in keys if k in os.environ]
    print("  清理后残留：", left or "无")
    if left:
        print("  [FAIL] 环境变量没清干净")
        ok = False

    pws._CFA.update(saved)
    print("MONO-ISOLATION-OK" if ok else "MONO-ISOLATION-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
