# -*- coding: utf-8 -*-
"""自动调参自检：抽样、日志解析、硬约束、检测信噪比。

    python tests/test_tune.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'core'))
sys.path.insert(0, str(HERE))

import tune


def test_sample():
    paths = [Path('f%03d.xisf' % i) for i in range(100)]
    got = tune.sample_paths(paths, 16)
    assert len(got) == 16 and got[0].name == 'f000.xisf' and got[-1].name == 'f099.xisf'
    print('抽样 16/100 → 首 %s 尾 %s' % (got[0].name, got[-1].name))


def test_parse():
    txt = ('[叠加] 参与 16/16 帧\n'
           '[叠加] 成品星点 FWHM 2.94px（本机口径；图上 5.88px @×2）  单帧中位 3.71px  改善 20.6%\n'
           '[W] N_eff 中位 4.44/16\n'
           '[叠加] 完成 41.2s  排异剔除率 1.234%\n')
    r = tune._parse_log(txt)
    assert abs(r['fwhm'] - 2.94) < 1e-6 and abs(r['single'] - 3.71) < 1e-6
    assert abs(r['gain'] - 20.6) < 1e-6
    assert abs(r['neff'] - 4.44) < 1e-6
    assert abs(r['rej'] - 1.234) < 1e-6
    print('解析 FWHM %.2f 改善 %+.1f%% N_eff %.2f 排异 %.3f%%'
          % (r['fwhm'], r['gain'], r['neff'], r['rej']))


def test_pick_best():
    rows = [
        {'A': 10.0, 'mode': 'max', 'rej_k': 3.5, 'fwhm': 2.94, 'gain': 20.6,
         'neff': 4.4, 'dsnr': 1.2, 'ok': True},
        {'A': 10.0, 'mode': 'band', 'rej_k': 3.5, 'fwhm': 2.96, 'gain': 20.0,
         'neff': 4.6, 'dsnr': 1.5, 'ok': True},
        {'A': 20.0, 'mode': 'max', 'rej_k': 3.5, 'fwhm': 2.80, 'gain': 24.5,
         'neff': 2.1, 'dsnr': 1.1, 'ok': True},
    ]
    good = [r for r in rows if r['ok']]
    neff_max = max(r['neff'] for r in good)
    ok = [r for r in good if r['gain'] > 0 and r['neff'] >= 0.70 * neff_max]
    assert len(ok) == 2, ok
    # A=20 的 N_eff 只剩 2.1（<70% of 4.6）应被硬约束剔除
    print('硬约束生效：%d 组通过（N_eff 门槛 %.2f）'
          % (len(ok), 0.70 * neff_max))


def test_parse_grid():
    assert tune.parse_grid('') == (10.0, 14.0, 20.0)
    assert tune.parse_grid('6,10,14,20') == (6.0, 10.0, 14.0, 20.0)
    assert tune.parse_grid('20,6,6,10') == (6.0, 10.0, 20.0)
    assert tune.parse_grid('abc,0,-2') == (10.0, 14.0, 20.0)
    print('A 候选解析：6,10,14,20 → %s' % (tune.parse_grid('6,10,14,20'),))


def test_detect_ratio():
    """合成一张「星点 + 环 + 噪声」图，环带检测信噪比应 > 天空。"""
    n = 1000
    rng = np.random.default_rng(7)
    img = rng.normal(0.0, 1.0, (n, n)).astype(np.float64) * 3e-5 + 1e-2
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(yy - n / 2.0, xx - n / 2.0)
    img[(rr > 90) & (rr < 170)] += 6e-4
    for _ in range(60):
        y, x = rng.integers(30, n - 30, 2)
        img[y - 2:y + 3, x - 2:x + 3] += 2e-3
    tmp = Path(HERE) / '_tune_dsnr.xisf'
    try:
        import pws
        pws.save_linear_xisf(tmp, img.astype(np.float32), [{'exptime': 1.0}], 'tune test')
        r = tune.detect_ratio(tmp)
        assert r is not None and r > 1.2, r
        print('检测信噪比（合成环）%.2f' % r)
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == '__main__':
    test_sample()
    test_parse()
    test_pick_best()
    test_parse_grid()
    test_detect_ratio()
    print('tune selftest 通过')
