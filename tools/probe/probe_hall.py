# -*- coding: utf-8 -*-
"""全量 H 组：逐帧配准结果统计（θ 分布、残差、匹配数），找出不可信帧。"""
import collections
import numpy as np
import core.pws as pws
import core.drizzle as dz


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H']
    print('H 帧数', len(h))
    fs = pws.prepare_frames(h, 'Hfull', 0, None)
    sig = np.asarray(fs.sigma, float)
    S = float(np.median(sig)) / np.maximum(sig, 1e-30)
    seed = int(np.argmax(S))
    print('seed', seed, 'sigma 范围 %.4g~%.4g' % (sig.min(), sig.max()))
    stars = pws.build_star_table(fs.frame(seed), n_star=120)
    sh = dz.measure_shifts(fs, stars, ref=seed, log=print)
    th = np.asarray(sh['theta'], float)
    sc = np.asarray(sh['scatter'], float)
    dx = np.asarray(sh['dx'], float)
    dy = np.asarray(sh['dy'], float)
    kk = np.asarray(sh.get('scale', np.ones(len(th))), float)
    fin = np.isfinite(th) & np.isfinite(sc)
    print('可用帧 %d/%d' % (fin.sum(), len(th)))
    flip = np.abs(((th + 180) % 360) - 180) > 90
    print('翻转帧 %d  近 0° %d' % (int(flip.sum()), int((~flip & fin).sum())))
    bad = fin & (sc > 0.6)
    print('scatter>0.6: %d   median %.3f  P90 %.3f' % (int(bad.sum()), np.nanmedian(sc), np.nanpercentile(sc, 90)))
    print('尺度偏离 1 超过 0.5%%: %d' % int((np.abs(kk - 1) > 5e-3).sum()))
    # 按 θ 分箱
    hh = collections.Counter(np.round(th[fin] / 5.0) * 5)
    print('θ 分箱(5°):', dict(sorted(hh.items(), key=lambda kv: -kv[1])[:12]))
    np.save(r'C:\Windows\Temp\h_theta.npy', th)
    np.save(r'C:\Windows\Temp\h_scatter.npy', sc)
    np.save(r'C:\Windows\Temp\h_dxdy.npy', np.c_[dx, dy])
    np.save(r'C:\Windows\Temp\h_scale.npy', kk)


if __name__ == '__main__':
    main()
