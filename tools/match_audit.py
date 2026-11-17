# -*- coding: utf-8 -*-
"""逐帧体检：哪些帧和参考图（master）是同一片天。

用法：
    match_audit.py <素材目录> <参考.xisf> [输出.csv] [--link-dir 目录]

原理：每帧只做「星点粗投票」——把帧缩到参考的像素尺度，取最亮的若干颗星，
两两差值投票求平移，命中数够多就算对得上。比完整配准快一个数量级（约 1-2 秒/帧）。
命中数 ≥ MIN_PAIR 判为可用；--link-dir 会把可用的帧硬链接过去，方便直接拿去叠加。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import matchref as M          # noqa: E402
import xisf_io                # noqa: E402

MIN_PAIR = 40


def stars_of(img, thr=99.5, n=600):
    return M.find_stars(img, thr, n)


def vote_pairs(sa, sb):
    """两团星点做平移投票，返回 (最佳位移, 命中对数)。"""
    if len(sa) < 12 or len(sb) < 12:
        return None, 0
    m = min(150, len(sa), len(sb))
    a, b = sa[:m], sb[:m]
    dx = np.round(b[:, 0][:, None] - a[:, 0][None, :]).ravel()
    dy = np.round(b[:, 1][:, None] - a[:, 1][None, :]).ravel()
    keep = np.abs(dx) < 4000
    dx, dy = dx[keep], dy[keep]
    if dx.size < 200:
        return None, 0
    grid = {}
    for i in range(dx.size):
        key = (dx[i], dy[i])
        grid[key] = grid.get(key, 0) + 1
    key, cnt = max(grid.items(), key=lambda kv: kv[1])
    shift = np.array(key, float)
    d, _ = cKDTree(b).query(a + shift, k=1, distance_upper_bound=3.0)
    return shift, int(np.isfinite(d).sum())


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    src_dir = Path(argv[1])
    ref_p = Path(argv[2])
    out_csv = Path(argv[3]) if len(argv) > 3 and not argv[3].startswith('--') else None
    link_dir = None
    if '--link-dir' in argv:
        link_dir = Path(argv[argv.index('--link-dir') + 1])
        link_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.rglob('*.xisf') if not p.name.startswith('PWS_'))
    if not files:
        print('没找到 xisf')
        return 1
    ref = M._load(ref_p)
    sr = stars_of(ref)
    print(f'参考 {ref_p.name}  {ref.shape[1]}×{ref.shape[0]}  星点 {len(sr)}')
    print(f'素材 {len(files)} 帧，判据：≥{MIN_PAIR} 对')
    rows, good = [], []
    for i, f in enumerate(files, 1):
        try:
            img = np.asarray(xisf_io.read_xisf(str(f))[0].data, np.float64)
        except Exception as exc:
            print(f'  [{i}/{len(files)}] {f.name[:40]} 读不动：{exc}')
            rows.append((f.name, -1, 0))
            continue
        k = 0.5 if img.shape[0] < ref.shape[0] * 0.8 else 1.0
        si = stars_of(M._grid(img, k))
        shift, n = vote_pairs(si, sr)
        rows.append((f.name, n, k))
        if n >= MIN_PAIR:
            good.append(f)
        print(f'  [{i}/{len(files)}] {f.name[:44]:<44} 对 {n:>4}  {"OK" if n >= MIN_PAIR else "--"}')

    if out_csv:
        with open(out_csv, 'w', encoding='utf-8') as fh:
            fh.write('file,pairs,scale\n')
            for name, n, k in rows:
                fh.write(f'{name},{n},{k}\n')
        print('写出', out_csv)
    if link_dir:
        for f in good:
            dst = link_dir / f.name
            if not dst.exists():
                try:
                    dst.hardlink_to(f)
                except OSError:
                    dst.symlink_to(f)
        print(f'硬链接 {len(good)} 帧 → {link_dir}')
    print(f'汇总：可用 {len(good)}/{len(files)} 帧')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
