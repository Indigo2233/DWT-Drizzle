# -*- coding: utf-8 -*-
"""按夜分组叠加：每一夜单独叠一版，再把这些夜成品配准合成一张。

为什么要这么干：不同夜的平场残差、天空梯度、透明度都不一样，
混在一起叠时它们会互相污染；先分夜、再合成，可以把「跨夜失配」
隔离在合成那一步，而不是让它进到每一帧的权重里。

用法：
    tools/stack_nights.py --photos G:\\HFG1-3\\calibrated --out G:\\HFG1\\lab\\NIGHTS \\
        --tag on --pick O --limit 60 --drizzle --drizzle-scale 2 --drizzle-pix 1.0 --bgdeg 1

输出：
    <out>\\night_<日期>\\PWS_<tag>_<日期>.xisf   每晚一版
    <out>\\PWS_<tag>_nights.xisf                 合成版
    <out>\\nights_report.csv                    每晚的帧数/FWHM/权重
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, 'core'), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import xisf_io

PYEXE = os.path.join(ROOT, 'py', 'python.exe')
MAIN = os.path.join(ROOT, 'main.py')
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def night_of(path):
    m = DATE_RE.search(os.path.basename(path))
    return m.group(1) if m else 'unknown'



def _plane(a, block=64, star_k=3.0):
    """用 64px 块中位拟合平面（跳过星点），返回 (coef, sky_median)。"""
    from scipy import ndimage
    h, w = a.shape
    gh, gw = max(1, h // block), max(1, w // block)
    g = a[:gh * block, :gw * block].reshape(gh, block, gw, block)
    med = np.median(g, axis=(1, 3))
    base = np.median(med)
    dev = np.abs(med - base)
    keep = dev < star_k * max(1.4826 * np.median(dev), 1e-9)
    yy, xx = np.mgrid[0:gh, 0:gw]
    M = np.c_[np.ones(gh * gw), (yy - gh / 2).ravel() / gh, (xx - gw / 2).ravel() / gw]
    sel = keep.ravel()
    if sel.sum() < 10:
        return (None, float(base))
    coef, *_ = np.linalg.lstsq(M[sel], med.ravel()[sel], rcond=None)
    return (coef, float(base))


def _plane_img(coef, shape, block=64):
    from scipy import ndimage
    h, w = shape
    gh, gw = max(1, h // block), max(1, w // block)
    yy, xx = np.mgrid[0:gh, 0:gw]
    M = np.c_[np.ones(gh * gw), (yy - gh / 2).ravel() / gh, (xx - gw / 2).ravel() / gw]
    p = (M @ coef).reshape(gh, gw)
    p = ndimage.zoom(p, (h / gh, w / gw), order=1)
    return p[:h, :w]


def match_bg(img, ref, log=print):
    """把一副成品的天空电平与平面坡对齐到 ref（只动一阶面+常数）。"""
    ca, sa = _plane(img)
    cb, sb = _plane(ref)
    if ca is None or cb is None:
        k = sb / max(sa, 1e-12)
        return img * k
    img2 = img - (_plane_img(ca, img.shape) - _plane_img(cb, ref.shape))
    return img2 * (sb / max(sa, 1e-12))

def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='nights')
    ap.add_argument('--pick', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per-night', type=int, default=0, help='每夜最多取多少帧')
    ap.add_argument('--min-frames', type=int, default=3)
    ap.add_argument('--match-bg', action='store_true', help='合成前把每晚的天空电平/坡对齐（实测常更差，默认关）')
    a, extra = ap.parse_known_args()

    import pws
    files, _ = pws.collect_frames(a.photos)
    groups = pws.group_frames_by_filter(files)
    pool = []
    for k, v in groups.items():
        if a.pick and k.upper() != a.pick.upper():
            continue
        pool.extend(v)
    pool.sort()
    if a.limit:
        pool = pool[:a.limit]
    by_night = {}
    for f in pool:
        by_night.setdefault(night_of(f), []).append(f)
    nights = [(d, v) for d, v in sorted(by_night.items()) if len(v) >= a.min_frames]
    print('夜数 %d，总帧 %d' % (len(nights), sum(len(v) for _, v in nights)))
    if not nights:
        return 1

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([os.path.join(ROOT, 'pydeps'),
                                         os.path.join(ROOT, 'core'), ROOT])
    env['PYTHONIOENCODING'] = 'utf-8'
    rows = []
    prods = []
    for d, fs_list in nights:
        use = fs_list[:a.per_night] if a.per_night else fs_list
        out = os.path.join(a.out, 'night_' + d)
        os.makedirs(out, exist_ok=True)
        tag = '%s_%s' % (a.tag, d.replace('-', ''))
        cmd = [PYEXE, '-B', MAIN, '--cli', '--photos', a.photos, '--out', out,
               '--tag', tag] + (['--pick', a.pick] if a.pick else []) + extra
        print('  叠 %s（%d 帧）…' % (d, len(use)), flush=True)
        # 用 hardlink 目录把这一夜单独喂进去
        link_dir = os.path.join(a.out, '_in_' + d)
        os.makedirs(link_dir, exist_ok=True)
        for f in use:
            dst = os.path.join(link_dir, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except OSError:
                    import shutil
                    shutil.copy2(f, dst)
        cmd[cmd.index('--photos') + 1] = link_dir
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        prod = os.path.join(out, 'PWS_%s.xisf' % tag)
        line = ''
        for ln in (r.stdout or '').splitlines():
            if '[输出] 成品星点' in ln:
                line = ln.strip()
        if not os.path.exists(prod):
            print('    失败：', (r.stdout or '')[-400:], (r.stderr or '')[-300:])
            continue
        m = re.search(r'FWHM ([\d.]+)px', line)
        fw = float(m.group(1)) if m else float('nan')
        neff_p = os.path.join(out, 'PWS_%s_neff.npy' % tag)
        neff = float(np.median(np.load(neff_p))) if os.path.exists(neff_p) else float(len(use))
        rows.append((d, len(use), fw, neff))
        prods.append((d, prod, neff))
        print('    %s  帧 %d  FWHM %.2f  N_eff %.1f' % (d, len(use), fw, neff), flush=True)

    if len(prods) < 2:
        print('夜数不足，退化为单夜成品：', prods)
        return 0

    import matchref
    ref_path = prods[0][1]
    ref = load(ref_path)
    acc = None
    wsum = 0.0
    for d, p, neff in prods:
        img = load(p)
        if img.shape == ref.shape:
            aligned = img
            cov = np.ones_like(img)
            print('  %s 与原网格一致，直接合并' % d)
        else:
            aligned, info = matchref.align(img, ref, log=print)
            if aligned is None:
                print('  %s 配准失败，跳过' % d)
                continue
            cov = info.get('cov')
            if cov is None:
                cov = np.ones_like(aligned)
            print('  %s 配准：k=%.4f θ=%.2f° 残差 %.2fpx 覆盖 %.1f%%'
                  % (d, info.get('k', 1), info.get('theta', 0), info['rms'], 100 * info['cover']))
        if a.match_bg:
            aligned = match_bg(aligned, ref, log=lambda *a, **k: None)
        w = float(neff)
        acc = aligned * w if acc is None else acc + aligned * w
        acc_cov = cov * w if 'acc_cov' not in dir() else acc_cov + cov * w
        wsum += w
    if acc is None or wsum <= 0:
        print('没有可合并的夜')
        return 1
    out_img = acc / np.maximum(acc_cov, 1e-9)
    out_p = __import__('pathlib').Path(a.out) / ('PWS_%s_nights.xisf' % a.tag)
    import pws as _pws
    _pws.save_linear_xisf(out_p, out_img.astype(np.float32), [{'exptime': 1200.0}],
                          'PWS nights %s' % a.tag, diag={'n_nights': len(prods)})
    print('合成完成：%s  %dx%d' % (out_p, out_img.shape[1], out_img.shape[0]))

    csv_p = os.path.join(a.out, 'nights_report.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        wr.writerow(['夜', '帧数', '成品FWHM(px)', 'N_eff中位'])
        for row in rows:
            wr.writerow([row[0], row[1], '%.2f' % row[2], '%.1f' % row[3]])
    print('报告：', csv_p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
