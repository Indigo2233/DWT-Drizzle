# -*- coding: utf-8 -*-
"""叠加前自动调参：抽几帧快速试几组参数，按实测指标给建议。

用法（命令行）：
    main.py --cli --photos <calibrated> --pick O --out <目录> --tune 16

做法：从素材里均匀抽 N 帧（默认 16），裁一块中心区域（默认 1200²），
对每个候选组合（A × r_mode × rej_k，A 候选可用 --tune-a 指定）各跑一遍完整的叠加流程，记录：
    成品星点 FWHM、单帧中位 FWHM、改善百分比、N_eff 中位、排异率、暗弱结构检测信噪比
然后按「硬约束 + 排序」给出一条建议。

暗弱结构检测信噪比（dsnr）= 环带三尺度显著性 Z 的中位 / 天空区 Z 的 P90。
它专门用来分辨那些「不改星点锐度、但影响暗弱结构」的参数。

为什么要抽帧：全量 305 帧一次 9 分钟，18 组就是 2 个多小时；抽 16 帧每组 30~40 秒，
总 15 分钟左右。抽样只影响绝对数值，不影响参数之间的相对优劣。
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

FWHM_RE = re.compile(r'成品星点 FWHM ([\d.]+)px.*?单帧中位 ([\d.]+)px.*?改善 (-?[\d.]+)%')
NEFF_RE = re.compile(r'N_eff 中位 ([\d.]+)')
REJ_RE = re.compile(r'排异剔除率 ([\d.]+)%')

# 候选网格：A 是唯一的风格旋钮，r_mode 是两条不同取向的 R 场定义，
# rej_k 是排异强度（卫星/飞机/云影多时要调高）
DEFAULT_A = (10.0, 14.0, 20.0)
MODE_GRID = ('max', 'band')
REJ_GRID = (2.5, 3.5, 4.0)


def parse_grid(text, default=DEFAULT_A):
    """把 '6,10,14' 解析成有序去重的浮点列表；空值/非法值退回 default。"""
    if not text:
        return tuple(default)
    out = []
    for piece in str(text).replace('，', ',').split(','):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = float(piece)
        except ValueError:
            continue
        if v > 0 and not any(abs(v - x) < 1e-9 for x in out):
            out.append(v)
    return tuple(sorted(out)) or tuple(default)


def sample_paths(paths: List[Path], n: int) -> List[Path]:
    """均匀抽 n 帧（首尾都取到）。"""
    paths = list(paths)
    if n <= 0 or len(paths) <= n:
        return paths
    if n == 1:
        return [paths[0]]
    idx = [int(round(i * (len(paths) - 1) / (n - 1))) for i in range(n)]
    return [paths[i] for i in sorted(set(idx))]


def _parse_log(text: str) -> Dict:
    out: Dict = {}
    m = FWHM_RE.search(text)
    if m:
        out['fwhm'] = float(m.group(1))
        out['single'] = float(m.group(2))
        out['gain'] = float(m.group(3))
    m = NEFF_RE.search(text)
    if m:
        out['neff'] = float(m.group(1))
    m = REJ_RE.search(text)
    if m:
        out['rej'] = float(m.group(1))
    return out


def detect_ratio(xisf_path, scales=(8, 32, 128), frac_ring=(70, 200)) -> Optional[float]:
    """成品图的暗弱结构检测信噪比：环带 Z 中位 / 天空 Z 的 P90。

    Z 是三个尺度的 DoG 显著性平方和开根（各尺度按自身 MAD 归一）。
    环带取画幅中心附近，天空取半径大的区域；星点先膨涨掩掉。
    """
    try:
        import numpy as np
        from scipy import ndimage
        import xisf_io
    except Exception:
        return None
    try:
        a = np.asarray(xisf_io.read_xisf(str(xisf_path))[0].data, np.float64)
    except Exception:
        return None
    if a.size == 0:
        return None
    hp = a - ndimage.gaussian_filter(a, 3.0)
    smad = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    if not np.isfinite(smad) or smad <= 0:
        return None
    stars = ndimage.binary_dilation(hp > 8 * smad, np.ones((21, 21)))
    z = np.zeros_like(a)
    for s in scales:
        d = ndimage.gaussian_filter(a, s) - ndimage.gaussian_filter(a, 1.6 * s)
        mad = 1.4826 * np.median(np.abs(d - np.median(d)))
        if mad > 0:
            z += (d / mad) ** 2
    z = np.sqrt(z)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    # 环带/天空半径按画幅缩放：固定像素半径只在 2500² 以上的成品上成立
    base = min(h, w)
    r0 = frac_ring[0] / 2500.0 * base if base < 2500 else frac_ring[0]
    r1 = frac_ring[1] / 2500.0 * base if base < 2500 else frac_ring[1]
    ring = (rr > r0) & (rr < r1) & ~stars
    sky = (rr > base * 0.45) & ~stars
    if ring.sum() < 200 or sky.sum() < 200:
        return None
    zr = float(np.median(z[ring]))
    zs = float(np.percentile(z[sky], 90))
    if zs <= 0:
        return None
    return zr / zs


def _run_one(exe: str, root: Path, photos: Path, out: Path, tag: str,
             crop: int, A: float, mode: str, rej_k: float, p) -> Dict:
    cmd = [exe, '-B', str(root / 'main.py'), '--cli',
           '--photos', str(photos), '--out', str(out), '--tag', tag,
           '--crop', str(crop),
           '--drizzle', '--drizzle-scale', str(p.drizzle_scale),
           '--drizzle-pix', str(p.drizzle_pix),
           '--bgdeg', str(p.bgdeg), '--flux-pow', str(p.flux_pow),
           '--A', str(A), '--r-mode', mode, '--rej-k', str(rej_k),
           '--gpu', str(p.gpu),
           '--frames-dir', str(out / tag)]     # 每个候选自己的 memmap，避免并发撞车。
    # 彩机素材必须把 CFA 参数带上，否则量的是没拆色的马赛克原图（FWHM/降噪全失真）
    _ce = str(getattr(p, 'cfa', 'off') or 'off').strip().lower()
    if _ce and _ce != 'off':
        cmd += ['--cfa', 'auto' if _ce in ('auto', '1', 'true', 'on') else _ce.upper()]
        _cd = str(getattr(p, 'cfa_debayer', 'bilinear') or 'bilinear').strip().lower()
        if _cd and _cd != 'bilinear':
            cmd += ['--cfa-debayer', _cd]
        _cw = str(getattr(p, 'cfa_weights', 'shared') or 'shared').strip().lower()
        if _cw and _cw != 'shared':
            cmd += ['--cfa-weights', _cw]
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    if _ce and _ce != 'off':
        env['DWT_CFA_CHANNELS'] = 'G'      # 调参只要亮度口径，别三通道全跑
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=str(root), env=env)
    txt = (r.stdout or '') + (r.stderr or '')
    rec = _parse_log(txt)
    rec['A'] = A
    rec['mode'] = mode
    rec['rej_k'] = rej_k
    rec['sec'] = time.time() - t0
    rec['ok'] = r.returncode == 0 and bool(rec.get('fwhm'))
    if rec['ok']:
        prod = out / ('PWS_%s.xisf' % tag)
        if not prod.exists() and _ce and _ce != 'off':
            # 彩机出的是 _R/_G/_B（+ 合成 RGB）：拿 G 当亮度口径与其他组对比
            prod = out / ('PWS_%s_G.xisf' % tag)
        if not prod.exists():
            cand = sorted(out.glob('PWS_%s*.xisf' % tag))
            prod = cand[0] if cand else None
        if prod is not None and Path(prod).exists():
            dr = detect_ratio(prod)
            if dr is not None:
                rec['dsnr'] = dr
    else:
        rec['err'] = txt[-400:]
    return rec


def run_tune(p, n_frames: int = 16, crop: int = 1200, log=print) -> Dict:
    """跑网格、打印表格、返回 {'rows': [...], 'best': {...}}。"""
    from pws import collect_frames, group_frames_by_filter

    files, _ = collect_frames(str(p.photos))
    groups = group_frames_by_filter(files)
    pick = (getattr(p, 'pick', '') or '').strip()
    if pick:
        names = [x.strip().lower() for x in re.split(r'[;,]', pick) if x.strip()]
        groups = {k: v for k, v in groups.items()
                  if k.lower() in names or (not k and ('' in names or '无滤镜信息' in names))}
    if not groups:
        log('[调参] 没有可用素材（检查素材目录与通道勾选）')
        return {'rows': [], 'best': None}
    ch, lst = max(groups.items(), key=lambda kv: len(kv[1]))
    sampled = sample_paths(lst, n_frames)
    a_grid = parse_grid(getattr(p, 'tune_a', ''))
    n_run = len(a_grid) * len(MODE_GRID) * len(REJ_GRID)
    log('[调参] 从 %s 通道 %d 帧里抽 %d 帧，裁剪 %d×%d'
        % (ch or '无滤镜信息', len(lst), len(sampled), crop, crop))
    _jobs0 = max(1, min(4, int(getattr(p, 'tune_jobs', 2) or 1)))
    log('[调参] A 候选 %s × R 场 %s × 排异 %s → 共 %d 组，%d 路并发，预计 %d 分钟'
        % (','.join('%g' % a for a in a_grid), '/'.join(MODE_GRID),
           '/'.join('%g' % r for r in REJ_GRID), n_run, _jobs0,
           max(1, round(n_run * 45 / 60.0 / _jobs0))))

    root = Path(__file__).resolve().parent.parent
    exe = sys.executable
    tmp = Path(tempfile.mkdtemp(prefix='dwt_tune_'))
    links = tmp / 'frames'
    links.mkdir()
    for f in sampled:
        try:
            os.link(str(f), str(links / Path(f).name))
        except Exception:
            shutil.copy2(str(f), str(links / Path(f).name))

    out = Path(str(p.out)) / '_tune'
    out.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(getattr(p, 'tune_jobs', 2) or 1))
    jobs = min(jobs, 4, max(1, len(a_grid) * len(MODE_GRID) * len(REJ_GRID)))
    rows: List[Dict] = []
    combo = [(A, mode, rk) for A in a_grid for mode in MODE_GRID for rk in REJ_GRID]
    log('[调参] 并发 %d 路' % jobs)
    try:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {}
            for A, mode, rk in combo:
                tag = 'tune_a%g_%s_k%g' % (A, mode, rk)
                log('[调参] 排队 A=%g r_mode=%s rej_k=%g' % (A, mode, rk))
                futs[ex.submit(_run_one, exe, root, links, out / tag, tag,
                               crop, A, mode, rk, p)] = (A, mode, rk)
            for fut in as_completed(futs):
                rec = fut.result()
                rows.append(rec)
                if rec['ok']:
                    log('[调参] ✔ A=%g %s k=%g → FWHM %.2fpx（改善 %+.1f%%） N_eff %.1f'
                        ' 排异 %.2f%% 检测 %.2f %.0fs'
                        % (rec['A'], rec['mode'], rec['rej_k'], rec['fwhm'], rec['gain'],
                           rec.get('neff', 0), rec.get('rej', 0),
                           rec.get('dsnr', 0) or 0, rec['sec']))
                else:
                    log('[调参] ✘ A=%g %s k=%g 失败：%s'
                        % (rec['A'], rec['mode'], rec['rej_k'], rec.get('err', '')[-160:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    good = [r for r in rows if r['ok']]
    best = None
    if good:
        neff_max = max(r.get('neff', 0.0) for r in good)
        # 硬约束：FWHM 得有改善；N_eff 不能塌到最高者的 70% 以下
        ok = [r for r in good if r['gain'] > 0 and r.get('neff', 0) >= 0.70 * neff_max]
        pool = ok or good
        # 主序按 FWHM 改善，同等（0.3% 以内）时看检测信噪比
        top = max(r['gain'] for r in pool)
        near = [r for r in pool if r['gain'] >= top - 0.3]
        best = max(near, key=lambda r: (r.get('dsnr', 0) or 0, r['gain']))
        log('')
        log('[调参] A      r_mode  rej_k  FWHM(px)  改善     N_eff   排异    检测  用时')
        for r in sorted(rows, key=lambda x: (x['A'], x['mode'], x['rej_k'])):
            if r['ok']:
                log('[调参] %-6g %-7s %-6g %-9.2f %+6.1f%%  %-7.1f %-6.2f %-6.2f %.0fs'
                    % (r['A'], r['mode'], r['rej_k'], r['fwhm'], r['gain'],
                       r.get('neff', 0.0), r.get('rej', 0.0),
                       r.get('dsnr', 0.0) or 0.0, r['sec']))
            else:
                log('[调参] %-6g %-7s %-6g 失败' % (r['A'], r['mode'], r['rej_k']))
        fwhms = [r['fwhm'] for r in good]
        spread = (max(fwhms) - min(fwhms)) / max(min(fwhms), 1e-9) * 100.0
        log('[调参] 建议：A=%g  R 场定义=%s  rej_k=%g'
            '（成品 FWHM %.2fpx，改善 %+.1f%%，N_eff %.1f，检测信噪比 %.2f）'
            % (best['A'], best['mode'], best['rej_k'], best['fwhm'], best['gain'],
               best.get('neff', 0), best.get('dsnr', 0) or 0))
        if spread < 1.5:
            log('[调参] 注意：各组的成品 FWHM 只差 %.1f%%（在抽样噪声内），'
                '说明这批数据对这几个参数不敏感，保持当前设置即可' % spread)
        else:
            log('[调参] 各组 FWHM 相差 %.1f%%，这个差异是真的' % spread)
        log('[调参] TUNE-BEST ' + json.dumps(
            {'A': best['A'], 'r_mode': best['mode'], 'rej_k': best['rej_k'],
             'fwhm': best['fwhm'], 'gain': best['gain'],
             'neff': best.get('neff', 0), 'dsnr': best.get('dsnr', 0)}, ensure_ascii=False))

    try:
        csv = Path(str(p.out)) / 'tune_report.csv'
        with open(csv, 'w', encoding='utf-8-sig', newline='') as fp:
            import csv as _csv
            w = _csv.writer(fp)
            w.writerow(['A', 'r_mode', 'rej_k', '成品FWHM(px)', '单帧FWHM(px)',
                        '改善(%)', 'N_eff中位', '排异(%)', '检测信噪比', '用时(s)', '成功'])
            for r in rows:
                w.writerow([r['A'], r['mode'], r['rej_k'],
                            round(r.get('fwhm', 0), 3), round(r.get('single', 0), 3),
                            round(r.get('gain', 0), 1), round(r.get('neff', 0), 2),
                            round(r.get('rej', 0), 3), round(r.get('dsnr', 0) or 0, 3),
                            round(r['sec'], 1), int(r['ok'])])
        log('[调参] 明细已写到 %s' % csv)
    except Exception as e:
        log('[调参] 写 CSV 失败：%s' % e)

    return {'rows': rows, 'best': best}
