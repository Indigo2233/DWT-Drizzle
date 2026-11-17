# -*- coding: utf-8 -*-
"""系统误差诊断：把帧分两半各叠一版，用 S=(A+B)/2 与 D=(A-B)/2 分离

    var(S) = var(真信号+系统误差) + σn²/2
    var(D) = σn²/2
  ⇒ var(真+系统) = var(S) − var(D)

逐尺度输出「系统误差占总方差的比例」，以及「若系统误差被完全压掉，SNR 还能涨多少倍」。
两半必须落在同一画布（用 --frame 固定），否则不可比。
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, 'core'), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import xisf_io

PYEXE = os.path.join(ROOT, 'py', 'python.exe')
MAIN = os.path.join(ROOT, 'main.py')


def collect(photos, pick='O'):
    import pws
    files, _ = pws.collect_frames(photos)
    groups = pws.group_frames_by_filter(files)
    out = []
    for k, v in groups.items():
        if pick and k != pick:
            continue
        out.extend(v)
    return sorted(out)


def split_links(files, root):
    a_dir = os.path.join(root, 'A')
    b_dir = os.path.join(root, 'B')
    for d in (a_dir, b_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for i, f in enumerate(files):
        d = a_dir if i % 2 == 0 else b_dir
        dst = os.path.join(d, os.path.basename(f))
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    return a_dir, b_dir


def run_stack(photos, out, tag, frame, extra):
    os.makedirs(out, exist_ok=True)
    cmd = [PYEXE, '-B', MAIN, '--cli', '--photos', photos, '--out', out,
           '--tag', tag, '--pick', 'O', '--frame', frame] + extra
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([os.path.join(ROOT, 'pydeps'),
                                         os.path.join(ROOT, 'core'), ROOT])
    env['PYTHONIOENCODING'] = 'utf-8'
    print('运行', tag, '…', flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = [ln for ln in (r.stdout or '').splitlines() if '[输出] 成品' in ln or '[叠加] 参与' in ln]
    print('  ' + ' | '.join(tail) if tail else '  (无输出)', flush=True)
    if r.returncode != 0:
        print((r.stdout or '')[-1500:], (r.stderr or '')[-800:], sep='\n')
    return r.returncode


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def analyze(pa, pb):
    A, B = load(pa), load(pb)
    S = 0.5 * (A + B)
    D = 0.5 * (A - B)
    print('\n画幅 %dx%d' % (A.shape[1], A.shape[0]))
    print('%-8s %12s %12s %12s %8s %10s' %
          ('尺度', 'var(S)', 'var(D)=噪声', '系统误差', '占比', '潜在增益'))
    for sig in (0, 2, 4, 8, 16, 32, 64):
        if sig > 0:
            s = ndimage.gaussian_filter(S, sig)
            d = ndimage.gaussian_filter(D, sig)
        else:
            s, d = S, D
        vs = float(np.var(s))
        vd = float(np.var(d))
        sysv = max(vs - vd, 0.0)
        frac = sysv / vs if vs > 0 else 0.0
        gain = np.sqrt(vs / vd) if vd > 0 else float('nan')
        print('%-8s %12.4g %12.4g %12.4g %7.1f%% %9.2f×'
              % ('原始' if sig == 0 else 'σ=%dpx' % sig, vs, vd, sysv, 100 * frac, gain))
    return S, D


def main():
    photos = sys.argv[1] if len(sys.argv) > 1 else r'G:\HFG1-3\calibrated'
    frame = sys.argv[2] if len(sys.argv) > 2 else \
        r'G:\HFG1-3\master2\master\masterLight_BIN-1_2758x2204_EXPOSURE-1200.00s_FILTER-O_mono_drizzle_2x_autocrop.xisf'
    limit = sys.argv[3] if len(sys.argv) > 3 else '40'
    bg = os.environ.get('SYS_BG', '1')
    tag = os.environ.get('SYS_TAG', '')
    extra = ['--limit', limit, '--drizzle', '--drizzle-scale', '2',
             '--drizzle-pix', '1.0', '--bgdeg', bg]
    work = r'G:\HFG1\lab\_sysdiag' + tag
    os.makedirs(work, exist_ok=True)
    files = collect(photos, 'O')[:int(limit)]
    print('帧数 %d' % len(files))
    a_dir, b_dir = split_links(files, work)
    pa = os.path.join(work, 'A_out', 'PWS_hA.xisf')
    pb = os.path.join(work, 'B_out', 'PWS_hB.xisf')
    run_stack(a_dir, os.path.dirname(pa), 'hA', frame, extra)
    run_stack(b_dir, os.path.dirname(pb), 'hB', frame, extra)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print('两半之一没有产出，退出')
        return 1
    analyze(pa, pb)
    return 0


if __name__ == '__main__':
    sys.exit(main())
