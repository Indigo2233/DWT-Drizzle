"""读取 PixInsight / WBPP 的 .xdrz（XDRZ 1.0）几何数据。

为什么需要它
------------
WBPP 的 drizzle 之所以能提高采样率，是因为它不做插值：配准只把
几何变换写进 .xdrz，drizzle 阶段拿**原始标定帧的像素**重新投影。
DWT 现在自己从星点测位移/旋转/尺度，也能做到同样的事；但如果已经
有一份 WBPP 的 .xdrz，直接用它记录的变换可以省掉自测环节，并且
和 PI 的解完全一致。

坐标约定（已用两帧相对几何验证：92.5% 星点在 3px 内匹配，
残差中位 (0.13, -0.16)px，rms 0.26px）
--------------------------------------------------
<AlignmentMatrix> 是行优先的 3x3，坐标按 (x, y) 存，方向是
    src = M @ [x_ref, y_ref, 1]
即"参考坐标 → 源坐标"，和直觉相反。

drizzle 内部表达的是反向的
    ref = k * R(theta) @ src + u
所以取 A = inv(M)，再对 A 的 2x2 部分分解出 (k, theta, u)。
"""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

_NS = '{http://www.pixinsight.com/xdrz}'


def _txt(root, tag: str) -> str:
    e = root.find(_NS + tag)
    if e is None or not e.text:
        return ''
    return e.text.strip()


def parse_xdrz(path: str) -> Optional[Dict]:
    """解析单个 .xdrz；失败返回 None。"""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    src = _txt(root, 'SourceImage')
    mat = _txt(root, 'AlignmentMatrix')
    if not src or not mat:
        return None
    try:
        vals = [float(v) for v in mat.replace('\n', ' ').split(',') if v.strip()]
        if len(vals) != 9:
            return None
        M = np.asarray(vals, np.float64).reshape(3, 3)
        Mi = np.linalg.inv(M)
    except Exception:
        return None
    a, b = float(Mi[0, 0]), float(Mi[1, 0])
    k = math.hypot(a, b)
    if not np.isfinite(k) or k <= 0:
        return None
    return {
        'file': path,
        'src': src.replace('/', '\\'),
        'stem': os.path.splitext(os.path.basename(src))[0],
        'k': k,
        'theta': math.degrees(math.atan2(b, a)),
        'ux': float(Mi[0, 2]),
        'uy': float(Mi[1, 2]),
        'M': M,
        'Mi': Mi,
    }


def load_xdrz_dir(path: str, log=None) -> Dict[str, Dict]:
    """把一个目录（或单个文件）里的 .xdrz 全部读进来，按源文件 stem 建索引。

    同一 stem 有多份时，取修改时间最新的那份（WBPP 重跑会覆盖）。
    """
    files: List[str] = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '**', '*.xdrz'), recursive=True))
    elif os.path.isfile(path) and path.lower().endswith('.xdrz'):
        files = [path]
    if not files:
        raise SystemExit(f'没有找到 .xdrz 文件：{path}')

    out: Dict[str, Dict] = {}
    bad = 0
    for f in files:
        d = parse_xdrz(f)
        if d is None:
            bad += 1
            continue
        old = out.get(d['stem'])
        if old is None or os.path.getmtime(f) >= os.path.getmtime(old['file']):
            out[d['stem']] = d
    if log is not None:
        log(f'[xdrz] 读入 {len(out)} 份几何数据（{path}）'
            + (f'，{bad} 份无法解析已跳过' if bad else ''))
    if not out:
        raise SystemExit(f'.xdrz 全部无法解析：{path}')
    return out


def transforms_for(frames: Iterable[str], xdrz_path: str, log=None) -> Dict:
    """把 .xdrz 里的变换对到具体的帧上。

    返回 dict：
      dx, dy  —— 与 drizzle.measure_shifts 同义（dx = -ux）
      theta   —— 度
      scale   —— 像元尺度 k
      ok      —— bool 数组，False 表示这一帧没有 .xdrz，不参与叠加
    """
    table = load_xdrz_dir(xdrz_path, log=log)
    frames = list(frames)
    n = len(frames)
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    th = np.full(n, np.nan)
    kk = np.full(n, np.nan)
    miss: List[str] = []
    for i, p in enumerate(frames):
        stem = os.path.splitext(os.path.basename(str(p)))[0]
        d = table.get(stem)
        if d is None:
            miss.append(stem)
            continue
        dx[i] = -d['ux']
        dy[i] = -d['uy']
        th[i] = d['theta']
        kk[i] = d['k']
    ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(th) & np.isfinite(kk)
    if log is not None:
        if miss:
            log(f'[xdrz] {len(miss)} 帧没有对应的 .xdrz（例如 {miss[0]}），已排除、不参与叠加')
        if ok.any():
            log(f'[xdrz] 配对 {int(ok.sum())}/{n} 帧  '
                f'旋转 {np.nanmin(th[ok]):+.2f}°~{np.nanmax(th[ok]):+.2f}°  '
                f'像元尺度 {np.nanmin(kk[ok]):.5f}~{np.nanmax(kk[ok]):.5f}')
    if not ok.any():
        raise SystemExit('.xdrz 里的源文件名和素材目录里的帧名对不上，请检查 --xdrz 指向的目录')
    return {'dx': dx, 'dy': dy, 'theta': th, 'scale': kk, 'ok': ok, 'miss': miss}
