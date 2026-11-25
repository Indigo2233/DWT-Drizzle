"""跨运行的配准结果缓存。

同素材反复试参数时，「读帧 + 量星点 FWHM + 测帧间位移」这几步与 A、排异 k、
吞吐权重、drizzle 倍率都无关，占掉了相当大一部分等待时间。这里把它们按
「素材指纹」存起来，第二次起直接复用。

指纹包含：逐帧路径 + 文件大小 + mtime、裁剪、中心、限帧、参考帧、星点数、
配准模式（aligned/xdrz/普通）。任何一项变了就重算——宁可重算，不要用错的位移
去叠加（配准错了成品是糊的，比慢更糟）。

缓存文件是 npz（浮点数组），不参与成品输出，可以直接删。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

CACHE_VER = 1          # 缓存格式版本；改动 payload 结构时 +1，旧缓存自动失效


def _frame_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f'{p.name}|{st.st_size}|{int(st.st_mtime)}'
    except OSError:
        return f'{p.name}|?'


def frames_key(frames: List[Path], extra: Optional[Dict] = None) -> str:
    """素材 + 影响配准的关键参数 → 短指纹"""
    h = hashlib.sha1()
    h.update(f'v{CACHE_VER}'.encode())
    for f in frames:
        h.update(_frame_sig(Path(f)).encode())
        h.update(b'\n')
    if extra:
        for k in sorted(extra):
            h.update(f'{k}={extra[k]};'.encode())
    return h.hexdigest()[:20]


def cache_root(p=None) -> Path:
    """缓存根目录：--cache-dir > DWT_CACHE_DIR > 用户目录/.dwt-cache"""
    d = ''
    try:
        d = str(getattr(p, 'cache_dir', '') or '')
    except Exception:
        d = ''
    d = d or os.environ.get('DWT_CACHE_DIR', '')
    if not d:
        d = str(Path.home() / '.dwt-cache')
    return Path(d).expanduser()


def _path(root: Path, key: str, kind: str) -> Path:
    return root / f'{kind}-{key}.npz'


def load_align(p, key: str) -> Optional[Dict]:
    try:
        import numpy as np
        f = _path(cache_root(p), key, 'align')
        if not f.is_file():
            return None
        with np.load(f, allow_pickle=False) as z:
            if int(z['ver']) != CACHE_VER:
                return None
            out = {k: z[k] for k in ('dx', 'dy', 'theta', 'scale', 'keep')}
            out['n_keep'] = int(z['n_keep'])
            out['n_fill'] = int(z['n_fill'])
        return out
    except Exception:
        return None


def save_align(p, key: str, sh: Dict) -> bool:
    """把 measure_shifts 的结果落盘；失败只记日志、不影响本次叠加"""
    try:
        import numpy as np
        root = cache_root(p)
        root.mkdir(parents=True, exist_ok=True)
        f = _path(root, key, 'align')
        tmp = f.with_name(f.stem + '.tmp.npz')
        np.savez_compressed(
            tmp,
            ver=np.int32(CACHE_VER),
            dx=np.asarray(sh['dx'], np.float64),
            dy=np.asarray(sh['dy'], np.float64),
            theta=np.asarray(sh.get('theta', np.zeros(len(sh['dx']))), np.float64),
            scale=np.asarray(sh.get('scale', np.ones(len(sh['dx']))), np.float64),
            keep=np.asarray(sh.get('keep', np.ones(len(sh['dx']), bool)), bool),
            n_keep=np.int32(int(sh.get('n_keep', 0))),
            n_fill=np.int32(int(sh.get('n_fill', 0))),
        )
        os.replace(tmp, f)
        return True
    except Exception as _e:
        print(f'[缓存] 保存失败：{_e}')
        return False


def describe(p) -> str:
    """给日志用的一句话（缓存目录）"""
    try:
        return str(cache_root(p))
    except Exception:
        return '(未知)'
