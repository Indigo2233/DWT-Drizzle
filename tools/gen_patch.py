# -*- coding: utf-8 -*-
"""用上游原始文件 + 我们的树，重新生成 drizzle.patch（2.7.27-C）。

上游只有 3 个文件与我们重叠：DWT_DetailCore.py、core/pws.py、core/pws_params.py；
其余都是我们新增的文件（整文件新增）。
"""
import difflib
import pathlib
import sys

UP = pathlib.Path('G:/M81/_work/up')
TREE = pathlib.Path('G:/DWT/dist/DWT-Drizzle-C')
OUT = pathlib.Path('G:/DWT/github/DWT-Drizzle/drizzle.patch')

MODIFIED = ['DWT_DetailCore.py', 'core/pws.py', 'core/pws_params.py']

# 新增文件：我们自己的模块（不含 py/ pydeps/ gpudeps/ tests/ tools/ docs/ 与包内数据）
ADDED = [
    'core/cfa.py', 'core/drizzle.py', 'core/bggrad.py', 'core/matchref.py',
    'core/trireg.py', 'core/cropcover.py', 'core/tune.py', 'core/gpu_dz.py',
    'core/gpu_band.py', 'core/xdrz.py', 'core/cfadz.py', 'core/dbg_band.py',
    'core/gpu_wgpu.py', 'core/gpu_wgpu_pws.py', 'core/gpu_wgpu_cfa_bucket.py',
    'core/gpu_wgpu_cfa_scatter.py', 'core/gpu_wgpu_cfa_fatom.py',
    'gui.py', 'eval_page.py', 'maps_page.py', 'monitor.py', 'help_page.py',
    'verify_gpu.py', 'fix_bounds.py',
]


def _lines(p: pathlib.Path):
    t = p.read_text(encoding='utf-8', errors='replace')
    return [ln + '\n' for ln in t.split('\n')]


def diff_pair(rel: str, a: pathlib.Path, b: pathlib.Path) -> str:
    la = _lines(a) if a else []
    lb = _lines(b)
    na = str(a) if a else '/dev/null'
    head = '--- %s\n+++ b/%s\n' % ('a/' + rel if a else '/dev/null', rel)
    body = ''.join(difflib.unified_diff(la, lb, fromfile='', tofile='', n=3))
    return head + body


def main():
    chunks = []
    for rel in MODIFIED:
        src = UP / rel
        dst = TREE / rel
        if not src.exists():
            print('missing upstream', rel, file=sys.stderr)
            return 2
        chunks.append(diff_pair(rel, src, dst))
    for rel in ADDED:
        dst = TREE / rel
        if dst.exists():
            chunks.append(diff_pair(rel, None, dst))
        else:
            print('skip missing', rel, file=sys.stderr)
    text = ''.join(chunks)
    OUT.write_text(text, encoding='utf-8', newline='\n')
    print('wrote %s  %d bytes  %d files' % (OUT, len(text.encode('utf-8')), len(chunks)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
