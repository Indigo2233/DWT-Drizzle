# -*- coding: utf-8 -*-
"""并排看两帧（不同日期）的星场，判断是否同一取景。"""
import numpy as np
from pathlib import Path
import core.pws as pws
import xisf_io


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H']
    picks = []
    for pat in ('2026-10-18', '2026-10-30', '2026-11-01', '2026-11-05', '2026-11-10'):
        for f in h:
            if pat in str(f):
                picks.append(f)
                break
    fs = pws.prepare_frames(picks, 'look', 0, None)
    tiles = []
    for i in range(len(picks)):
        a = np.asarray(fs.frame(i), np.float64)
        lo = float(np.median(a)); hi = float(np.percentile(a, 99.9))
        d = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1e-9) * 6, 0, None)) / np.arcsinh(6.0)
        im = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        tiles.append(im[::4, ::4])
        print(i, Path(str(picks[i])).name[:40], a.shape, 'sky %.3g' % lo)
    np.save(r'C:\Windows\Temp\frames_look.npy', np.stack(tiles))


if __name__ == '__main__':
    main()
