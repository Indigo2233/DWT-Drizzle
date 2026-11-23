# -*- coding: utf-8 -*-
"""三角形（asterism）配准：对旋转、尺度、平移全免疫，不依赖初始猜测。

思路：
  1) 两边各取最亮的 M 颗星；
  2) 每三点组成一个三角形，用「按边长排序后的两条比值」当描述子
     （对旋转/平移/尺度/镜像都不变）；
  3) 描述子近似的三角形对 -> 顶点按边长顺序一一对应 -> Umeyama 解相似变换；
  4) 用全部星点验证，投内点最多的那个解。
"""
import numpy as np
from scipy.spatial import cKDTree


def _tri_index(n: int) -> np.ndarray:
    """(i<j<k) 全部三元组，向量化生成（90 颗时 11.7 万行，纯 python 双循环要 0.4s）"""
    ii, jj = np.triu_indices(n, 1)
    keep = jj < n - 1
    ii, jj = ii[keep], jj[keep]
    kk = np.repeat(np.arange(n), 0)          # 占位，下面按行补
    out_i, out_j, out_k = [], [], []
    for k in range(2, n):
        m = jj < k
        out_i.append(ii[m])
        out_j.append(jj[m])
        out_k.append(np.full(int(m.sum()), k, np.int64))
    return np.stack([np.concatenate(out_i), np.concatenate(out_j),
                     np.concatenate(out_k)], 1)


def _tri_descriptors(pts, max_pts=48, tol_min=8.0):
    n = len(pts)
    if n < 6:
        return None, None
    p = np.asarray(pts[:max_pts], np.float64)
    idx = _tri_index(len(p))
    a = np.linalg.norm(p[idx[:, 0]] - p[idx[:, 1]], axis=1)
    b = np.linalg.norm(p[idx[:, 1]] - p[idx[:, 2]], axis=1)
    c = np.linalg.norm(p[idx[:, 2]] - p[idx[:, 0]], axis=1)
    sides = np.stack([a, b, c], 1)
    order = np.argsort(-sides, axis=1)          # 长的在前
    s = np.take_along_axis(sides, order, 1)
    ok = (s[:, 0] > tol_min) & (s[:, 2] > 0.12 * s[:, 0]) & (s[:, 1] > 0.2 * s[:, 0])
    desc = np.stack([s[:, 1] / np.maximum(s[:, 0], 1e-6),
                     s[:, 2] / np.maximum(s[:, 0], 1e-6)], 1)
    # 顶点按边长顺序：对边最长、对边次长、对边最短
    # side0 是 (0,1)，其对角是 2；side1 是 (1,2)，对角是 0；side2 是 (2,0)，对角是 1
    corners = np.stack([idx[:, 2], idx[:, 0], idx[:, 1]], 1)
    v = np.take_along_axis(corners, order, 1)
    return desc[ok], (idx[ok], v[ok], p)


def _umeyama(p, q):
    """求相似变换 q ≈ s·R·p + t。"""
    pc = p.mean(0)
    qc = q.mean(0)
    P = p - pc
    Q = q - qc
    C = Q.T @ P / len(p)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var = (P ** 2).sum() / len(p)
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-12))
    t = qc - s * (R @ pc)
    return s, R, t


def prepare(pts, max_pts=48):
    """预先算好参考帧的三角形描述子，逐帧兜底时复用（省一半时间）"""
    d, meta = _tri_descriptors(pts, max_pts)
    if d is None:
        return None
    return {'desc': d, 'idx': meta[0], 'v': meta[1], 'p': meta[2],
            'tree': cKDTree(d)}


def solve(src_pts, ref_pts, max_pts=48, tol=0.03, inlier_px=2.5, log=None,
          ref_prep=None):
    ds, (di, dv, dp) = _tri_descriptors(src_pts, max_pts)
    if ref_prep is not None:
        dr, (rj, rv, rp), tree = ref_prep['desc'], (ref_prep['idx'], ref_prep['v'],
                                                    ref_prep['p']), ref_prep['tree']
    else:
        dr, (rj, rv, rp) = _tri_descriptors(ref_pts, max_pts)
        tree = cKDTree(dr) if dr is not None and len(dr) else None
    if ds is None or dr is None or len(ds) == 0 or len(dr) == 0:
        return None
    if tree is None:
        return None
    # 只取最近的两个三角形；query_ball_point 在 1.6 万个描述子上会返回
    # 三百多万个候选（tol=0.03），光是逐个 Umeyama 就要好几秒。
    _dd, _jj = tree.query(ds, k=2, distance_upper_bound=tol)
    pairs = []
    for _i in range(len(ds)):
        for _c in range(_jj.shape[1]):
            _j = int(_jj[_i, _c])
            if _j < len(dr) and np.isfinite(_dd[_i, _c]):
                pairs.append((float(_dd[_i, _c]), _i, _j))
    pairs.sort()
    if log:
        log('  [三角] 三角形 %d vs %d，候选对 %d' % (len(ds), len(dr), len(pairs)))
    best = None
    ref_all = ref_pts
    tree2 = cKDTree(ref_all)
    _want = 0.6 * max(1, len(src_pts))
    for _dv, i, j in pairs[:4000]:
        p3 = dp[dv[i]]
        q3 = rp[rv[j]]
        try:
            s, R, t = _umeyama(p3, q3)
        except Exception:
            continue
        if not (0.1 < s < 10.0):
            continue
        q = (s * (R @ src_pts.T)).T + t
        d, idx = tree2.query(q, distance_upper_bound=inlier_px)
        n_in = int(np.isfinite(d).sum())
        if best is None or n_in > best[0]:
            best = (n_in, s, R, t)
            if n_in >= _want:      # 够好了就收工，别把几千个候选试完
                break
    if best is None:
        return None
    n_in, s, R, t = best
    # 同一套光学的两帧之间不可能有 2% 以上的尺度差；历史上出现过 0.4348 这种
    # 伪解（三角形碰巧相似），它一旦被采纳，整帧会被当成放大/缩小 2.3 倍对齐，
    # 叠出来就是星点脱线。宁可不认，交给平移兜底。
    if not (0.98 <= float(s) <= 1.02):
        if log:
            log('  [三角] 尺度 %.4f 不符合同一光路（应在 1±2%%），放弃该解' % float(s))
        return None
    # 用内点重解一次，收紧精度
    q = (s * (R @ src_pts.T)).T + t
    d, idx = cKDTree(ref_all).query(q, distance_upper_bound=inlier_px)
    m = np.isfinite(d)
    if m.sum() >= 6:
        s, R, t = _umeyama(src_pts[m], ref_all[idx[m]])
        q = (s * (R @ src_pts.T)).T + t
        d, idx = cKDTree(ref_all).query(q, distance_upper_bound=inlier_px)
        m = np.isfinite(d)
        n_in = int(m.sum())
    rms = float(np.sqrt((d[m] ** 2).mean())) if m.any() else 99.0
    th = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    if log:
        log('  [三角] 内点 %d 对  尺度 %.4f  旋转 %.2f°  残差 %.2fpx' % (n_in, s, th, rms))
    return {'n_in': n_in, 'scale': s, 'theta': th, 'R': R, 't': t, 'rms': rms}
