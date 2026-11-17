# -*- coding: utf-8 -*-
"""叠加评价页：对一张 drizzle 成品做「质量体检」。

版式沿用主界面的风格（同样的字体、浅灰底、右侧一行灰色说明），左边挑文件、
右边出数字与曲线，全部本地计算，不联网、不调模型。

指标分四块：
  1. 基本   —— 尺寸、天空电平、背景 σ
  2. 星点   —— 数量、FWHM 中位与分位、分半径的 FWHM 与轴比曲线
  3. 权重   —— N_eff（等效帧数）分布、细节区占比；有 R 场时给 R 场读数
  4. 帧权重 —— 有 _weights.txt 时给 C / S 范围、最锐与最糊的帧号
"""
from __future__ import annotations

import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

import xisf_io

import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
FONT = (_FONT_FAMILY, 10)
FONT_S = (_FONT_FAMILY, 9)
FONT_B = (_FONT_FAMILY, 11, "bold")
BG = "#f3f4f6"
FG = "#1f2937"
MUTED = "#6b7280"


# ---------------------------------------------------------------- 计算

def detect_stars(img: np.ndarray, thr_pct: float = 99.6):
    """峰值检测 + 二阶矩，返回 (x, y, a, b, pa) 五列。不吃背景，只看亮峰。"""
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(img, 5)
    pts = np.argwhere((img == mx) & (img > np.percentile(img, thr_pct)))
    out = []
    h, w = img.shape
    for y, x in pts:
        if y < 8 or x < 8 or y > h - 9 or x > w - 9:
            continue
        t = img[y - 7:y + 8, x - 7:x + 8].astype(np.float64)
        t = t - np.median(t)
        t[t < 0] = 0
        s = t.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[-7:8, -7:8]
        mx_, my_ = (t * xx).sum() / s, (t * yy).sum() / s
        cxx = (t * (xx - mx_) ** 2).sum() / s
        cyy = (t * (yy - my_) ** 2).sum() / s
        cxy = float((t * (xx - mx_) * (yy - my_)).sum() / s)
        ev, _ = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
        if ev[1] <= 0:
            continue
        out.append((x, y, math.sqrt(max(ev[1], 1e-9)), math.sqrt(max(ev[0], 1e-9))))
    return np.array(out) if out else np.zeros((0, 4))


def fwhm_from_moments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """二阶矩 → FWHM（高斯近似：FWHM = 2√(2ln2)·σ = 2.3548σ）"""
    return 2.35482 * np.sqrt(a * a + b * b) / math.sqrt(2.0)


def evaluate(xisf_path: str, neff_path: str = "", r_path: str = "",
             w_path: str = "", log=print) -> dict:
    res: dict = {"path": xisf_path}
    img = np.asarray(xisf_io.read_xisf(xisf_path)[0].data, np.float64)
    ok = img > 0
    med = float(np.median(img[ok])) if ok.any() else 0.0
    sig = float(1.4826 * np.median(np.abs(img[ok] - med))) if ok.any() else 0.0
    res.update(shape=img.shape, sky=med, sigma=sig,
               cover=float(ok.mean()) if ok.any() else 0.0)
    log(f'尺寸 {img.shape[1]}×{img.shape[0]}  天空 {med:.4g}  背景σ {sig:.3g}')

    st = detect_stars(img.astype(np.float32))
    if len(st) >= 10:
        a, b = st[:, 2], st[:, 3]
        fw = fwhm_from_moments(a, b)
        cy, cx = img.shape[0] / 2.0, img.shape[1] / 2.0
        r = np.hypot(st[:, 0] - cx, st[:, 1] - cy)
        rmax = max(cy, cx)
        rings = []
        for lo, hi in ((0.0, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 0.95), (0.95, 1.6)):
            k = (r >= lo * rmax) & (r < hi * rmax)
            if k.sum() >= 5:
                rings.append({"lo": lo, "hi": hi, "n": int(k.sum()),
                              "fwhm": float(np.median(fw[k])),
                              "ba": float(np.median(b[k] / a[k]))})
        res.update(n_star=int(len(st)), fwhm=float(np.median(fw)),
                   fwhm_p10=float(np.percentile(fw, 10)),
                   fwhm_p90=float(np.percentile(fw, 90)),
                   rings=rings)
        log(f'星点 {len(st)}   FWHM 中位 {np.median(fw):.2f}px '
            f'(P10 {np.percentile(fw, 10):.2f} / P90 {np.percentile(fw, 90):.2f})')
    else:
        res.update(n_star=int(len(st)), rings=[])
        log('星点太少，跳过形状统计')

    if neff_path and Path(neff_path).exists():
        ne = np.load(neff_path)
        live = ne[ne > 0]
        res["neff"] = {"med": float(np.median(live)) if live.size else 0.0,
                       "p10": float(np.percentile(live, 10)) if live.size else 0.0,
                       "p90": float(np.percentile(live, 90)) if live.size else 0.0,
                       "zero": float((ne <= 0).mean())}
        log(f'N_eff 中位 {res["neff"]["med"]:.1f}  (P10 {res["neff"]["p10"]:.1f} / '
            f'P90 {res["neff"]["p90"]:.1f})')
        if r_path and Path(r_path).exists():
            rf = np.load(r_path)
            res["R"] = {"mean": float(rf.mean()),
                        "gt50": float((rf > 0.5).mean()),
                        "lt10": float((rf < 0.1).mean())}
            if live.size == rf.size:
                det = rf.ravel() > 0.5
                if det.any():
                    res["R"]["detail_neff"] = float(np.median(ne.ravel()[det]))
            log(f'R 场 均值 {res["R"]["mean"]:.3f}  R>0.5 占 {res["R"]["gt50"]:.2%}')

    if w_path and Path(w_path).exists():
        rows = []
        for ln in Path(w_path).read_text("utf-8", errors="ignore").splitlines():
            p = ln.split()
            if len(p) >= 4:
                try:
                    # 列：idx fwhm_px C_sharp sigma S_snr（旧文件可能只有四列）
                    s_snr = float(p[4]) if len(p) >= 5 else 1.0
                    rows.append((int(p[0]), float(p[1]), float(p[2]), s_snr))
                except ValueError:
                    continue
        if rows:
            arr = np.array(rows, np.float64)
            res["weights"] = {"n": len(rows),
                              "C_min": float(arr[:, 2].min()), "C_max": float(arr[:, 2].max()),
                              "S_min": float(arr[:, 3].min()), "S_max": float(arr[:, 3].max()),
                              "sharp": int(arr[arr[:, 1].argmin(), 0]),
                              "soft": int(arr[arr[:, 1].argmax(), 0]),
                              "fwhm_min": float(arr[:, 1].min()),
                              "fwhm_max": float(arr[:, 1].max())}
            log(f'权重表 {len(rows)} 帧  C {res["weights"]["C_min"]:.2f}~'
                f'{res["weights"]["C_max"]:.2f}')
    return res


def verdict(res: dict) -> list:
    """把数字翻译成两句人话。"""
    out = []
    rings = res.get("rings") or []
    if len(rings) >= 2:
        f0, f1 = rings[0]["fwhm"], rings[-1]["fwhm"]
        gain = (f1 - f0) / max(f0, 1e-9)
        if gain > 0.12:
            out.append(f'边缘星点比中心差 {gain * 100:.0f}%（{f0:.2f}→{f1:.2f}px）'
                       '——先怀疑配准或 PSF 差异，不是 drizzle 的问题')
        else:
            out.append(f'全画幅星点均匀（中心 {f0:.2f} → 边缘 {f1:.2f}px，差 {gain * 100:.0f}%）')
        ba1 = rings[-1]["ba"]
        if ba1 < 0.85:
            out.append(f'边缘轴比 {ba1:.2f}，星点被拉长——多半是配准残差随半径增大')
        elif ba1 > 0.95:
            out.append(f'边缘轴比 {ba1:.2f}，形状保持得很好')
    ne = res.get("neff")
    if ne and ne.get("p10", 0) < 0.35 * max(ne.get("med", 1), 1e-9):
        out.append('N_eff 的 P10 远低于中位：有一片区域只有少数帧覆盖，'
                   '那里的噪声会明显更高（画面四角常见）')
    w = res.get("weights")
    if w and w["C_max"] > 2.5:
        out.append(f'最锐的帧权重到 {w["C_max"]:.1f}× 等权——细节区基本由少数几帧决定，'
                   '帧间 PSF 差异会被放大')
    return out


# ---------------------------------------------------------------- 界面

class EvalPage(ttk.Frame):
    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.configure(style="TFrame")
        self.res: dict | None = None
        self._busy = False
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        p = ttk.Labelframe(self, text=" 要评价的成品 ", padding=10)
        p.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 8))
        p.columnconfigure(1, weight=1)
        self.v_xisf = tk.StringVar()
        self.v_neff = tk.StringVar()
        self.v_r = tk.StringVar()
        self.v_w = tk.StringVar()

        ttk.Label(p, text="成品 .xisf").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(p, textvariable=self.v_xisf).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(p, text="浏览…", width=8, command=self._pick_xisf).grid(row=0, column=2, padx=(8, 0))

        for i, (lab, var, key) in enumerate((
                ("N_eff .npy", self.v_neff, "neff"),
                ("R 场 .npy", self.v_r, "R"),
                ("权重表 .txt", self.v_w, "weights")), start=1):
            ttk.Label(p, text=lab).grid(row=i, column=0, sticky="w", padx=(0, 8))
            ttk.Entry(p, textvariable=var).grid(row=i, column=1, sticky="ew", pady=3)
            ttk.Button(p, text="浏览…", width=8,
                       command=lambda v=var: self._pick_aux(v)).grid(row=i, column=2, padx=(8, 0))

        ttk.Label(p, text="选好成品后，同目录的 _R.npy / _neff.npy / _weights.txt 会自动填上"
                          "（它们和成品是同一次运行的产物）",
                  foreground=MUTED, font=FONT_S).grid(row=4, column=0, columnspan=3,
                                                     sticky="w", pady=(4, 0))

        bar = tk.Frame(self, bg=BG)
        bar.grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 6))
        self.btn = ttk.Button(bar, text="开始评价", width=12, style="Run.TButton",
                              command=self.run_eval)
        self.btn.pack(side="left")
        self.lab = tk.Label(bar, text="读成品 → 数星点 → 看 N_eff / 权重，全部本地算，几十秒",
                            bg=BG, fg=MUTED, font=FONT_S)
        self.lab.pack(side="left", padx=12)

        # 报告 + 曲线
        body = tk.Frame(self, bg=BG)
        body.grid(row=2, column=0, sticky="nsew", padx=2)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        f1 = ttk.Labelframe(body, text=" 读数 ", padding=6)
        f1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        f1.rowconfigure(0, weight=1)
        f1.columnconfigure(0, weight=1)
        self.txt = tk.Text(f1, wrap="word", font=FONT, bg="#ffffff", fg=FG,
                           relief="flat", padx=10, pady=8, height=18, width=52)
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(f1, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.tag_configure("h", font=FONT_B, foreground="#0f766e")
        self.txt.tag_configure("k", foreground=MUTED)
        self.txt.tag_configure("warn", foreground="#b45309")

        f2 = ttk.Labelframe(body, text=" 分半径曲线 ", padding=6)
        f2.grid(row=0, column=1, sticky="nsew")
        f2.rowconfigure(0, weight=1)
        f2.columnconfigure(0, weight=1)
        self.cv = tk.Canvas(f2, bg="#ffffff", highlightthickness=0, height=250, width=330)
        self.cv.grid(row=0, column=0, sticky="nsew")
        self.cv.bind("<Configure>", lambda e: self._draw())

    # ---------- 选文件 ----------
    def _pick_xisf(self):
        p = filedialog.askopenfilename(title="选择叠加成品",
                                       filetypes=[("XISF", "*.xisf"), ("所有文件", "*.*")])
        if not p:
            return
        self.v_xisf.set(p)
        base = Path(p)
        stem = base.name[:-5] if base.name.lower().endswith(".xisf") else base.stem
        for var, suf in ((self.v_neff, "_neff.npy"), (self.v_r, "_R.npy"), (self.v_w, "_weights.txt")):
            cand = base.with_name(stem + suf)
            if cand.exists():
                var.set(str(cand))
            else:
                var.set("")

    def _pick_aux(self, var):
        p = filedialog.askopenfilename(title="选择附属文件",
                                       filetypes=[("数据", "*.npy *.txt"), ("所有文件", "*.*")])
        if p:
            var.set(p)

    # ---------- 评价 ----------
    def run_eval(self):
        if self._busy:
            return
        path = self.v_xisf.get().strip()
        if not path or not Path(path).exists():
            self._say("先选一个成品 .xisf", warn=True)
            return
        self._busy = True
        self.btn.configure(state="disabled")
        self._say("评价中…")
        args = (path, self.v_neff.get().strip(), self.v_r.get().strip(), self.v_w.get().strip())

        def work():
            try:
                lines = []
                res = evaluate(*args, log=lambda m: lines.append(m))
                self.after(0, lambda: self._done(res, lines, None))
            except Exception as e:  # 界面上要能看见原因，不要静默
                self.after(0, lambda: self._done(None, [], e))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, res, lines, err):
        self._busy = False
        self.btn.configure(state="normal")
        if err is not None:
            self._say(f"评价失败：{err}", warn=True)
            self.res = None
            return
        self.res = res
        self._fill_text(res, lines)
        self._draw()
        self._say("完成")

    def _say(self, msg, warn=False):
        self.lab.configure(text=msg, fg="#b45309" if warn else MUTED)

    def _fill_text(self, res, lines):
        t = self.txt
        t.delete("1.0", "end")
        h, w = res["shape"]
        t.insert("end", "叠加评价\n", "h")
        t.insert("end", f"{Path(res['path']).name}\n", "k")
        t.insert("end", "\n")
        t.insert("end", "画幅        ", "k")
        t.insert("end", f"{w} × {h}    有效覆盖 {res['cover']:.1%}\n")
        t.insert("end", "天空 / 噪声 ", "k")
        t.insert("end", f"{res['sky']:.4g}  /  σ {res['sigma']:.3g}\n")
        t.insert("end", "\n")

        if res.get("n_star", 0) >= 10:
            t.insert("end", "星点\n", "h")
            t.insert("end", "  数量      ", "k")
            t.insert("end", f"{res['n_star']}\n")
            t.insert("end", "  FWHM      ", "k")
            t.insert("end", f"中位 {res['fwhm']:.2f}px   "
                            f"P10 {res['fwhm_p10']:.2f} / P90 {res['fwhm_p90']:.2f}\n")
            for r in res.get("rings", []):
                t.insert("end", f"  r {r['lo'] * 100:>3.0f}–{r['hi'] * 100:>3.0f}%   ", "k")
                t.insert("end", f"FWHM {r['fwhm']:.2f}px   轴比 {r['ba']:.2f}   n={r['n']}\n")
            t.insert("end", "\n")

        ne = res.get("neff")
        if ne:
            t.insert("end", "权重 / 等效帧数\n", "h")
            t.insert("end", "  N_eff     ", "k")
            t.insert("end", f"中位 {ne['med']:.1f}   P10 {ne['p10']:.1f} / P90 {ne['p90']:.1f}\n")
            t.insert("end", "  空像素    ", "k")
            t.insert("end", f"{ne['zero']:.2%}\n")
            rr = res.get("R")
            if rr:
                t.insert("end", "  R 场      ", "k")
                t.insert("end", f"均值 {rr['mean']:.3f}   >0.5 占 {rr['gt50']:.2%}   "
                                f"<0.1 占 {rr['lt10']:.2%}\n")
                if "detail_neff" in rr:
                    t.insert("end", "  细节区 N_eff ", "k")
                    t.insert("end", f"{rr['detail_neff']:.1f}\n")
            t.insert("end", "\n")

        wt = res.get("weights")
        if wt:
            t.insert("end", "帧权重\n", "h")
            t.insert("end", "  帧数      ", "k")
            t.insert("end", f"{wt['n']}（单帧 FWHM {wt['fwhm_min']:.2f}~{wt['fwhm_max']:.2f}px）\n")
            t.insert("end", "  C / S     ", "k")
            t.insert("end", f"C {wt['C_min']:.2f}~{wt['C_max']:.2f}    "
                            f"S {wt['S_min']:.2f}~{wt['S_max']:.2f}\n")
            t.insert("end", "  最锐 / 最糊 ", "k")
            t.insert("end", f"第 {wt['sharp']} 帧 / 第 {wt['soft']} 帧\n")
            t.insert("end", "\n")

        for line in verdict(res):
            t.insert("end", "结论  ", "h")
            t.insert("end", line + "\n")

    # ---------- 曲线 ----------
    def _draw(self):
        cv = self.cv
        cv.delete("all")
        res = self.res
        W = max(cv.winfo_width(), 200)
        H = max(cv.winfo_height(), 200)
        if not res or not res.get("rings"):
            cv.create_text(W / 2, H / 2, text="跑一次评价，这里会出现 FWHM 与轴比随半径的变化",
                           fill=MUTED, font=FONT_S)
            return
        rings = res["rings"]
        xs = [(r["lo"] + r["hi"]) / 2 * 100 for r in rings]
        fw = [r["fwhm"] for r in rings]
        ba = [r["ba"] for r in rings]

        pad_l, pad_r, pad_t = 58, 16, 30
        gap = 80
        hh = max(60.0, (H - pad_t - gap - 40) / 2.0)

        def panel(y0, vals, title, color, ref):
            cv.create_text(pad_l, y0 - 12, text=title, anchor="w", fill=FG, font=FONT_S)
            lo, hi = min(vals + [ref]), max(vals + [ref])
            m = (hi - lo) * 0.25 + 1e-6
            lo, hi = lo - m, hi + m
            x0, x1 = pad_l, W - pad_r
            y1 = y0 + hh
            cv.create_line(x0, y1, x1, y1, fill="#cbd5e1")
            cv.create_line(x0, y0, x0, y1, fill="#cbd5e1")
            for frac in (0.0, 0.5, 1.0):
                yy = y1 - frac * hh
                cv.create_line(x0 - 3, yy, x0, yy, fill="#cbd5e1")
                cv.create_text(x0 - 6, yy, text=f'{lo + frac * (hi - lo):.2f}',
                               anchor="e", fill=MUTED, font=FONT_S)
                cv.create_line(x0, yy, x1, yy, fill="#eef2f7")
            pts = []
            for xv, v in zip(xs, vals):
                px = x0 + (xv - xs[0]) / max(xs[-1] - xs[0], 1e-9) * (x1 - x0)
                py = y1 - (v - lo) / (hi - lo) * hh
                pts += [px, py]
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=2, smooth=True)
            for i in range(0, len(pts), 2):
                cv.create_oval(pts[i] - 2.5, pts[i + 1] - 2.5, pts[i] + 2.5, pts[i + 1] + 2.5,
                               fill=color, outline="")
            for xv in (xs[0], xs[len(xs) // 2], xs[-1]):
                px = x0 + (xv - xs[0]) / max(xs[-1] - xs[0], 1e-9) * (x1 - x0)
                cv.create_text(px, y1 + 12, text=f"{xv:.0f}%", fill=MUTED, font=FONT_S)

        panel(pad_t, fw, "FWHM (px)   vs   到画幅中心的半径", "#0f766e", res["fwhm"])
        panel(pad_t + hh + gap, ba, "轴比 b/a   vs   半径", "#b45309", 1.0)
