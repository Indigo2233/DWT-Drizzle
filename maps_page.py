# -*- coding: utf-8 -*-

"""分区权重场 / 等效帧数查看页。



原版 DWT 叠完之后会给两张诊断图：

  · 分区权重场 R：逐像素「这个位置该不该按清晰度加权」，1 = 细节区（按帧锐度

    分配权重），0 = 朦胧区（退化成纯信噪比加权）。

  · 等效帧数 N_eff：该像素上「等效有多少帧在贡献」，N_eff=(ΣW)²/ΣW²。它低说明

    权重被少数几帧垄断（细节区正常），它接近 0 说明那儿根本没覆盖到。



本页读的是叠加时顺手写出的 `<成品>_R.npy` / `<成品>_neff.npy`，不重算任何东西。

"""

from __future__ import annotations



import base64

import os

import struct

import zlib

import sys

import tkinter as tk

from pathlib import Path

from tkinter import filedialog, ttk



import numpy as np



HERE = Path(__file__).resolve().parent

for _p in (HERE / "pydeps", HERE / "core", HERE):

    _s = str(_p)

    if _s not in sys.path:

        sys.path.insert(0, _s)



import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
FONT = (_FONT_FAMILY, 10)

FONT_S = (_FONT_FAMILY, 9)

FONT_B = (_FONT_FAMILY, 10, "bold")

FG = "#1f2937"

DIM = "#6b7280"

CARD = "#ffffff"



# 两条色带：R 用「墨绿→黄」，N_eff 用「深蓝→青→黄」（都接近 viridis 的观感）

_RAMP_R = [(0.00, (74, 86, 104)), (0.25, (13, 96, 106)), (0.50, (38, 150, 130)),

           (0.75, (140, 194, 96)), (1.00, (250, 232, 130))]

_RAMP_N = [(0.00, (10, 12, 40)), (0.30, (30, 70, 130)), (0.55, (20, 140, 150)),

           (0.80, (120, 200, 120)), (1.00, (250, 245, 170))]





def _lut(ramp, n=256):

    xs = np.array([p[0] for p in ramp], np.float64)

    cs = np.array([p[1] for p in ramp], np.float64)

    t = np.linspace(0, 1, n)

    return np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], -1).astype(np.uint8)





def _to_png(photo_rgb):
    """纯标准库写 PNG（Tk 8.6 的 PhotoImage 认 base64 PNG，不依赖 PIL）"""
    h, w, _ = photo_rgb.shape
    zero = bytes(1)
    raw = b"".join(zero + photo_rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    head = bytes([0x89]) + b"PNG\r\n" + bytes([0x1A, 0x0A])
    return (head
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def _fit(a, max_w, max_h):

    """等比缩小到不超过 max_w × max_h（用步长抽稀，不做插值，保持数值真）"""

    h, w = a.shape

    st = max(1, int(np.ceil(max(h / float(max_h), w / float(max_w)))))

    return a[::st, ::st], st





def _stats(a):

    live = a[np.isfinite(a)]

    if live.size == 0:

        return {}

    q = np.percentile(live, [1, 10, 50, 90, 99])

    return {"min": float(np.nanmin(live)), "max": float(np.nanmax(live)),

            "p1": float(q[0]), "p10": float(q[1]), "p50": float(q[2]),

            "p90": float(q[3]), "p99": float(q[4]),

            "empty": float(np.mean(live <= 0)) if live.max() > 1e-9 else 0.0}





class MapsPage(ttk.Frame):

    def __init__(self, master, app=None):

        super().__init__(master, padding=0)

        self.app = app

        self.R = None

        self.N = None

        self._img_r = None

        self._img_n = None

        self._build()



    # ---------- 界面 ----------

    def _build(self):

        self.columnconfigure(0, weight=1)

        self.rowconfigure(2, weight=1)



        head = ttk.Frame(self)

        head.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))

        head.columnconfigure(1, weight=1)

        ttk.Label(head, text="权重场", font=FONT_B).grid(row=0, column=0, sticky="w")

        ttk.Label(head, text="叠加时顺手存下的 _R.npy / _neff.npy，直接看，不重算",

                  foreground=DIM, font=FONT_S).grid(row=0, column=1, sticky="w", padx=12)



        paths = ttk.Labelframe(self, text="要看的产物", padding=10)

        paths.grid(row=1, column=0, sticky="ew", padx=14, pady=(8, 0))

        paths.columnconfigure(1, weight=1)



        ttk.Label(paths, text="成品 .xisf").grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.v_prod = tk.StringVar()

        ttk.Entry(paths, textvariable=self.v_prod).grid(row=0, column=1, sticky="ew", pady=3)

        ttk.Button(paths, text="浏览…", width=8, command=self._pick).grid(

            row=0, column=2, padx=(8, 0))



        bar = ttk.Frame(paths)

        bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        ttk.Button(bar, text="载入", command=self.load).pack(side="left")

        ttk.Button(bar, text="载入最近一次", command=self.load_last).pack(side="left", padx=8)

        self.v_log = tk.BooleanVar(value=False)

        ttk.Checkbutton(bar, text="N_eff 用对数色标", variable=self.v_log,

                        command=self.render).pack(side="left", padx=(12, 0))

        ttk.Button(bar, text="存 PNG", command=self.save_png).pack(side="left", padx=12)

        self.lab_hint = ttk.Label(bar, text="选成品后自动找同目录的 _R.npy / _neff.npy",

                                  foreground=DIM, font=FONT_S)

        self.lab_hint.pack(side="left", padx=10)



        body = ttk.Frame(self)

        body.grid(row=2, column=0, sticky="nsew", padx=14, pady=(8, 12))

        body.columnconfigure(0, weight=1)

        body.columnconfigure(1, weight=1)

        body.rowconfigure(0, weight=1)



        self.cv_r, self.info_r = self._panel(body, 0, "分区权重场 R",

                                             "1 = 细节区按锐度加权，0 = 朦胧区按信噪比")

        self.cv_n, self.info_n = self._panel(body, 1, "等效帧数 N_eff",

                                             "(ΣW)²/ΣW²：这个像素等效几帧在贡献")

        for cv in (self.cv_r, self.cv_n):

            cv.bind("<Motion>", lambda e, c=cv: self._hover(e, c))



    def _panel(self, parent, col, title, sub):

        f = ttk.Labelframe(parent, text=" " + title + " ", padding=8)

        f.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))

        f.columnconfigure(0, weight=1)

        f.rowconfigure(1, weight=1)

        ttk.Label(f, text=sub, foreground=DIM, font=FONT_S).grid(

            row=0, column=0, sticky="w", pady=(0, 4))

        cv = tk.Canvas(f, bg="#0b1220", highlightthickness=0, height=420)

        cv.grid(row=1, column=0, sticky="nsew")

        info = ttk.Label(f, text="—", foreground="#374151", font=FONT_S, justify="left")

        info.grid(row=2, column=0, sticky="w", pady=(6, 0))

        return cv, info



    # ---------- 载入 ----------

    def _pick(self):

        p = filedialog.askopenfilename(title="选叠加成品",

                                       filetypes=[("XISF", "*.xisf"), ("全部", "*.*")])

        if p:

            self.v_prod.set(p)

            self.load()



    def _companions(self, prod: str):

        p = Path(prod)

        stem = p.stem

        if stem.startswith("PWS_"):

            stem = stem[4:]

        r = p.with_name(p.name.replace(".xisf", "_R.npy"))

        n = p.with_name(p.name.replace(".xisf", "_neff.npy"))

        return r, n



    def load_last(self):

        if self.app is not None:

            out = getattr(self.app, "_last_out", None)

            tag = getattr(self.app, "_last_stem", None)

            if out and tag:

                cand = Path(out) / (tag + ".xisf")

                if cand.exists():

                    self.v_prod.set(str(cand))

                    self.load()

                    return

        self.lab_hint.configure(text="还没有跑过叠加；先在上面那一页跑一次")



    def load(self):

        prod = self.v_prod.get().strip()

        if not prod:

            self.lab_hint.configure(text="先选一个成品 .xisf")

            return

        r, n = self._companions(prod)

        ok = []

        try:

            self.R = np.load(str(r)).astype(np.float32) if r.exists() else None

            ok.append("R" if self.R is not None else "R ✗")

        except Exception:

            self.R = None

        try:

            self.N = np.load(str(n)).astype(np.float32) if n.exists() else None

            ok.append("N_eff" if self.N is not None else "N_eff ✗")

        except Exception:

            self.N = None

        self.lab_hint.configure(text="读入 " + " / ".join(ok) + "：" +

                                (str(r) if self.R is not None else str(n)))

        self.render()



    # ---------- 画 ----------

    def render(self):

        if self.R is not None:

            self._draw(self.cv_r, self.info_r, self.R, _RAMP_R, "R", log=False, gamma=0.45)

        else:

            self._blank(self.cv_r, "没有 _R.npy（这版可能没开 drizzle）")

        if self.N is not None:

            self._draw(self.cv_n, self.info_n, self.N, _RAMP_N, "N_eff",

                       log=bool(self.v_log.get()))

        else:

            self._blank(self.cv_n, "没有 _neff.npy")



    def _blank(self, cv, msg):

        cv.delete("all")

        cv.create_text(cv.winfo_width() // 2 or 200, 120, text=msg,

                       fill="#64748b", font=FONT)



    def _draw(self, cv, info, a, ramp, name, log=False, gamma=1.0):

        cv.update_idletasks()

        W = max(320, cv.winfo_width() - 4)

        H = max(200, cv.winfo_height() - 4)

        sub, st = _fit(a, W, H - 44)

        v = sub.astype(np.float64)

        if log:

            v = np.log10(np.maximum(v, 1e-3))

        lo, hi = np.percentile(v[np.isfinite(v)], [0.5, 99.5])

        if not np.isfinite(lo) or hi - lo < 1e-12:

            hi = lo + 1.0

        d = np.clip((v - lo) / (hi - lo), 0, 1)
        if gamma != 1.0:
            d = np.power(d, gamma)

        rgb = _lut(ramp)[(d * 255).astype(np.int32)]

        rgb[~np.isfinite(v)] = (0, 0, 0)

        # 每个画布自己持有照片对象，否则第二个画布会覆盖第一个、旧图被回收成黑块
        cv._img = tk.PhotoImage(data=base64.b64encode(_to_png(rgb)))

        cv.delete("all")

        cv.create_image(W // 2 + 2, H // 2 + 2, anchor="center", image=cv._img)

        cv.configure(scrollregion=(0, 0, rgb.shape[1], rgb.shape[0]))
        # 色标条（48 段渐变 + 两端数值），放在图下方
        bx0, bw2, bh = 12, min(260, W - 24), 10
        by = H // 2 + rgb.shape[0] // 2 + 8
        lut = _lut(ramp)
        for i in range(48):
            c = tuple(int(v) for v in lut[int(i * 255 / 47)])
            cv.create_rectangle(bx0 + i * bw2 / 48, by,
                                bx0 + (i + 1) * bw2 / 48 + 1, by + bh,
                                outline="", fill="#%02x%02x%02x" % c)
        cv.create_text(bx0, by + bh + 8, text=("%.4g" % lo), anchor="w",
                       fill=DIM, font=FONT_S)
        cv.create_text(bx0 + bw2, by + bh + 8, text=("%.4g" % hi), anchor="e",
                       fill=DIM, font=FONT_S)

        s = _stats(a)
        if name == "R":
            txt = ("R（色标 %.3g~%.3g，低值已提亮）：均值 %.3f   中位 %.3f   "
                   ">0.5 占 %.2f%%   <0.1 占 %.1f%%"
                   "\n缩略 %d×%d（1/%d）"
                   % (lo, hi, float(np.nanmean(a)), s.get("p50", 0),
                      100 * float(np.mean(a > 0.5)), 100 * float(np.mean(a < 0.1)),
                      rgb.shape[1], rgb.shape[0], st))
        else:
            txt = ("N_eff（色标 %.4g~%.4g%s）：中位 %.1f   P10 %.1f / P90 %.1f   "
                   "空 %.2f%%"
                   "\n缩略 %d×%d（1/%d）"
                   % (lo, hi, "，对数" if log else "",
                      s.get("p50", 0), s.get("p10", 0), s.get("p90", 0),
                      100 * s.get("empty", 0), rgb.shape[1], rgb.shape[0], st))
        info.configure(text=txt)
        cv._map = a



    def _hover(self, ev, cv):

        a = getattr(cv, "_map", None)

        if a is None:

            return

        h, w = a.shape

        cv.update_idletasks()

        W = max(320, cv.winfo_width() - 4)

        H = max(200, cv.winfo_height() - 4)

        sub, st = _fit(a, W, H - 44)

        y = int(ev.y / max(1, cv.winfo_height()) * sub.shape[0])

        x = int(ev.x / max(1, cv.winfo_width()) * sub.shape[1])

        if 0 <= y < sub.shape[0] and 0 <= x < sub.shape[1]:

            name = "R" if cv is self.cv_r else "N_eff"

            self.lab_hint.configure(

                text="%s @ (%d, %d) = %.3f" % (name, x * st, y * st, float(sub[y, x])))



    def save_png(self):

        prod = self.v_prod.get().strip()

        if not prod or self.R is None:

            return

        out = str(Path(prod).with_name(Path(prod).stem + "_maps.png"))

        try:

            from PIL import Image       # 只有本机 DSH 运行时带 PIL

        except Exception:

            self.lab_hint.configure(text="这台机器的 python 没有 PIL，存不了 PNG")

            return

        tiles = []

        for a, ramp, log in ((self.R, _RAMP_R, False), (self.N, _RAMP_N, True)):

            if a is None:

                continue

            v = np.log10(np.maximum(a.astype(np.float64), 1e-3)) if log else a.astype(np.float64)

            lo, hi = np.percentile(v[np.isfinite(v)], [0.5, 99.5])

            d = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)

            tiles.append(_lut(ramp)[(d * 255).astype(np.int32)])

        if not tiles:

            return

        h = min(t.shape[0] for t in tiles)

        img = np.concatenate([t[:h] for t in tiles], axis=1)

        Image.fromarray(img).save(out)

        self.lab_hint.configure(text="已写出 " + out)

