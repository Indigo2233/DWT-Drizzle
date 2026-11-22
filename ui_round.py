# -*- coding: utf-8 -*-
"""分支 D 的圆角控件：白底圆角卡片 + 圆角按钮（纯 tkinter，无第三方依赖）。"""
from __future__ import annotations

import tkinter as tk

BG = '#f5f5f7'
CARD = '#ffffff'
BORDER = '#e3e3e8'
FG = '#1d1d1f'
FG2 = '#6e6e73'
ACCENT = '#0071e3'
ACCENT_D = '#0059b3'


def round_rect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """在 Canvas 上画一个圆角矩形（用平滑多边形近似，兼容所有 Tk 版本）。"""
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    pts = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return cv.create_polygon(pts, smooth=True, splinesteps=24, **kw)


class RoundCard(tk.Frame):
    """白底圆角卡片。内容放进 `card.body`；`arow()` 会自动解包。"""

    def __init__(self, master, text: str = '', padding: int = 14,
                 radius: int = 16, **kw):
        parent_bg = BG
        try:
            parent_bg = master.cget('bg')
        except Exception:
            pass
        super().__init__(master, bg=parent_bg, bd=0, highlightthickness=0, **kw)
        self._pad = padding
        self._radius = radius
        self._title = (text or '').strip()
        self.cv = tk.Canvas(self, bg=parent_bg, highlightthickness=0, bd=0,
                            height=10)
        self.cv.pack(fill='both', expand=True)
        self.body = tk.Frame(self.cv, bg=CARD)
        self._win = self.cv.create_window(padding, padding + (22 if self._title else 0),
                                          anchor='nw', window=self.body)
        self._title_id = None
        self.cv.bind('<Configure>', self._redraw)
        self.body.bind('<Configure>', self._fit)

    # --- 布局 ---
    def _redraw(self, _ev=None):
        w = max(40, self.cv.winfo_width())
        h = max(40, self.cv.winfo_height())
        self.cv.delete('cardbg')
        round_rect(self.cv, 1, 1, w - 1, h - 1, self._radius, fill=CARD,
                   outline=BORDER, width=1, tags='cardbg')
        self.cv.tag_lower('cardbg')
        if self._title:
            if self._title_id is None:
                self._title_id = self.cv.create_text(
                    self._pad + 2, 12, text=self._title, anchor='nw',
                    fill=FG2, font=('Segoe UI Variable Display', 9))
            self.cv.coords(self._title_id, self._pad + 2, 12)
        top = self._pad + (24 if self._title else 0)
        self.cv.coords(self._win, self._pad, top)
        self.cv.itemconfigure(self._win, width=max(40, w - 2 * self._pad))
        if self.body.winfo_reqheight() > 0:
            self.cv.configure(height=self.body.winfo_reqheight() + top + self._pad)
            self.configure(height=self.cv.winfo_reqheight())

    def _fit(self, _ev=None):
        self.after_idle(self._redraw)


class RoundButton(tk.Canvas):
    """圆角按钮（悬停/按下有反馈）。用法同 Button：text=, command=。"""

    def __init__(self, master, text: str, command=None, kind: str = 'plain',
                 radius: int = 10, pad_x: int = 18, pad_y: int = 9,
                 font=None, width: int = 0, **kw):
        bg = BG
        try:
            bg = master.cget('bg')
        except Exception:
            pass
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, **kw)
        self._cmd = command
        self._kind = kind
        self._r = radius
        self._hover = False
        f = font or ('Segoe UI Variable Display', 10)
        self._txt = self.create_text(pad_x, pad_y, text=text, anchor='nw',
                                     fill=self._fg(), font=f)
        bb = self.bbox(self._txt) or (0, 0, pad_x * 2, pad_y * 2)
        w, h = bb[2] + pad_x, bb[3] + pad_y
        if width:
            w = max(w, width)
        self.configure(width=w, height=h)
        self._cw, self._ch = w, h   # 注意：不能用 _w/_h，tkinter 内部用 _w 存控件路径
        self._shape = round_rect(self, 0, 0, w, h, radius,
                                 fill=self._fill(), outline=self._outline(), width=1)
        self.tag_lower(self._shape)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _fill(self):
        if self._kind == 'primary':
            return ACCENT_D if self._hover else ACCENT
        if self._kind == 'danger':
            return '#fff1f0' if self._hover else CARD
        return '#f0f0f3' if self._hover else CARD

    def _fg(self):
        if self._kind == 'primary':
            return '#ffffff'
        if self._kind == 'danger':
            return '#ff3b30'
        return FG

    def _outline(self):
        return ACCENT if self._kind == 'primary' else BORDER

    def _repaint(self):
        self.itemconfigure(self._shape, fill=self._fill())
        self.itemconfigure(self._txt, fill=self._fg())

    def _on_enter(self, _e):
        self._hover = True
        self._repaint()

    def _on_leave(self, _e):
        self._hover = False
        self._repaint()

    def _on_press(self, _e):
        self.move(self._txt, 0, 1)

    def _on_release(self, _e):
        self.move(self._txt, 0, -1)
        if self._cmd:
            self._cmd()

    def set_text(self, s: str):
        self.itemconfigure(self._txt, text=s)

    def set_state(self, enabled: bool):
        self.configure(cursor='' if enabled else 'arrow')
        self.itemconfigure(self._txt, fill=self._fg() if enabled else '#c7c7cc')
