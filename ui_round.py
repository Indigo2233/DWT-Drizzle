# -*- coding: utf-8 -*-
"""分支 D 的圆角卡片（纯 tkinter）。

实现要点：卡片本身是 tk.Frame（原有 grid/pack 代码照常工作），
在它上面先 place 一个 Canvas 画圆角白底；后创建的控件自然叠在画布之上。
"""
from __future__ import annotations

import tkinter as tk

BG = '#f5f5f7'
CARD = '#ffffff'
CARD_DARK = '#0f1520'
BORDER = '#e3e3e8'
FG = '#1d1d1f'
FG2 = '#6e6e73'
ACCENT = '#0071e3'
ACCENT_D = '#0059b3'


def round_rect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """用平滑多边形画圆角矩形。"""
    r = max(1, min(int(r), (x1 - x0) // 2, (y1 - y0) // 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
           x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
           x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, splinesteps=24, **kw)


class RoundCard(tk.Frame):
    """白底圆角卡片；`card` 本身就是内容容器。"""

    def __init__(self, master, text: str = '', padding: int = 14,
                 radius: int = 16, dark: bool = False, **kw):
        parent_bg = BG
        try:
            parent_bg = master.cget('bg')
        except Exception:
            pass
        super().__init__(master, bg=parent_bg, bd=0, highlightthickness=0, **kw)
        self._pad = padding
        self._radius = radius
        self._title = (text or '').strip()
        self._dark = dark
        self._fill = CARD_DARK if dark else CARD
        self._cv = tk.Canvas(self, bg=parent_bg, highlightthickness=0, bd=0)
        self._cv.place(x=0, y=0, relwidth=1, relheight=1)
        tk.Misc.lower(self._cv)   # Canvas.lower 是「降低画布内元素」，控件层级要用 Misc.lower
        self.body = self
        self._tid = None
        self._seen = set()
        self._spaced = False
        self.bind('<Configure>', self._redraw)
        self.after(120, self._pad_children)

    def _pad_children(self):
        """给卡片内的控件加内边距（tk 没有 padding，只能逐个 grid/pack 调）。"""
        try:
            kids = [w for w in self.winfo_children() if w is not self._cv]
            if not kids:
                self.after(120, self._pad_children)
                return
            if not self._spaced and self._title:
                # 标题单独占第 0 行，其它控件整体下移一行
                for w in kids:
                    gi = w.grid_info()
                    if gi:
                        w.grid_configure(row=int(gi['row']) + 1)
                lb = tk.Label(self, text=self._title, bg=self._fill,
                              fg=('#9aa4b2' if self._dark else FG2),
                              font=('Segoe UI Variable Display', 9))
                lb.grid(row=0, column=0, columnspan=8, sticky='w',
                        padx=self._pad + 2, pady=(self._pad, 2))
                self._seen.add(lb)
                self._spaced = True
            for w in kids:
                if w in self._seen:
                    continue
                self._seen.add(w)
                gi = w.grid_info()
                if gi:
                    _c = int(gi.get('column', 0))
                    w.grid_configure(padx=(self._pad if _c == 0 else 8,
                                           self._pad if _c >= 2 else 8),
                                     pady=4)
                else:
                    w.pack_configure(padx=self._pad, pady=(self._pad, 4))
            self.after(150, self._pad_children)
        except Exception:
            pass

    def _redraw(self, _ev=None):
        w = max(40, self.winfo_width())
        h = max(40, self.winfo_height())
        self._cv.delete('bg')
        round_rect(self._cv, 1, 1, w - 1, h - 1, self._radius,
                   fill=self._fill,
                   outline='#1c2431' if self._dark else BORDER,
                   width=1, tags='bg')
        # 标题改用真正的 Label（见 _pad_children），画在 Canvas 上会被 spacer 盖住


class RoundButton(tk.Canvas):
    """圆角按钮（悬停/按下反馈）。"""

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
        self._hover = False
        f = font or ('Segoe UI Variable Display', 10)
        self._txt = self.create_text(pad_x, pad_y, text=text, anchor='nw',
                                     fill=self._fg(), font=f)
        bb = self.bbox(self._txt) or (0, 0, pad_x * 2, pad_y * 2)
        w, h = bb[2] + pad_x, bb[3] + pad_y
        if width:
            w = max(w, width)
        self.configure(width=w, height=h)
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
        return '#ececf0' if self._hover else '#f6f6f8'

    def _fg(self):
        if self._kind == 'primary':
            return '#ffffff'
        if self._kind == 'danger':
            return '#ff3b30'
        return FG

    def _outline(self):
        return ACCENT if self._kind == 'primary' else '#d2d2d8'

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


class SegTabs(tk.Frame):
    """分段控件（胶囊）+ 内容区，替代 ttk.Notebook。

    用法与 Notebook 对齐：add(page, text=...)、select(page)。
    page 必须是本控件的子控件（gui.py 里就是 `tk.Frame(self.nb)`）。
    """

    def __init__(self, master, bg=None, **kw):
        _bg = bg or BG
        super().__init__(master, bg=_bg, **kw)
        self._bg = _bg
        try:
            self._s = max(1.0, self.winfo_fpixels('1i') / 96.0)
        except Exception:
            self._s = 1.0
        self._pages = []          # [(page, title, seg_index)]
        self._cur = None
        self._bar = tk.Canvas(self, bg=_bg, highlightthickness=0, height=self._p(46))
        self._bar.pack(fill='x', padx=self._p(10), pady=(self._p(6), self._p(2)))
        self._bar.bind('<Button-1>', self._on_click)
        self._hot = -1
        self._last_w = 0
        self._bar.bind('<Motion>', self._on_motion)
        self._bar.bind('<Configure>', self._on_resize)

    def _p(self, v):
        return int(round(v * self._s))

    def _fn(self, v):
        return max(7, int(round(v * self._s)))

    # --- 数据 -------------------------------------------------------
    def add(self, page, text=''):
        self._pages.append([page, text, 0])
        page.pack_forget()
        if self._cur is None:
            self.select(page)
        self._draw()
        return page

    def select(self, page=None):
        if page is None:
            return self._cur
        for p, _t, _i in self._pages:
            if p is page:
                self._cur = p
                for q, _t2, _i2 in self._pages:
                    if q is not p:
                        q.pack_forget()
                p.pack(fill='both', expand=True, padx=self._p(10), pady=(0, self._p(10)))
        self._draw()
        return page

    # --- 绘制 -------------------------------------------------------
    def _geom(self):
        W = max(self._bar.winfo_width(), 200)
        n = max(1, len(self._pages))
        pad = self._p(4)
        seg_w = min(self._p(150), max(self._p(96), (W - pad * 2) // n))
        h = self._p(36)
        y0 = (self._p(46) - h) // 2
        return W, n, pad, seg_w, h, y0

    def _on_resize(self, e):
        # 只在宽度真正变化时重画，避免「画内容 → 触发 Configure → 再画」的死循环
        if abs(e.width - self._last_w) >= 2:
            self._last_w = e.width
            self._draw()

    def _draw(self):
        c = self._bar
        c.delete('all')
        if not self._pages:
            return
        W, n, pad, seg_w, h, y0 = self._geom()
        total = seg_w * n
        cx0 = max(0, (W - total) // 2)
        round_rect(c, cx0, y0, cx0 + total, y0 + h, self._p(10),
                   fill='#ececf0', outline=BORDER, width=1)
        cur = next((i for i, (_p, _t, _) in enumerate(self._pages) if _p is self._cur), 0)
        for i, (_p, title, _x) in enumerate(self._pages):
            x0 = cx0 + i * seg_w
            if i == cur:
                round_rect(c, x0 + 2, y0 + 2, x0 + seg_w - 2, y0 + h - 2, self._p(9),
                           fill=CARD, outline='#d8d8dd', width=1)
            elif i == self._hot:
                round_rect(c, x0 + 2, y0 + 2, x0 + seg_w - 2, y0 + h - 2, self._p(9),
                           fill='#f4f4f7', outline='', width=0)
            c.create_text(x0 + seg_w / 2, y0 + h / 2, text=title,
                          fill=(FG if i == cur else FG2),
                          font=('Segoe UI Variable Display', self._fn(9)))

    def _hit(self, x):
        W, n, pad, seg_w, h, y0 = self._geom()
        total = seg_w * n
        cx0 = max(0, (W - total) // 2)
        if x < cx0 or x >= cx0 + total:
            return -1
        return min(n - 1, int((x - cx0) // seg_w))

    def _on_motion(self, e):
        i = self._hit(e.x)
        if i != self._hot:
            self._hot = i
            self._draw()

    def _on_click(self, e):
        i = self._hit(e.x)
        if 0 <= i < len(self._pages):
            self.select(self._pages[i][0])
