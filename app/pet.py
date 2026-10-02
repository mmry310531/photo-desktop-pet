"""桌面寵物本體：讓訓練好的寵物在 Windows 桌面上生活。

牠會：在工作列上散步、坐下發呆、累了躺下睡覺（你離開電腦太久牠也會睡）、
跳到其他視窗的頂端、從視窗邊緣掉下去、跟著視窗一起移動；
你可以拎起牠丟出去、點牠、用滑鼠來回摸牠、叫牠過來。
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

import winenv
from common import launch, launch_cmd, list_packs, load_pack, load_settings, pack_dir, save_settings

APP_KEY = "photo-desktop-pet-v1"
GRAVITY = 2600.0  # px/s²


# ======================================================================= 世界
@dataclass
class Surface:
    x1: float
    x2: float
    y: float
    hwnd: int = 0  # 0 = 螢幕底部（工作列上方）


class World:
    """掌握「地面」在哪：每個螢幕的底部 + 其他視窗沒被遮住的頂邊。"""

    def __init__(self, settings):
        self.settings = settings
        self.surfaces: list[Surface] = []
        self.win_rects: dict[int, tuple[float, float, float, float]] = {}
        self.refresh()

    # 實體像素 -> Qt 邏輯座標（處理 125%/150% 縮放與多螢幕）
    def _mapper(self):
        screens = QApplication.screens()
        mons = winenv.monitors()
        maps = []
        for dev, (l, t, r, b) in mons:
            qs = next((s for s in screens if s.name() == dev), None)
            if qs is None:
                qs = next((s for s in screens if s.geometry().topLeft() == QPoint(l, t)), None)
            if qs is None:
                continue
            g = qs.geometry()
            dpr = qs.devicePixelRatio() or 1.0
            maps.append(((l, t, r, b), g.left(), g.top(), dpr))

        def to_logical(x, y):
            for (l, t, r, b), gx, gy, dpr in maps:
                if l <= x < r and t <= y < b:
                    return gx + (x - l) / dpr, gy + (y - t) / dpr
            if maps:  # 不在任何螢幕上：用最近的那個
                (l, t, r, b), gx, gy, dpr = min(
                    maps, key=lambda m: abs(x - (m[0][0] + m[0][2]) / 2) + abs(y - (m[0][1] + m[0][3]) / 2))
                return gx + (x - l) / dpr, gy + (y - t) / dpr
            d = QApplication.primaryScreen().devicePixelRatio() or 1.0
            return x / d, y / d

        return to_logical

    def bounds(self):
        r = QRectF()
        for s in QApplication.screens():
            r = r.united(QRectF(s.availableGeometry()))
        return r

    def refresh(self):
        surfaces = []
        for s in QApplication.screens():
            ag = s.availableGeometry()
            surfaces.append(Surface(ag.left(), ag.left() + ag.width(), ag.top() + ag.height(), 0))
        self.win_rects = {}
        if self.settings.get("climb_windows", True) and winenv.IS_WIN:
            to_l = self._mapper()
            screen_tops = [s.availableGeometry().top() for s in QApplication.screens()]
            higher: list[tuple[float, float, float, float]] = []
            for hwnd, (l, t, r, b), zoomed in winenv.top_windows():
                x1, y1 = to_l(l, t)
                x2, y2 = to_l(r - 1, b - 1)
                rect = (x1, y1, x2, y2)
                self.win_rects[hwnd] = rect
                if not zoomed and all(abs(y1 - st) > 40 for st in screen_tops):
                    segs = [(x1, x2)]
                    for hx1, hy1, hx2, hy2 in higher:  # 減去被上層視窗蓋住的部分
                        if hy1 - 2 <= y1 <= hy2:
                            nxt = []
                            for a, c in segs:
                                if hx2 <= a or hx1 >= c:
                                    nxt.append((a, c))
                                else:
                                    if hx1 > a:
                                        nxt.append((a, hx1))
                                    if hx2 < c:
                                        nxt.append((hx2, c))
                            segs = nxt
                    for a, c in segs:
                        if c - a >= 70:
                            surfaces.append(Surface(a, c, y1, hwnd))
                higher.append(rect)
        self.surfaces = surfaces

    def support_at(self, x, y, tol=3.0, hwnd=None):
        best = None
        for s in self.surfaces:
            if s.x1 <= x <= s.x2 and abs(s.y - y) <= tol and (hwnd is None or s.hwnd == hwnd):
                if best is None or s.hwnd:  # 視窗優先於地板
                    best = s
        return best

    def landing(self, x, y_prev, y_new):
        cands = [s for s in self.surfaces if s.x1 <= x <= s.x2 and y_prev - 1 <= s.y <= y_new + 0.5]
        return min(cands, key=lambda s: s.y) if cands else None

    def floor_below(self, x):
        cands = [s for s in self.surfaces if s.hwnd == 0 and s.x1 <= x <= s.x2]
        if not cands:
            cands = [s for s in self.surfaces if s.hwnd == 0]
            x_ok = min(cands, key=lambda s: min(abs(x - s.x1), abs(x - s.x2)))
            return x_ok
        return max(cands, key=lambda s: s.y)


# ======================================================================= 圖片
def pil_to_pixmap(im, scale, dpr, mirror):
    from PIL import Image

    w, h = max(1, round(im.width * scale * dpr)), max(1, round(im.height * scale * dpr))
    im2 = im.resize((w, h), Image.LANCZOS)
    if mirror:
        im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
    qi = QImage(im2.tobytes(), w, h, w * 4, QImage.Format_RGBA8888).copy()
    pm = QPixmap.fromImage(qi)
    pm.setDevicePixelRatio(dpr)
    return pm


class Unit:
    """一段動作：一張靜止照片（1 格），或從影片做出的連續動畫（很多格）。

    kind: still = 靜止照片；loop = 可循環動作（走路、呼吸、搖尾巴）；trans = 轉場（坐下、趴下、起身）
    每一格：(朝右的圖, 朝左的圖, 朝右時的錨點 x, 寬, 高)
    """

    def __init__(self, frames, fps, pose, kind="still", to=None):
        self.frames, self.fps, self.pose, self.kind, self.to = frames, fps or 10, pose, kind, to
        self.real = False
        self.speed = 0.0
        self.w = max(f[3] for f in frames)
        self.h = max(f[4] for f in frames)
        # 舊程式碼相容
        self.right, self.left = frames[0][0], frames[0][1]

    @property
    def animated(self):
        return len(self.frames) > 1

    def duration(self):
        return len(self.frames) / self.fps

    def frame_at(self, t, loop=True, reverse=False):
        n = len(self.frames)
        i = int(t * self.fps)
        i = i % n if loop else min(i, n - 1)
        return self.frames[n - 1 - i if reverse else i]


POSES_ALL = ("walk", "sit", "lie")
# 每種行為需要的身體姿勢
POSE_FOR = {"walk": "walk", "chase": "walk", "look": "walk", "sit": "sit", "stretch": "sit", "sleep": "lie"}


def load_sprites(name, settings, dpr):
    """讀寵物包 → (loops: {姿勢: [Unit]}, trans: {(從, 到): [(Unit, 是否倒放)]}, pack)"""
    import numpy as np
    from PIL import Image

    pack = load_pack(name)
    d = pack_dir(name)
    entries = []
    for m in pack["images"]:
        try:
            if m.get("frames"):
                ims = [(Image.open(d / f["file"]).convert("RGBA"), f.get("ax")) for f in m["frames"]]
            else:
                ims = [(Image.open(d / m["file"]).convert("RGBA"), None)]
            entries.append((m, ims))
        except Exception:
            pass
    if not entries:
        return {}, {}, pack

    def lum(im):
        arr = np.asarray(im).astype(np.float32)
        msk = arr[:, :, 3] > 128
        return float(arr[:, :, :3][msk].mean()) if msk.any() else 128.0

    lums = [lum(ims[len(ims) // 2][0]) for _, ims in entries]
    target_l = float(np.median(lums))
    base = settings.get("base_px", 130) * settings.get("size", 1.0)
    target_area = base * base * 0.55
    loops = {k: [] for k in POSES_ALL}
    trans = {}
    for (m, ims), l in zip(entries, lums):
        gain = float(np.clip(1 + 0.5 * (target_l / max(l, 1) - 1), 0.75, 1.35))  # 亮度往整組中間拉
        mid = ims[len(ims) // 2][0]
        if m.get("kind") == "trans":
            # 轉場：用開頭幾格的大小對齊，接上前後動作時才不會忽大忽小
            area = float(np.mean([(np.asarray(im.getchannel("A")) > 128).sum() for im, _ in ims[:3]]))
        else:
            area = m.get("area") or int((np.asarray(mid.getchannel("A")) > 128).sum())
        sc = math.sqrt(target_area / max(area, 1))
        maxh = max(im.height for im, _ in ims)
        maxw = max(im.width for im, _ in ims)
        sc = min(sc, base * 1.5 / maxh, base * 2.4 / maxw)
        if m.get("real"):
            # 寫實片段：全部共用同一個比例（坐姿在畫布中約佔 50% 高 → 顯示成 base*1.3 高）
            sc = base * 1.3 / (480 * 0.5)
            gain = 1.0
        facing_right = m.get("facing", "right") == "right"
        frames = []
        for im, ax in ims:
            if gain != 1.0:
                arr = np.asarray(im).astype(np.float32)
                arr[:, :, :3] = np.clip(arr[:, :, :3] * gain, 0, 255)
                im = Image.fromarray(arr.astype(np.uint8), "RGBA")
            if ax is None:  # 靜止照片：錨點用身體質心
                solid = np.asarray(im.getchannel("A")) > 128
                ax = float(np.where(solid)[1].mean()) if solid.any() else im.width / 2
            right = pil_to_pixmap(im, sc, dpr, mirror=not facing_right)
            left = pil_to_pixmap(im, sc, dpr, mirror=facing_right)
            w, h = right.width() / dpr, right.height() / dpr
            axr = ax * sc if facing_right else w - ax * sc
            frames.append((right, left, axr, w, h))
        pose = m.get("pose", "walk")
        kind = m.get("kind") or ("loop" if len(frames) > 1 else "still")
        if kind == "trans" and m.get("to") in POSES_ALL and m.get("to") != pose:
            u = Unit(frames, m.get("fps"), pose, "trans", m["to"])
            u.real = bool(m.get("real"))
            trans.setdefault((pose, m["to"]), []).append((u, False))
            trans.setdefault((m["to"], pose), []).append((u, True))  # 倒放：坐下 ↔ 起身
            # 轉場的頭尾也可以當靜止姿勢
            loops[pose].append(Unit([frames[0]], 10, pose))
            loops[m["to"]].append(Unit([frames[-1]], 10, m["to"]))
        else:
            u = Unit(frames, m.get("fps"), pose, kind)
            u.real = bool(m.get("real"))
            u.speed = (m.get("px_per_s") or 0) * sc  # 寫實走路：螢幕上的移動速度＝影片裡的步速，腳不會打滑
            loops.setdefault(pose, []).append(u)
    # 有寫實影片的姿勢就只用寫實影片（混用照片會大小、光線不一致，一眼假）
    for k in POSES_ALL:
        real = [u for u in loops.get(k, []) if getattr(u, "real", False)]
        if real:
            loops[k] = real
    if any(getattr(u, "real", False) for v in trans.values() for u, _ in v):
        trans = {k: [(u, r) for u, r in v if getattr(u, "real", False)] or v for k, v in trans.items()}
    allu = [u for v in loops.values() for u in v]
    loops["walk"] = loops["walk"] or loops["sit"] or allu
    loops["sit"] = loops["sit"] or loops["walk"]
    loops["lie"] = loops["lie"] or loops["sit"]
    return loops, trans, pack


# ======================================================================= 特效
@dataclass
class Particle:
    text: str
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: QColor
    size: int = 14


# ======================================================================= 寵物
class Pet(QWidget):
    def __init__(self, ctrl, name):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowDoesNotAcceptFocus | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.ctrl, self.name, self.world = ctrl, name, ctrl.world
        self.particles: list[Particle] = []
        self.bubble = ("", 0.0)
        self.load()

        # 物理狀態（x = 腳底中心，y = 腳底）
        b = self.world.bounds()
        self.x = random.uniform(b.left() + 150, b.right() - 150)
        self.y = b.top() - 10
        self.vx = random.uniform(-200, 200)
        self.vy = 0.0
        self.dir = random.choice([-1, 1])
        self.support: Surface | None = None
        self.support_ref = None
        self.state = "fall"
        self.state_t = 0.0
        self.state_len = 0.0
        self.t = 0.0
        self.phase = 0.0
        self.speed = 90.0
        self.squash = 0.0
        self.energy = random.uniform(0.6, 1.0)
        self.pose = "walk"
        self.sprite = self.pick_loop("walk")
        self.unit_t = 0.0
        self.unit_loop, self.unit_rev = True, False
        self.queue = []
        self.pending = None
        self.fade_from, self.fade_t, self.fade_len = None, 0.0, 0.2
        self.target_x = None
        self.edge_choice = None
        self.drag_off = QPointF()
        self.drag_samples = []
        self.press_pos = None
        self.dragging = False
        self.drag_rot = 0.0
        self.pet_meter = 0.0
        self.last_mouse = None
        self.was_away = False
        self.show()

    # ---------------- 載入
    def load(self):
        dpr = self.screen().devicePixelRatio() if self.screen() else 1.0
        self.sprites, self.trans, self.pack = load_sprites(self.name, self.ctrl.settings, dpr)
        if not self.sprites:
            raise RuntimeError(f"{self.name} 沒有任何圖片")
        allsp = [s for v in self.sprites.values() for s in v] + [u for v in self.trans.values() for u, _ in v]
        mw = max(s.w for s in allsp)
        mh = max(s.h for s in allsp)
        self.W = int(max(mw, mh) * 1.5 + 40)
        self.H = int(mh * 1.9 + 60)
        self.FOOT = 14  # 腳底到視窗下緣的距離（留給陰影）
        self.setFixedSize(self.W, self.H)
        self.phrases = self.pack.get("phrases") or ["喵～"]
        if hasattr(self, "sprite"):
            self.queue, self.pending = [], None
            self.sprite = self.pick_loop(self.pose)
            self.unit_t, self.unit_loop, self.unit_rev = 0.0, True, False

    # ---------------- 動畫
    def pick_loop(self, pose):
        units = self.sprites.get(pose) or self.sprites["sit"]
        anim = [u for u in units if u.animated]
        if anim and random.random() < 0.85:  # 有真的會動的影片就優先用
            return random.choice(anim)
        return random.choice(units)

    def cur_frame(self):
        return self.sprite.frame_at(self.unit_t, self.unit_loop, self.unit_rev)

    @property
    def fw(self):
        return self.cur_frame()[3]

    @property
    def fh(self):
        return self.cur_frame()[4]

    def play(self, unit, loop=True, reverse=False, fade=0.18):
        """換一段動作；新舊畫面淡入淡出，不會硬切。"""
        if unit is self.sprite and loop == self.unit_loop and reverse == self.unit_rev:
            return
        if fade > 0 and getattr(self, "sprite", None) is not None:
            self.fade_from = (self.cur_frame(), self.dir)
            self.fade_t = self.fade_len = fade
        self.sprite, self.unit_t, self.unit_loop, self.unit_rev = unit, 0.0, loop, reverse

    def plan(self, a, b):
        """找從姿勢 a 到 b 的路線（例如 走→坐→躺）。

        有轉場影片的步驟用影片（代價 1），沒有的用淡入淡出補（代價 3），
        所以「走→坐」沒影片、「坐→躺」有影片時，會先淡到坐、再真的趴下去。
        回傳步驟清單：(Unit, 是否倒放) 或 ("fade", 姿勢)。
        """
        import heapq

        best = {a: (0, [])}
        heap = [(0, a, [])]
        while heap:
            cost, cur, path = heapq.heappop(heap)
            if cur == b:
                return path
            if cost > best.get(cur, (1e9,))[0]:
                continue
            for nxt in POSES_ALL:
                if nxt == cur:
                    continue
                lst = self.trans.get((cur, nxt))
                # 淡入淡出的代價看姿勢差多遠（走→躺 比 走→坐 遠），優先走「有真實轉場」的路
                gap = abs(POSES_ALL.index(cur) - POSES_ALL.index(nxt))
                step, c = ((random.choice(lst)), 1) if lst else (("fade", nxt), 1 + 2 * gap)
                nc = cost + c
                if nc < best.get(nxt, (1e9,))[0]:
                    best[nxt] = (nc, path + [step])
                    heapq.heappush(heap, (nc, nxt, path + [step]))
        return None

    def start_step(self, step):
        if step[0] == "fade":
            self.play(self.pick_loop(step[1]), fade=0.35)
            self.step_len = 0.45
            self.step_to = step[1]
        else:
            u, rev = step
            self.play(u, loop=False, reverse=rev, fade=0.1)
            self.step_len = u.duration()
            self.step_to = u.pose if rev else u.to

    # ---------------- 狀態切換
    def set_state(self, st, length=None):
        length = length if length is not None else 3.0
        need = self.pose if st == "petted" else POSE_FOR.get(st, self.pose)
        if self.grounded() and need != self.pose:
            path = self.plan(self.pose, need)
            if path and any(step[0] != "fade" for step in path):  # 用真實的轉場動作過去（坐下、趴下、起身…）
                self.queue = path
                self.pending = (st, length)
                self.state, self.state_t = "trans", 0.0
                self.start_step(self.queue.pop(0))
                return
            self.play(self.pick_loop(need), fade=0.35)  # 完全沒有轉場影片：慢一點淡過去
        elif self.sprite.pose != need or self.sprite.kind == "trans" or random.random() < 0.4:
            self.play(self.pick_loop(need))
        self.pose = need
        self.state, self.state_t = st, 0.0
        self.state_len = length

    def trans_tick(self):
        """這一步轉場播完 → 下一步，或進入原本要做的事。"""
        if self.unit_t < self.step_len:
            return
        self.pose = self.step_to or self.pose
        if self.queue:
            self.start_step(self.queue.pop(0))
            return
        st, length = self.pending or ("sit", 3.0)
        self.pending = None
        self.state = None
        self.set_state(st, length)

    def grounded(self):
        return self.state not in ("fall", "drag")

    def interrupt(self):
        self.queue, self.pending = [], None

    def start_fall(self, vx=None, vy=0.0):
        self.support = None
        self.vx = self.dir * self.speed if vx is None else vx
        self.vy = vy
        self.state = "fall"
        self.state_t = 0
        self.interrupt()
        if self.pose != "walk" or not self.unit_loop:
            self.pose = "walk"
            self.play(self.pick_loop("walk"), fade=0.1)

    def summon_drop(self):
        g = QApplication.primaryScreen().availableGeometry()
        self.x = random.uniform(g.left() + 100, g.right() - 100)
        self.y = g.top() - 20
        self.start_fall(vx=random.uniform(-150, 150))

    def say(self, text, secs=2.2):
        self.bubble = (text, secs)

    def emit(self, text, n=1, color=QColor(255, 80, 120), size=16, spread=30, vy=-60):
        top = self.y - self.fh
        for _ in range(n):
            self.particles.append(Particle(text, self.x + random.uniform(-spread, spread),
                                           top + random.uniform(-5, 15), random.uniform(-15, 15),
                                           vy * random.uniform(0.7, 1.3), random.uniform(1.2, 2.0), color, size))

    def decide(self):
        """當前動作做完了，決定下一件事。"""
        quiet = self.ctrl.settings.get("quiet")
        idle = winenv.idle_seconds()
        hour = time.localtime().tm_hour
        night = hour >= 23 or hour < 6
        mouse = QCursor.pos()
        mouse_near = abs(mouse.x() - self.x) < 700 and abs(mouse.y() - self.y) < 400
        opts = {
            "walk": 0 if quiet else 4.0 * self.energy,
            "sit": 2.5,
            "look": 1.5,
            "sleep": (0.25 + (1 - self.energy) * 2.5 + (8 if idle > 90 else 0) + (2 if night else 0)),
            "chase": 0 if quiet or not mouse_near or idle > 30 else 1.0,
            "climb": 0 if quiet else (1.6 * self.energy if self.climb_target() else 0),
        }
        st = random.choices(list(opts), weights=list(opts.values()))[0]
        if st == "walk":
            if random.random() < 0.6:
                self.dir = random.choice([-1, 1])
            self.set_state("walk", random.uniform(2.5, 8))
            if self.sprite.speed:  # 寫實走路：跟影片裡的步速一樣
                self.speed = self.sprite.speed * (random.uniform(0.9, 1.1) if random.random() < 0.85 else 1.6)
            elif self.sprite.animated:  # 用影片走路：速度配合步伐
                self.speed = self.sprite.w * 0.9 * (random.uniform(0.8, 1.15) if random.random() < 0.8 else 1.9)
            else:
                self.speed = random.uniform(55, 110) if random.random() < 0.8 else random.uniform(180, 260)
        elif st == "sit":
            self.set_state("sit", random.uniform(4, 12))
        elif st == "look":
            self.dir *= -1
            self.set_state("look", random.uniform(1.5, 3.5))
        elif st == "sleep":
            self.set_state("sleep", random.uniform(15, 50) if idle < 90 else random.uniform(60, 180))
        elif st == "chase":
            self.set_state("chase", 6)
            self.speed = (self.sprite.speed * 1.6 if self.sprite.speed else
                          (self.sprite.w * 1.6) if self.sprite.animated else random.uniform(150, 230))
        elif st == "climb":
            self.jump_to(self.climb_target())

    def climb_target(self):
        if not self.support:
            return None
        cands = []
        for s in self.world.surfaces:
            if s is self.support or s.hwnd == 0:
                continue
            dy = self.y - s.y
            if 60 < dy < 420:
                tx = min(max(self.x, s.x1 + 30), s.x2 - 30)
                if abs(tx - self.x) < 450 and s.x2 - s.x1 > 90:
                    cands.append((s, tx))
        return random.choice(cands) if cands else None

    def jump_to(self, target):
        s, tx = target
        h = self.y - s.y + 50
        vy = -math.sqrt(2 * GRAVITY * h)
        t_up = -vy / GRAVITY
        t_down = math.sqrt(2 * 50 / GRAVITY)
        self.dir = 1 if tx > self.x else -1
        self.squash = 0.25
        self.start_fall(vx=(tx - self.x) / (t_up + t_down), vy=vy)

    # ---------------- 每一幀
    def tick(self, dt):
        self.t += dt
        self.state_t += dt
        self.squash = max(0.0, self.squash - dt * 3)
        b = self.world.bounds()
        half = self.fw / 2
        # 動畫時間：走路的影片會跟移動速度同步（腳步不會滑），掉落／被拎著時暫停
        if self.state not in ("fall", "drag"):
            rate = 1.0
            if self.state in ("walk", "chase") and self.sprite.speed:
                rate = max(0.4, min(2.5, self.speed / self.sprite.speed))
            elif self.state in ("walk", "chase") and self.sprite.animated:
                rate = max(0.4, min(2.5, self.speed / max(self.sprite.w * 0.9, 1)))
            elif self.state == "sleep":
                rate = 0.8
            self.unit_t += dt * rate
        if self.fade_t > 0:
            self.fade_t = max(0.0, self.fade_t - dt)

        idle = winenv.idle_seconds()
        if idle > 90:
            self.was_away = True
        elif self.was_away and idle < 1.5:
            self.was_away = False
            if self.state == "sleep":
                self.set_state("stretch", 1.6)
                self.say(random.choice(["你回來了！", "!", "喵！"]))

        if self.state == "drag":
            pass
        elif self.state == "fall":
            y_prev = self.y
            self.vy = min(self.vy + GRAVITY * dt, 3200)
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.x - half < b.left():
                self.x, self.vx = b.left() + half, abs(self.vx) * 0.5
            if self.x + half > b.right():
                self.x, self.vx = b.right() - half, -abs(self.vx) * 0.5
            if self.vy > 0:
                s = self.world.landing(self.x, y_prev, self.y)
                if s is None and self.y > b.bottom() + 200:  # 掉出世界：拉回地板
                    s = self.world.floor_below(self.x)
                    self.x = min(max(self.x, s.x1 + half), s.x2 - half)
                if s is not None:
                    self.y = s.y
                    if self.vy > 1300:  # 摔得重會彈一下
                        self.squash = 0.6
                        self.vy = -self.vy * 0.22
                        self.vx *= 0.5
                        self.y -= 1
                        if random.random() < 0.5:
                            self.emit("💫", 1, size=14)
                    else:
                        self.land(s)
        else:
            self.ground_tick(dt)

        # 特效
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]
        if self.bubble[1] > 0:
            self.bubble = (self.bubble[0], self.bubble[1] - dt)
        self.pet_meter = max(0.0, self.pet_meter - dt * 250)

        self.place()
        self.update()

    def land(self, s):
        self.support = s
        self.support_ref = self.world.win_rects.get(s.hwnd) if s.hwnd else None
        self.vx = self.vy = 0
        self.squash = max(self.squash, 0.35)
        self.pose = "walk"
        self.set_state("look", random.uniform(0.6, 1.5))

    def ground_tick(self, dt):
        st = self.state
        self.energy = min(1.0, self.energy + dt * (0.02 if st == "sleep" else 0.0)) - (dt * 0.004 if st in ("walk", "chase") else 0)
        self.energy = max(0.0, self.energy)

        if st == "trans":
            self.trans_tick()
            return
        if st in ("walk", "chase"):
            if st == "chase":
                mx = QCursor.pos().x()
                if abs(mx - self.x) < 40:
                    self.set_state("sit", random.uniform(3, 6))
                    self.say(random.choice(self.phrases))
                    return
                self.dir = 1 if mx > self.x else -1
            nx = self.x + self.dir * self.speed * dt
            self.phase += self.speed * dt * 0.11
            s = self.support
            if s and (nx < s.x1 + 10 or nx > s.x2 - 10):
                if s.hwnd and self.edge_choice is None:
                    self.edge_choice = random.random() < 0.3  # 30% 機率直接走下去
                if s.hwnd and self.edge_choice:
                    if nx < s.x1 or nx > s.x2:
                        self.edge_choice = None
                        self.x = nx
                        self.start_fall(vx=self.dir * self.speed)
                        return
                else:
                    probe = nx + self.dir * 14  # 相鄰螢幕的地板接得上就繼續走
                    nx2 = self.world.support_at(probe, self.y)
                    if nx2 is None or (nx2.x1, nx2.x2) == (s.x1, s.x2):
                        self.dir *= -1
                        self.edge_choice = None
                        nx = self.x
                    else:
                        self.support = nx2
            self.x = nx
            if self.state_t > self.state_len:
                self.edge_choice = None
                self.decide()
        elif st == "sleep":
            if random.random() < dt * 0.9:
                self.emit("z", 1, QColor(120, 150, 255), size=random.choice([12, 15, 19]), spread=8, vy=-30)
            if self.state_t > self.state_len and winenv.idle_seconds() < 60:
                self.set_state("stretch", 1.6)
        elif self.state_t > self.state_len:
            self.decide()

    def follow_support(self):
        """被站著的視窗移動／關閉／被蓋住時的反應。每次重新掃描世界後呼叫。"""
        if not self.grounded() or self.support is None:
            return
        s = self.support
        if s.hwnd:
            r = self.world.win_rects.get(s.hwnd)
            if r is None:
                self.start_fall(vx=0)
                return
            if self.support_ref:
                dx, dy = r[0] - self.support_ref[0], r[1] - self.support_ref[1]
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    self.x += dx
                    self.y += dy
                    if dy > 25 or dy < -25:
                        self.squash = 0.2
            self.support_ref = r
            ns = self.world.support_at(self.x, self.y, tol=3, hwnd=s.hwnd)
        else:
            ns = self.world.support_at(self.x, self.y, tol=4)
            if ns is None:  # 工作列位置／解析度改了
                ns = self.world.floor_below(self.x)
                self.y = ns.y
        if ns is None:
            self.start_fall(vx=0)
        else:
            self.support = ns

    # ---------------- 繪圖
    def place(self):
        self.move(round(self.x - self.W / 2), round(self.y - self.H + self.FOOT))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        cx, fy = self.W / 2, self.H - self.FOOT
        fr = self.cur_frame()
        pm = fr[0] if self.dir > 0 else fr[1]
        fax = fr[2] if self.dir > 0 else fr[3] - fr[2]
        fwid, fht = fr[3], fr[4]
        anim = self.sprite.animated
        sx = sy = 1.0
        rot = 0.0
        off = 0.0
        st = self.state
        if anim and st not in ("fall", "drag", "petted"):
            pass  # 真實影片動畫：不再加假的晃動
        elif st in ("walk", "chase"):
            k = min(1.0, self.speed / 120)
            off = -abs(math.sin(self.phase)) * 5 * k
            rot = math.sin(self.phase) * 2.5 * self.dir
        elif st == "sleep":
            sy = 1 + 0.03 * math.sin(self.t * 1.3)
            sx = 1 - 0.01 * math.sin(self.t * 1.3)
        elif st == "stretch":
            k = math.sin(min(self.state_t / self.state_len, 1) * math.pi)
            sx, sy = 1 + 0.12 * k, 1 - 0.1 * k
        elif st == "petted":
            rot = math.sin(self.t * 9) * 3
            sy = 1 + 0.02 * math.sin(self.t * 9)
        elif st == "fall":
            v = max(-1, min(1, self.vy / 2000))
            sy, sx = 1 + 0.1 * abs(v), 1 - 0.07 * abs(v)
            rot = max(-20, min(20, self.vx / 60))
        elif st not in ("drag", "trans"):
            sy = 1 + 0.018 * math.sin(self.t * 2.4)
            sx = 1 - 0.008 * math.sin(self.t * 2.4)
        if self.squash > 0:
            q = self.squash
            sy *= 1 - 0.35 * q
            sx *= 1 + 0.3 * q

        # 陰影
        if self.grounded():
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 45))
            p.drawEllipse(QPointF(cx, fy), fwid * 0.38 * sx, 5)

        def draw(pm, ax, w, h, opacity):
            p.save()
            p.setOpacity(opacity)
            if st == "drag":
                # 被拎著：以抓的點為支點晃
                p.translate(cx, fy - h)
                p.rotate(self.drag_rot)
                p.drawPixmap(QRectF(-ax, 0, w, h), pm, QRectF(pm.rect()))
            else:
                p.translate(cx, fy + off)
                p.rotate(rot)
                p.scale(sx, sy)
                p.drawPixmap(QRectF(-ax, -h, w, h), pm, QRectF(pm.rect()))
            p.restore()

        if self.fade_t > 0 and self.fade_from:
            (ofr, odir) = self.fade_from
            k = self.fade_t / max(self.fade_len, 1e-3)
            draw(ofr[0] if odir > 0 else ofr[1], ofr[2] if odir > 0 else ofr[3] - ofr[2], ofr[3], ofr[4], k)
            draw(pm, fax, fwid, fht, 1 - k * 0.6)
        else:
            draw(pm, fax, fwid, fht, 1.0)

        # 粒子：愛心、Zzz…
        ox, oy = self.x - cx, self.y - fy
        for pt in self.particles:
            a = max(0, min(255, int(255 * min(1.0, pt.life / 0.6))))
            c = QColor(pt.color)
            c.setAlpha(a)
            p.setPen(c)
            f = QFont("Segoe UI Emoji" if winenv.IS_WIN else "", pt.size)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QPointF(pt.x - ox - pt.size / 2, pt.y - oy), pt.text)

        # 對話泡泡
        text, left = self.bubble
        if left > 0 and text:
            f = QFont("Microsoft JhengHei UI" if winenv.IS_WIN else "", 11)
            p.setFont(f)
            fm = p.fontMetrics()
            tw, th = fm.horizontalAdvance(text) + 18, fm.height() + 10
            top = fy - fht * sy - th - 14
            r = QRectF(cx - tw / 2, max(2, top), tw, th)
            path = QPainterPath()
            path.addRoundedRect(r, 9, 9)
            p.setOpacity(min(1.0, left / 0.4))
            p.setPen(QPen(QColor(60, 60, 60, 160), 1.2))
            p.setBrush(QColor(255, 255, 255, 235))
            p.drawPath(path)
            p.setPen(QColor(40, 40, 40))
            p.drawText(r, Qt.AlignCenter, text)
        p.end()

    # ---------------- 滑鼠互動
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.press_pos = e.globalPosition()
            self.press_time = time.time()
            self.drag_off = QPointF(e.globalPosition().x() - self.x, e.globalPosition().y() - (self.y - self.fh))
            self.drag_samples = [(time.time(), e.globalPosition())]
        elif e.button() == Qt.RightButton:
            self.ctrl.build_menu(self).exec(e.globalPosition().toPoint())

    def mouseMoveEvent(self, e):
        g = e.globalPosition()
        if self.press_pos is not None and e.buttons() & Qt.LeftButton:
            if not self.dragging and (g - self.press_pos).manhattanLength() > 6:
                self.dragging = True
                self.state = "drag"
                self.support = None
                self.interrupt()
                self.pose = "walk"
                self.play(self.pick_loop("walk"), fade=0.1)
                self.drag_off = QPointF(0, 0)
                if random.random() < 0.5:
                    self.say(random.choice(["放我下來！", "咦？", "喵！？"]))
            if self.dragging:
                prev = self.drag_samples[-1][1]
                self.drag_rot = max(-35, min(35, self.drag_rot * 0.7 - (g.x() - prev.x()) * 0.9))
                self.drag_samples = (self.drag_samples + [(time.time(), g)])[-6:]
                self.x = g.x()
                self.y = g.y() + self.fh
                self.place()
                self.update()
        else:
            # 用滑鼠來回摸牠
            if self.last_mouse is not None and self.grounded():
                self.pet_meter += (g - self.last_mouse).manhattanLength()
                if self.pet_meter > 900:
                    self.pet_meter = 0
                    self.emit("❤", 2)
                    if self.state not in ("petted", "sleep"):
                        self.set_state("petted", 2.5)
                        self.energy = min(1, self.energy + 0.1)
                        if random.random() < 0.4:
                            self.say(random.choice(["呼嚕嚕…", "好舒服～", "❤"]))
            self.last_mouse = g

    def leaveEvent(self, e):
        self.last_mouse = None

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.dragging:
            self.dragging = False
            (t0, p0), (t1, p1) = self.drag_samples[0], self.drag_samples[-1]
            dtt = max(t1 - t0, 0.016)
            vx = max(-3000, min(3000, (p1.x() - p0.x()) / dtt))
            vy = max(-3000, min(3000, (p1.y() - p0.y()) / dtt))
            self.dir = 1 if vx >= 0 else -1
            self.start_fall(vx=vx, vy=vy)
        elif self.press_pos is not None:
            self.on_click()
        self.press_pos = None

    def mouseDoubleClickEvent(self, e):
        if self.grounded():
            self.squash = 0.3
            self.start_fall(vx=self.dir * 120, vy=-1100)
            self.emit("♪", 2, QColor(255, 170, 0))

    def on_click(self):
        if self.state == "sleep":
            self.set_state("stretch", 1.6)
            self.say(random.choice(["!?", "…幹嘛啦", "嗯…？"]))
        elif self.grounded():
            self.emit("❤", 1)
            if random.random() < 0.6:
                self.say(random.choice(self.phrases))
            self.set_state("petted", 1.5)

    def come_here(self):
        if self.grounded() and self.support and self.support.hwnd == 0:
            self.speed = 240
            self.set_state("chase", 10)
        else:
            m = QCursor.pos()
            self.x = m.x()
            self.y = m.y() - 200
            self.start_fall(vx=0)


# ======================================================================= 總控（系統匣、多隻寵物、單一執行個體）
class Controller:
    def __init__(self, app):
        self.app = app
        self.settings = load_settings()
        self.world = World(self.settings)
        self.pets: list[Pet] = []
        self.last = time.perf_counter()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.scan = QTimer()
        self.scan.timeout.connect(self.rescan)
        self.scan.start(200)
        self.raise_timer = QTimer()
        self.raise_timer.timeout.connect(lambda: [p.raise_() for p in self.pets])
        self.raise_timer.start(3000)

        self.server = QLocalServer()
        QLocalServer.removeServer(APP_KEY)
        self.server.listen(APP_KEY)
        self.server.newConnection.connect(self.on_ipc)

        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon()
            self.tray.setToolTip("桌面寵物")
            self.tray.activated.connect(lambda r: r == QSystemTrayIcon.Trigger and [p.come_here() for p in self.pets])

    # ---------------- 寵物
    def summon(self, name):
        for p in self.pets:
            if p.name == name:
                p.load()  # 重新讀取（剛餵了新照片）
                p.summon_drop()
                p.say("我變得更像我了！")
                return
        try:
            pet = Pet(self, name)
        except Exception as e:
            print("無法召喚", name, e)
            return
        self.pets.append(pet)
        if name not in self.settings["active_pets"]:
            self.settings["active_pets"].append(name)
            save_settings(self.settings)
        self.update_tray()

    def dismiss(self, pet):
        self.pets.remove(pet)
        pet.close()
        pet.deleteLater()
        if pet.name in self.settings["active_pets"] and pet.name not in [p.name for p in self.pets]:
            self.settings["active_pets"].remove(pet.name)
            save_settings(self.settings)
        if not self.pets:
            self.app.quit()
        self.update_tray()

    def update_tray(self):
        if not self.tray:
            return
        if self.pets:
            pm = self.pets[0].sprites["sit"][0].frames[0][0]
            self.tray.setIcon(QIcon(pm))
        self.tray.setContextMenu(self.build_menu(None))
        self.tray.show()

    # ---------------- 選單
    def build_menu(self, pet):
        m = QMenu()
        targets = [pet] if pet else self.pets

        def act(text, fn, menu=m, checkable=False, checked=False):
            a = QAction(text, menu)
            a.setCheckable(checkable)
            a.setChecked(checked)
            a.triggered.connect(fn)
            menu.addAction(a)
            return a

        act("叫牠過來", lambda: [p.come_here() for p in targets])
        act("讓牠睡覺", lambda: [p.set_state("sleep", 120) for p in targets if p.grounded()])
        act("叫醒", lambda: [p.set_state("stretch", 1.6) for p in targets if p.state == "sleep"])
        m.addSeparator()
        sub = m.addMenu("再召喚一隻")
        for n in list_packs():
            act(n, lambda _=False, n=n: self.summon(n), menu=sub)
        act("餵新照片（開啟訓練器）…", lambda: launch("trainer"))
        m.addSeparator()
        size = m.addMenu("大小")
        for label, v in (("小", 0.7), ("中", 1.0), ("大", 1.4), ("特大", 2.0)):
            act(label, lambda _=False, v=v: self.set_size(v), menu=size, checkable=True,
                checked=abs(self.settings.get("size", 1) - v) < 0.01)
        act("安靜模式（待在原地不亂跑）", self.toggle("quiet"), checkable=True, checked=self.settings.get("quiet", False))
        act("可以跳到視窗上", self.toggle("climb_windows"), checkable=True,
            checked=self.settings.get("climb_windows", True))
        if winenv.IS_WIN:
            act("開機自動啟動", self.toggle_autostart, checkable=True, checked=self.autostart_path().exists())
        m.addSeparator()
        if pet:
            act(f"讓 {pet.name} 先離開", lambda: self.dismiss(pet))
        act("全部結束", self.app.quit)
        return m

    def toggle(self, key):
        def f():
            self.settings[key] = not self.settings.get(key, False if key == "quiet" else True)
            save_settings(self.settings)
            self.world.refresh()
            self.update_tray()
        return f

    def set_size(self, v):
        self.settings["size"] = v
        save_settings(self.settings)
        for p in self.pets:
            p.load()
        self.update_tray()

    def autostart_path(self):
        return Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/桌面寵物.vbs"

    def toggle_autostart(self):
        f = self.autostart_path()
        if f.exists():
            f.unlink()
        else:
            cmd = " ".join(f'""{a}""' if i < 2 else a for i, a in enumerate(launch_cmd("pet")))
            f.write_text(f'CreateObject("WScript.Shell").Run "{cmd}", 0\r\n',
                         encoding="utf-16")
        self.update_tray()

    # ---------------- 迴圈
    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now
        for p in list(self.pets):
            p.tick(dt)

    def rescan(self):
        self.world.refresh()
        for p in self.pets:
            p.follow_support()

    def on_ipc(self):
        sock = self.server.nextPendingConnection()

        def read():
            msg = bytes(sock.readAll()).decode("utf-8", "ignore").strip()
            if msg.startswith("summon:"):
                self.summon(msg[7:])
            elif msg == "show":
                for p in self.pets:
                    p.come_here()
        sock.readyRead.connect(read)


def main():
    args = sys.argv[1:]
    name = args[args.index("--summon") + 1] if "--summon" in args and len(args) > args.index("--summon") + 1 else None

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 已經在跑了？把要求轉給它就好
    sock = QLocalSocket()
    sock.connectToServer(APP_KEY)
    if sock.waitForConnected(300):
        sock.write((f"summon:{name}" if name else "show").encode("utf-8"))
        sock.waitForBytesWritten(1000)
        sock.disconnectFromServer()
        return

    ctrl = Controller(app)
    names = [name] if name else (load_settings().get("active_pets") or list_packs()[:1])
    names = [n for n in names if n in list_packs()]
    if not names:
        # 還沒有寵物：直接打開訓練器
        launch("trainer")
        return
    for n in names:
        ctrl.summon(n)
    app.exec()


if __name__ == "__main__":
    main()
