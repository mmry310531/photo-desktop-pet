"""寵物訓練器：把一堆寵物照片「餵」進來，變成桌面寵物的姿勢庫。

GUI：python trainer.py
CLI：python trainer.py --cli 寵物名字 照片資料夾 [更多檔案或資料夾...] [--hq]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import POSES, ROOT, list_packs, load_pack, pack_dir, pythonw, safe_name, save_pack  # noqa: E402


# ---------------------------------------------------------------- 核心（GUI/CLI 共用）
def save_results(name: str, results, pack: dict) -> dict:
    d = pack_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    used = {im["file"] for im in pack["images"]}
    n = len(pack["images"])
    for r in results:
        n += 1
        fname = f"{int(time.time())}_{n:03d}.png"
        while fname in used:
            n += 1
            fname = f"{int(time.time())}_{n:03d}.png"
        used.add(fname)
        r.image.save(d / fname, optimize=True)
        pack["images"].append({
            "file": fname, "pose": r.pose, "facing": r.facing,
            "area": r.area, "hash": str(r.hash), "source": Path(r.source).name,
        })
    pack["name"] = name
    save_pack(name, pack)
    return pack


def run_cli(argv: list[str]) -> None:
    from cutout import collect_images, get_detector, get_session, process

    hq = "--hq" in argv
    argv = [a for a in argv if a != "--hq"]
    if len(argv) < 2:
        print("用法：python trainer.py --cli 寵物名字 照片資料夾 [...] [--hq]")
        sys.exit(1)
    name, paths = safe_name(argv[0]), argv[1:]
    pack = load_pack(name)
    files = collect_images(paths)
    print(f"找到 {len(files)} 張照片，載入 AI 去背模型（第一次會下載約 170MB）…")
    sess = get_session(hq)
    det = get_detector(print)
    hashes = [int(im["hash"]) for im in pack["images"] if im.get("hash")]
    good = []
    for i, f in enumerate(files, 1):
        r = process(f, sess, hashes, det)
        tag = r.error or f"{r.extra.get('detected', '')} {POSES[r.pose]}，面向{'左' if r.facing == 'left' else '右'} {r.warning}"
        print(f"[{i}/{len(files)}] {f.name}: {tag}")
        if r.image is not None:
            good.append(r)
            hashes.append(r.hash)
    pack = save_results(name, good, pack)
    print(f"完成！{name} 現在有 {len(pack['images'])} 個姿勢，存在 {pack_dir(name)}")


# ---------------------------------------------------------------- GUI
def run_gui() -> None:
    from PySide6.QtCore import QSize, Qt, QThread, Signal
    from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListView,
        QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton,
        QVBoxLayout, QWidget,
    )

    def pil_to_icon(im) -> QIcon:
        im = im.convert("RGBA")
        qi = QImage(im.tobytes(), im.width, im.height, im.width * 4, QImage.Format_RGBA8888).copy()
        return QIcon(QPixmap.fromImage(qi))

    class Worker(QThread):
        progress = Signal(int, int, str)
        result = Signal(object)
        finished_all = Signal(int, int)

        def __init__(self, files, hq, hashes):
            super().__init__()
            self.files, self.hq, self.hashes = files, hq, list(hashes)
            self.stop = False

        def run(self):
            from cutout import get_detector, get_session, process

            try:
                self.progress.emit(0, len(self.files), "載入 AI 去背模型（第一次會下載約 170MB，請稍候）…")
                sess = get_session(self.hq)
                det = get_detector(lambda m: self.progress.emit(0, len(self.files), m))
            except Exception as e:
                self.progress.emit(0, len(self.files), f"模型下載失敗，請檢查網路後再試：{e}")
                self.finished_all.emit(0, len(self.files))
                return
            ok = bad = 0
            for i, f in enumerate(self.files, 1):
                if self.stop:
                    break
                self.progress.emit(i - 1, len(self.files), f"處理中：{f.name}")
                r = process(f, sess, self.hashes, det)
                if r.image is not None:
                    ok += 1
                    self.hashes.append(r.hash)
                else:
                    bad += 1
                self.result.emit(r)
            self.progress.emit(len(self.files), len(self.files), "完成")
            self.finished_all.emit(ok, bad)

    class Trainer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("寵物訓練器 — 餵照片給牠")
            self.resize(980, 720)
            self.setAcceptDrops(True)
            self.pending: list[Path] = []
            self.results = []  # 新處理好的 CutResult
            self.worker = None

            lay = QVBoxLayout(self)
            top = QHBoxLayout()
            top.addWidget(QLabel("寵物名字："))
            self.name = QComboBox()
            self.name.setEditable(True)
            self.name.addItems(list_packs() or ["我的寵物"])
            self.name.setMinimumWidth(200)
            self.name.currentTextChanged.connect(self.load_existing)
            top.addWidget(self.name)
            self.hq = QCheckBox("高品質毛邊（較慢，模型約 900MB）")
            top.addWidget(self.hq)
            top.addStretch()
            lay.addLayout(top)

            self.hint = QLabel("① 把寵物照片（或整個資料夾）拖進這個視窗，或按下方按鈕加入。"
                               "照片越多、姿勢越多樣（坐、站、躺），牠在桌面上的動作就越豐富。\n"
                               "② 按「開始訓練」：AI 會自動去背、判斷姿勢。③ 檢查結果，右鍵可修正姿勢/面向/刪除。④ 按「儲存並召喚」。")
            self.hint.setWordWrap(True)
            lay.addWidget(self.hint)

            btns = QHBoxLayout()
            b1 = QPushButton("加入照片…"); b1.clicked.connect(self.add_files)
            b2 = QPushButton("加入資料夾…"); b2.clicked.connect(self.add_folder)
            self.go = QPushButton("開始訓練 ▶"); self.go.clicked.connect(self.start)
            self.pending_lbl = QLabel("待處理：0 張")
            for b in (b1, b2, self.go):
                btns.addWidget(b)
            btns.addWidget(self.pending_lbl)
            btns.addStretch()
            lay.addLayout(btns)

            self.bar = QProgressBar(); self.status = QLabel("")
            lay.addWidget(self.bar); lay.addWidget(self.status)

            self.grid = QListWidget()
            self.grid.setViewMode(QListView.IconMode)
            self.grid.setIconSize(QSize(150, 150))
            self.grid.setGridSize(QSize(175, 200))
            self.grid.setResizeMode(QListView.Adjust)
            self.grid.setMovement(QListView.Static)
            self.grid.setSelectionMode(QListWidget.ExtendedSelection)
            self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
            self.grid.customContextMenuRequested.connect(self.menu)
            lay.addWidget(self.grid, 1)

            bottom = QHBoxLayout()
            self.count = QLabel("")
            bottom.addWidget(self.count)
            bottom.addStretch()
            save = QPushButton("儲存並召喚到桌面 🐾"); save.clicked.connect(self.save_and_launch)
            save.setMinimumHeight(36)
            bottom.addWidget(save)
            lay.addLayout(bottom)
            self.load_existing(self.name.currentText())

        # ---- 拖放
        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()

        def dropEvent(self, e):
            self.enqueue([u.toLocalFile() for u in e.mimeData().urls()])

        def add_files(self):
            fs, _ = QFileDialog.getOpenFileNames(self, "選擇寵物照片", "",
                                                 "圖片 (*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp)")
            self.enqueue(fs)

        def add_folder(self):
            d = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
            if d:
                self.enqueue([d])

        def enqueue(self, paths):
            from cutout import collect_images

            new = collect_images(paths)
            have = {str(p) for p in self.pending}
            self.pending += [p for p in new if str(p) not in have]
            self.pending_lbl.setText(f"待處理：{len(self.pending)} 張")

        # ---- 既有寵物包
        def load_existing(self, name):
            self.grid.clear()
            pack = load_pack(safe_name(name))
            d = pack_dir(name)
            for im in pack["images"]:
                it = QListWidgetItem(QIcon(str(d / im["file"])), "")
                it.setData(Qt.UserRole, ("old", im))
                self.grid.addItem(it)
                self.refresh_label(it)
            for r in self.results:  # 還沒儲存的新結果保留（例如訓練完才改名字）
                it = QListWidgetItem(pil_to_icon(r.image), "")
                it.setData(Qt.UserRole, ("new", r))
                self.grid.addItem(it)
                self.refresh_label(it)
            self.update_count()

        def refresh_label(self, it):
            kind, obj = it.data(Qt.UserRole)
            pose = obj["pose"] if kind == "old" else obj.pose
            facing = obj["facing"] if kind == "old" else obj.facing
            warn = "" if kind == "old" else (" ⚠" if obj.warning else "")
            it.setText(f"{POSES[pose]} · 頭朝{'←' if facing == 'left' else '→'}{warn}"
                       + ("" if kind == "old" else " 🆕"))
            if kind == "new" and obj.warning:
                it.setToolTip(obj.warning)

        def update_count(self):
            c = {k: 0 for k in POSES}
            for i in range(self.grid.count()):
                kind, o = self.grid.item(i).data(Qt.UserRole)
                c[o["pose"] if kind == "old" else o.pose] += 1
            self.count.setText("　".join(f"{POSES[k]}：{v}" for k, v in c.items())
                               + ("　（缺「躺」的話，牠睡覺時會用坐姿）" if c["lie"] == 0 else ""))

        # ---- 訓練
        def start(self):
            if not self.pending:
                QMessageBox.information(self, "沒有照片", "先把寵物照片拖進來，或按「加入照片」。")
                return
            if self.worker and self.worker.isRunning():
                return
            pack = load_pack(safe_name(self.name.currentText()))
            hashes = [int(im["hash"]) for im in pack["images"] if im.get("hash")]
            hashes += [r.hash for r in self.results]
            self.worker = Worker(self.pending, self.hq.isChecked(), hashes)
            self.pending = []
            self.pending_lbl.setText("待處理：0 張")
            self.worker.progress.connect(self.on_progress)
            self.worker.result.connect(self.on_result)
            self.worker.finished_all.connect(self.on_done)
            self.go.setEnabled(False)
            self.worker.start()

        def on_progress(self, i, n, msg):
            self.bar.setMaximum(max(n, 1)); self.bar.setValue(i); self.status.setText(msg)

        def on_result(self, r):
            if r.image is None:
                self.status.setText(f"{Path(r.source).name}：{r.error}")
                return
            self.results.append(r)
            it = QListWidgetItem(pil_to_icon(r.image), "")
            it.setData(Qt.UserRole, ("new", r))
            self.grid.addItem(it)
            self.refresh_label(it)
            self.grid.scrollToItem(it)
            self.update_count()

        def on_done(self, ok, bad):
            self.go.setEnabled(True)
            self.status.setText(f"完成：成功 {ok} 張" + (f"，略過 {bad} 張（重複或找不到主體）" if bad else "")
                                + "。檢查一下，再按「儲存並召喚」。")

        # ---- 修正
        def menu(self, pos):
            items = self.grid.selectedItems()
            if not items:
                return
            m = QMenu(self)
            for k, v in POSES.items():
                a = QAction(f"改成：{v}", m)
                a.triggered.connect(lambda _=False, k=k: self.set_pose(items, k))
                m.addAction(a)
            m.addSeparator()
            a = QAction("左右翻轉面向（牠倒退走時用）", m); a.triggered.connect(lambda: self.flip(items)); m.addAction(a)
            a = QAction("刪除", m); a.triggered.connect(lambda: self.delete(items)); m.addAction(a)
            m.exec(self.grid.mapToGlobal(pos))

        def set_pose(self, items, k):
            for it in items:
                kind, o = it.data(Qt.UserRole)
                if kind == "old":
                    o["pose"] = k
                else:
                    o.pose = k
                self.refresh_label(it)
            self.update_count()

        def flip(self, items):
            for it in items:
                kind, o = it.data(Qt.UserRole)
                if kind == "old":
                    o["facing"] = "left" if o["facing"] == "right" else "right"
                else:
                    o.facing = "left" if o.facing == "right" else "right"
                self.refresh_label(it)

        def delete(self, items):
            for it in items:
                kind, o = it.data(Qt.UserRole)
                if kind == "old":
                    o["_deleted"] = True
                else:
                    self.results.remove(o)
                self.grid.takeItem(self.grid.row(it))
            self.update_count()

        # ---- 儲存
        def save_and_launch(self):
            name = safe_name(self.name.currentText())
            pack = load_pack(name)
            # 套用對舊圖的修改
            old = {}
            for i in range(self.grid.count()):
                kind, o = self.grid.item(i).data(Qt.UserRole)
                if kind == "old":
                    old[o["file"]] = o
            kept = []
            for im in pack["images"]:
                o = old.get(im["file"])
                if o is None:  # 被刪掉
                    try:
                        (pack_dir(name) / im["file"]).unlink()
                    except OSError:
                        pass
                    continue
                kept.append({k: v for k, v in o.items() if not k.startswith("_")})
            pack["images"] = kept
            pack = save_results(name, self.results, pack)
            self.results = []
            if not pack["images"]:
                QMessageBox.warning(self, "還沒有照片", "這隻寵物還沒有任何姿勢，先餵幾張照片吧。")
                return
            self.load_existing(name)
            subprocess.Popen([pythonw(), str(ROOT / "app" / "pet.py"), "--summon", name], cwd=str(ROOT))
            self.status.setText(f"已儲存 {len(pack['images'])} 個姿勢，{name} 跑到你的桌面上了！"
                                "（之後可以繼續餵新照片，再按一次儲存就會更新）")

    app = QApplication(sys.argv)
    w = Trainer()
    w.show()
    app.exec()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli(sys.argv[2:])
    else:
        run_gui()
