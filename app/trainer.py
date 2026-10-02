"""寵物訓練器：把一堆寵物照片「餵」進來，變成桌面寵物的姿勢庫。

GUI：python trainer.py
CLI：python trainer.py --cli 寵物名字 照片資料夾 [更多檔案或資料夾...] [--hq]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (POSES, PACK_EXT, export_pack, import_pack, launch, list_packs, load_pack,  # noqa: E402
                    pack_dir, safe_name, save_pack)


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
        entry = {"file": fname, "pose": r.pose, "facing": r.facing,
                 "area": r.area, "hash": str(r.hash), "source": Path(r.source).name}
        for k in ("score", "clip"):
            if k in r.extra:
                entry[k] = r.extra[k]
        det = (r.extra.get("detected") or "").split(" ")[0]
        if det:
            from aifill import SPECIES

            entry["species"] = SPECIES.get(det, "pet")
        if r.frames:  # 影片 -> 連續動畫：每一格各存一張
            stem = fname[:-4]
            entry["fps"] = r.fps
            entry["kind"] = r.extra.get("kind", "loop")
            if entry["kind"] == "trans":
                entry["to"] = r.extra["to"]
            if r.extra.get("ai"):
                entry["ai"] = True
            entry["frames"] = []
            for k, (im, ax) in enumerate(r.frames):
                ff = f"{stem}_f{k:03d}.png"
                im.save(d / ff, optimize=True)
                entry["frames"].append({"file": ff, "ax": round(ax, 1)})
            entry["file"] = entry["frames"][0]["file"]
        else:
            r.image.save(d / fname, optimize=True)
        pack["images"].append(entry)
    pack["name"] = name
    save_pack(name, pack)
    return pack


def run_cli(argv: list[str]) -> None:
    from cutout import collect_images, get_detector, get_session, process_any

    hq = "--hq" in argv
    argv = [a for a in argv if a != "--hq"]
    if len(argv) < 2:
        print("用法：python trainer.py --cli 寵物名字 照片資料夾 [...] [--hq]")
        sys.exit(1)
    name, paths = safe_name(argv[0]), argv[1:]
    pack = load_pack(name)
    files = collect_images(paths)
    print(f"找到 {len(files)} 個照片／影片，載入 AI 模型…")
    sess = get_session(hq)
    det = get_detector(print)
    try:
        from cutout import get_clip

        get_clip(print)
    except Exception as e:
        print("姿勢辨識模型無法使用，改用形狀判斷：", e)
    hashes = [int(im["hash"]) for im in pack["images"] if im.get("hash")]
    good = []
    for i, f in enumerate(files, 1):
        r = process_any(f, sess, hashes, det, lambda m: print("  " + m, end="\r"))
        clip = ""
        if r.frames:
            k = r.extra.get("kind")
            clip = f"🎞 {len(r.frames)} 格{'轉場→' + POSES[r.extra['to']] if k == 'trans' else '循環'} "
        tag = r.error or f"{clip}{r.extra.get('detected', '')} {POSES[r.pose]}，面向{'左' if r.facing == 'left' else '右'} {r.warning}"
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
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
        QLineEdit, QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton,
        QRadioButton, QVBoxLayout, QWidget,
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
            self.fatal = ""

        def run(self):
            from cutout import get_detector, get_session, process_any

            try:
                self.progress.emit(0, len(self.files), "載入 AI 模型…（第一次會下載，畫面停住是正常的）")
                sess = get_session(self.hq, lambda m: self.progress.emit(0, len(self.files), m))
                det = get_detector(lambda m: self.progress.emit(0, len(self.files), m))
            except Exception as e:
                self.fatal = f"AI 模型還沒下載好，照片都還沒處理（已下載的部分會保留，再按一次會接著下載）：{e}"
                self.progress.emit(0, len(self.files), self.fatal)
                self.finished_all.emit(0, 0)
                return
            try:  # 姿勢辨識模型是加分項，下載失敗就用形狀判斷，不影響其他功能
                from cutout import get_clip

                get_clip(lambda m: self.progress.emit(0, len(self.files), m))
            except Exception:
                pass
            ok = bad = 0
            for i, f in enumerate(self.files, 1):
                if self.stop:
                    break
                self.progress.emit(i - 1, len(self.files), f"處理中：{f.name}")
                r = process_any(f, sess, self.hashes, det,
                                lambda m, i=i: self.progress.emit(i - 1, len(self.files), m))
                if r.image is not None:
                    ok += 1
                    self.hashes.append(r.hash)
                else:
                    bad += 1
                self.result.emit(r)
            self.progress.emit(len(self.files), len(self.files), "完成")
            self.finished_all.emit(ok, bad)

    class AIWorker(QThread):
        progress = Signal(int, int, str)
        result = Signal(object)
        finished_all = Signal(int, int)

        def __init__(self, name, jobs, backend):
            super().__init__()
            self.name, self.jobs, self.backend = name, jobs, backend
            self.last_error = ""

        def run(self):
            import realclips
            from cutout import get_session

            n = len(self.jobs)
            ok = bad = 0
            try:
                sess = get_session(False, lambda m: self.progress.emit(0, n, m))
            except Exception as e:
                self.last_error = f"模型下載中斷（已下載的部分會保留，再按一次會接著下載）：{e}"
                self.finished_all.emit(0, n)
                return
            animal = "cat"
            try:
                import aifill
                from common import load_pack

                animal = aifill._animal(load_pack(self.name))
            except Exception:
                pass
            for i, j in enumerate(self.jobs):
                say = (lambda m, i=i, j=j: self.progress.emit(i, n, f"[{i + 1}/{n}] {m}"))
                try:
                    realclips.generate(self.name, self.backend, sess, which=[j.key], status=say, animal=animal)
                    ok += 1
                    self.result.emit("reload")
                except Exception as e:
                    bad += 1
                    self.last_error = f"{j.label} 失敗：{e}"
                    self.progress.emit(i, n, self.last_error)
                    if "額度" in str(e):  # 免費額度用完，後面的也不用試了
                        bad += n - i - 1
                        break
            self.progress.emit(n, n, "完成")
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
            self.hq = QCheckBox("高品質毛邊（要另外下載約 900MB 的模型，處理也慢很多；一般不需要）")
            top.addWidget(self.hq)
            top.addStretch()
            lay.addLayout(top)

            import spec

            self.hint = QLabel("把照片、影片（或整個資料夾）拖進這個視窗 → 按「一鍵產出」，"
                               "就會自動去背、判斷姿勢、補齊動作、放到桌面。準備越多，牠越像活的：<br>"
                               + spec.SPEC_TEXT)
            self.hint.setWordWrap(True)
            lay.addWidget(self.hint)
            self.spec_lbl = QLabel()
            self.spec_lbl.setWordWrap(True)
            self.spec_lbl.setStyleSheet("QLabel{background:palette(base);border:1px solid palette(mid);"
                                        "border-radius:6px;padding:6px}")
            lay.addWidget(self.spec_lbl)
            opts = QHBoxLayout()
            from common import load_settings

            st = load_settings()
            self.auto_ai = QCheckBox("缺的動作用 AI 補（預設免費 Hugging Face；按「🪄 AI 設定」可換 fal.ai）")
            self.auto_ai.setChecked(st.get("auto_ai", True))
            self.auto_launch = QCheckBox("完成後直接放到桌面")
            self.auto_launch.setChecked(True)
            opts.addWidget(self.auto_ai)
            opts.addWidget(self.auto_launch)
            opts.addStretch()
            lay.addLayout(opts)

            btns = QHBoxLayout()
            b1 = QPushButton("加入照片…"); b1.clicked.connect(self.add_files)
            b2 = QPushButton("加入資料夾…"); b2.clicked.connect(self.add_folder)
            self.go = QPushButton("一鍵產出 ▶"); self.go.clicked.connect(self.start)
            bimp = QPushButton("匯入寵物包…"); bimp.clicked.connect(self.import_dialog)
            bexp = QPushButton("匯出這隻…"); bexp.clicked.connect(self.export_dialog)
            bimp.setToolTip("朋友分享的 .petpack 檔（也可以直接拖進視窗）")
            bexp.setToolTip("存成 .petpack 檔，傳給朋友就能直接用，不用再訓練")
            self.pending_lbl = QLabel("待處理：0 張")
            for b in (b1, b2, self.go, bimp, bexp):
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
            aib = QPushButton("🪄 AI 設定／手動補…"); aib.clicked.connect(self.ai_fill)
            aib.setToolTip("沒拍到的動作（走路、坐下、趴下…）用 AI 從照片生成")
            bottom.addWidget(aib)
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
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            packs = [p for p in paths if p.lower().endswith(PACK_EXT)]
            for p in packs:
                self.do_import(p)
            self.enqueue([p for p in paths if p not in packs])

        def import_dialog(self):
            fs, _ = QFileDialog.getOpenFileNames(self, "選擇寵物包", "", f"寵物包 (*{PACK_EXT})")
            for f in fs:
                self.do_import(f)

        def do_import(self, path):
            try:
                name = import_pack(path)
            except Exception as e:
                QMessageBox.warning(self, "匯入失敗", f"{Path(path).name}：{e}")
                return
            if self.name.findText(name) < 0:
                self.name.addItem(name)
            self.name.setCurrentText(name)
            self.status.setText(f"已匯入「{name}」，按「儲存並召喚」就能放到桌面上。")

        def export_dialog(self):
            name = self.save_only()
            if not name:
                return
            f, _ = QFileDialog.getSaveFileName(self, "匯出寵物包", str(Path.home() / f"{name}{PACK_EXT}"),
                                               f"寵物包 (*{PACK_EXT})")
            if f:
                if not f.lower().endswith(PACK_EXT):
                    f += PACK_EXT
                export_pack(name, f)
                self.status.setText(f"已匯出到 {f}（{Path(f).stat().st_size / 1e6:.1f} MB），傳給朋友，他用「匯入寵物包」就能直接用。")

        def add_files(self):
            fs, _ = QFileDialog.getOpenFileNames(self, "選擇寵物照片或影片", str(Path.home() / "Pictures"),
                                                 "照片或影片 (*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp *.mp4 *.mov *.m4v *.avi *.mkv *.webm *.gif)")
            self.enqueue(fs)

        def add_folder(self):
            d = QFileDialog.getExistingDirectory(self, "選擇照片資料夾", str(Path.home() / "Pictures"))
            if d:
                self.enqueue([d])

        def enqueue(self, paths):
            from cutout import collect_images

            new = collect_images(paths)
            if paths and not new:
                QMessageBox.information(
                    self, "沒有找到照片或影片",
                    "選的位置裡沒有支援的檔案。\n支援：JPG、PNG、HEIC、WEBP、AVIF、JFIF、MP4、MOV 等。\n"
                    "（OneDrive／iCloud 只在雲端的檔案請先下載到電腦）")
                return
            have = {str(p) for p in self.pending}
            add = [p for p in new if str(p) not in have]
            self.pending += add
            self.pending_lbl.setText(f"待處理：{len(self.pending)} 個")
            if add:
                self.status.setText(f"加入 {len(add)} 個照片／影片，自動開始處理…")
                if not (self.worker and self.worker.isRunning()):
                    self.process_pending(finish=False)

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
            nframes = len(obj.get("frames", [])) if kind == "old" else len(obj.frames)
            ckind = obj.get("kind") if kind == "old" else obj.extra.get("kind")
            to = obj.get("to") if kind == "old" else obj.extra.get("to")
            clip = f"🎞{nframes}格 " if nframes else ""
            posetxt = f"{POSES[pose]}→{POSES[to]}" if ckind == "trans" and to else POSES[pose]
            it.setText(f"{clip}{posetxt} · 頭朝{'←' if facing == 'left' else '→'}{warn}"
                       + ("" if kind == "old" else " 🆕"))
            if kind == "new" and obj.warning:
                it.setToolTip(obj.warning)

        def grid_entries(self):
            out = []
            for i in range(self.grid.count()):
                kind, o = self.grid.item(i).data(Qt.UserRole)
                if kind == "old":
                    out.append({"pose": o["pose"], "kind": o.get("kind") or ("loop" if o.get("frames") else None),
                                "to": o.get("to"), "animated": bool(o.get("frames"))})
                else:
                    out.append({"pose": o.pose, "kind": o.extra.get("kind"), "to": o.extra.get("to"),
                                "animated": bool(o.frames)})
            return out

        def update_count(self):
            import spec

            self.spec_lbl.setText(spec.html_table(spec.coverage(self.grid_entries())))
            c = {k: 0 for k in POSES}
            for i in range(self.grid.count()):
                kind, o = self.grid.item(i).data(Qt.UserRole)
                c[o["pose"] if kind == "old" else o.pose] += 1
            self.count.setText("　".join(f"{POSES[k]}：{v}" for k, v in c.items())
                               + ("　（缺「躺」的話，牠睡覺時會用坐姿）" if c["lie"] == 0 else ""))

        # ---- 訓練
        def start(self):
            """一鍵產出：處理還沒處理的素材 → 存檔 → AI 補 → 放到桌面。"""
            if self.worker and self.worker.isRunning():
                self.finish_after = True
                self.status.setText("素材還在處理中，處理完會自動接著產出…")
                return
            if self.pending:
                self.process_pending(finish=True)
                return
            if self.grid.count():  # 沒有新素材：直接進下一步（補 AI、存檔、放桌面）
                self.on_done(0, 0)
                return
            QMessageBox.information(self, "沒有素材", "先把寵物照片或影片拖進來，或按「加入照片」。")

        def process_pending(self, finish):
            """加入素材後自動開始去背、判斷姿勢；finish=True 時處理完接著一鍵產出。"""
            self.finish_after = finish
            self.batch_ok = getattr(self, "batch_ok", 0)
            self.batch_bad = getattr(self, "batch_bad", 0)
            pack = load_pack(safe_name(self.name.currentText()))
            hashes = [int(im["hash"]) for im in pack["images"] if im.get("hash")]
            hashes += [r.hash for r in self.results]
            self.worker = Worker(self.pending, self.hq.isChecked(), hashes)
            self.pending = []
            self.pending_lbl.setText("待處理：0 個")
            self.worker.progress.connect(self.on_progress)
            self.worker.result.connect(self.on_result)
            self.worker.finished_all.connect(self.on_batch_done)
            self.worker.start()

        def on_batch_done(self, ok, bad):
            if self.worker.fatal:  # 模型沒載入：素材放回待處理，不要假裝處理過
                self.pending = list(self.worker.files) + self.pending
                self.pending_lbl.setText(f"待處理：{len(self.pending)} 個")
                self.batch_ok = self.batch_bad = 0
                self.status.setText(self.worker.fatal)
                return
            self.batch_ok += ok
            self.batch_bad += bad
            if self.pending:  # 處理中又加了新的
                self.process_pending(self.finish_after)
                return
            ok, bad = self.batch_ok, self.batch_bad
            self.batch_ok = self.batch_bad = 0
            if self.finish_after:
                self.on_done(ok, bad)
            else:
                self.status.setText(f"處理好 {ok} 個" + (f"，略過 {bad} 個（重複、太模糊或找不到寵物）" if bad else "")
                                    + "。可以繼續加素材，或按「一鍵產出 ▶」完成並放到桌面。")

        def on_progress(self, i, n, msg):
            self.bar.setMaximum(max(n, 1)); self.bar.setValue(i); self.status.setText(msg)

        def on_result(self, r):
            if isinstance(r, str) and r == "reload":  # 寫實片段已直接存進寵物包
                self.load_existing(safe_name(self.name.currentText()))
                return
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
            """素材處理完 → 存檔 →（缺的用 AI 補）→ 放到桌面，一路做完。"""
            from common import load_settings, save_settings

            self.go.setEnabled(True)
            summary = (f"處理完成：成功 {ok} 個" + (f"，略過 {bad} 個（重複、太模糊或找不到寵物）" if bad else "")) if ok or bad else ""
            st = load_settings()
            st["auto_ai"] = self.auto_ai.isChecked()
            save_settings(st)
            name = self.save_only()
            if not name:
                return
            if self.auto_ai.isChecked():
                import aifill

                import realclips

                jobs = realclips.missing(name)
                if jobs:
                    if st.get("ai_backend") == "fal" and st.get("fal_key"):
                        backend = aifill.Fal(st["fal_key"])
                    else:
                        backend = aifill.HuggingFace(st.get("hf_token", ""), st.get("hf_space", ""))
                    self.status.setText(summary + f"　接著用 AI 補 {len(jobs)} 段動作…")
                    self.start_ai(name, jobs, backend)
                    return
            self.finish(name, summary)

        def finish(self, name, summary=""):
            import spec

            lv = spec.level(spec.coverage(self.grid_entries()))
            msg = f"{summary}　完成度：{spec.LEVELS[lv][0]}。"
            if self.auto_launch.isChecked():
                launch("pet", "--summon", name)
                msg += f"{name} 已經放到你的桌面上了！"
            else:
                msg += "按「儲存並召喚」就能放到桌面。"
            self.status.setText(msg + "（不滿意的可以右鍵刪除／修正，再按一次一鍵產出）")

        def start_ai(self, name, jobs, backend):
            self.worker = AIWorker(name, jobs, backend)
            self.worker.progress.connect(self.on_progress)
            self.worker.result.connect(self.on_result)
            self.worker.finished_all.connect(self.on_ai_done)
            self.go.setEnabled(False)
            self.worker.start()

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
            name = self.save_only()
            if not name:
                return
            launch("pet", "--summon", name)
            self.status.setText(f"已儲存，{name} 跑到你的桌面上了！"
                                "（之後可以繼續餵新照片或影片，再按一次儲存就會更新）")

        def save_only(self):
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
                        for fn in [im["file"]] + [fr["file"] for fr in im.get("frames", [])]:
                            (pack_dir(name) / fn).unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                kept.append({k: v for k, v in o.items() if not k.startswith("_")})
            pack["images"] = kept
            pack = save_results(name, self.results, pack)
            self.results = []
            if not pack["images"]:
                QMessageBox.warning(self, "還沒有照片", "這隻寵物還沒有任何姿勢，先餵幾張照片吧。")
                return None
            self.load_existing(name)
            return name

        # ---- AI 補齊動作
        def ai_fill(self):
            import aifill
            from common import load_settings, save_settings

            if self.worker and self.worker.isRunning():
                return
            name = self.save_only()  # 先存檔，才知道還缺什麼
            if not name:
                return
            import realclips

            jobs = realclips.missing(name)
            if not jobs:
                QMessageBox.information(self, "動作都齊了", "寫實動作影片都已經做好了（或缺少趴姿／坐姿照片當起點）。")
                return
            st = load_settings()
            dlg = QDialog(self)
            dlg.setWindowTitle("🎬 AI 寫實動作")
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("AI 會從你最好的一張趴姿、一張坐姿照片出發，生成連續的真實動作影片，\n"
                               "每段的開頭結尾都接在同一張照片上，切換動作時不會跳。還缺這些（勾選要生成的）："))
            checks = []
            for j in jobs:
                c = QCheckBox(j.label)
                c.setChecked(True)
                v.addWidget(c)
                checks.append((c, j))
            v.addWidget(QLabel("<b>用哪個 AI？</b>"))
            hf = QRadioButton("Hugging Face（免費，但每天只夠做 1～2 段；用完隔天再繼續）")
            fal = QRadioButton("fal.ai（付費，最快最穩）")
            (fal if st.get("ai_backend") == "fal" else hf).setChecked(True)
            v.addWidget(hf)
            hf_tok = QLineEdit(st.get("hf_token", ""))
            hf_tok.setPlaceholderText("Hugging Face token（可留空；登入後的額度比較多）")
            hf_tok.setEchoMode(QLineEdit.Password)
            v.addWidget(hf_tok)
            l1 = QLabel('<a href="https://huggingface.co/settings/tokens">免費註冊並取得 token（選 Read 權限即可）</a>')
            l1.setOpenExternalLinks(True)
            v.addWidget(l1)
            v.addWidget(fal)
            fal_key = QLineEdit(st.get("fal_key", ""))
            fal_key.setPlaceholderText("fal.ai API key")
            fal_key.setEchoMode(QLineEdit.Password)
            v.addWidget(fal_key)
            cost = QLabel()
            l2 = QLabel('<a href="https://fal.ai/dashboard/keys">取得 fal.ai API key（需儲值）</a>')
            l2.setOpenExternalLinks(True)
            v.addWidget(l2)
            v.addWidget(cost)

            def upd():
                n = sum(c.isChecked() for c, _ in checks)
                secs = sum(realclips.CLIPS[j.key][2] for c, j in checks if c.isChecked())
                cost.setText(f"預估費用：約 US${secs * aifill.FAL_PRICE_PER_SEC:.2f}（共 {secs:.0f} 秒影片 × US${aifill.FAL_PRICE_PER_SEC}/秒）"
                             if fal.isChecked() else f"共 {n} 段，免費（每天約 1～2 段，用完隔天按一次會接著做）")
            for c, _ in checks:
                c.toggled.connect(upd)
            fal.toggled.connect(upd)
            upd()
            v.addWidget(QLabel("<small>金鑰只存在你電腦的 settings.json，不會上傳到其他地方。"
                               "AI 生成的動作偶爾會走樣，生成後一樣可以在這裡檢查、刪除。</small>"))
            bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)
            if dlg.exec() != QDialog.Accepted:
                return
            chosen = [j for c, j in checks if c.isChecked()]
            if not chosen:
                return
            st["ai_backend"] = "fal" if fal.isChecked() else "hf"
            st["hf_token"], st["fal_key"] = hf_tok.text().strip(), fal_key.text().strip()
            save_settings(st)
            if fal.isChecked():
                if not st["fal_key"]:
                    QMessageBox.warning(self, "缺少 API key", "請先填入 fal.ai 的 API key。")
                    return
                backend = aifill.Fal(st["fal_key"])
            else:
                backend = aifill.HuggingFace(st["hf_token"], st.get("hf_space", ""))
            self.start_ai(name, chosen, backend)

        def on_ai_done(self, ok, bad):
            self.go.setEnabled(True)
            name = self.save_only()
            msg = f"AI 補了 {ok} 段" + (f"，{bad} 段沒成功（{self.worker.last_error}）" if bad else "")
            if name:
                self.finish(name, msg)

    app = QApplication(sys.argv)
    w = Trainer()
    w.show()
    app.exec()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli(sys.argv[2:])
    else:
        run_gui()
