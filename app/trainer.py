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
        entry = {"file": fname, "pose": r.pose, "facing": r.facing,
                 "area": r.area, "hash": str(r.hash), "source": Path(r.source).name}
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

        def run(self):
            from cutout import get_detector, get_session, process_any

            try:
                self.progress.emit(0, len(self.files), "載入 AI 模型…（第一次會下載，畫面停住是正常的）")
                sess = get_session(self.hq, lambda m: self.progress.emit(0, len(self.files), m))
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
            import aifill
            from cutout import get_detector, get_session

            n = len(self.jobs)
            ok = bad = 0
            try:
                sess = get_session(False, lambda m: self.progress.emit(0, n, m))
                det = get_detector(lambda m: self.progress.emit(0, n, m))
            except Exception as e:
                self.last_error = f"模型下載失敗：{e}"
                self.finished_all.emit(0, n)
                return
            for i, j in enumerate(self.jobs):
                say = (lambda m, i=i, j=j: self.progress.emit(i, n, f"[{i + 1}/{n}] {j.label}：{m}"))
                try:
                    r = aifill.run_job(self.name, j, self.backend, sess, det, say)
                    if r.image is None:
                        raise RuntimeError(r.error)
                    ok += 1
                    self.result.emit(r)
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
            self.hq = QCheckBox("高品質毛邊（較慢，模型約 900MB）")
            top.addWidget(self.hq)
            top.addStretch()
            lay.addLayout(top)

            self.hint = QLabel("① 把寵物照片、影片（或整個資料夾）拖進這個視窗，或按下方按鈕加入。"
                               "姿勢越多樣（坐、站、躺）越好；🎞 側面走路 3～5 秒的影片會變成真的會動腳的連續動畫。\n"
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
            aib = QPushButton("🪄 AI 補齊動作…"); aib.clicked.connect(self.ai_fill)
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
            self.enqueue([u.toLocalFile() for u in e.mimeData().urls()])

        def add_files(self):
            fs, _ = QFileDialog.getOpenFileNames(self, "選擇寵物照片或影片", "",
                                                 "照片或影片 (*.jpg *.jpeg *.png *.webp *.heic *.heif *.bmp *.mp4 *.mov *.m4v *.avi *.mkv *.webm *.gif)")
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
            nframes = len(obj.get("frames", [])) if kind == "old" else len(obj.frames)
            ckind = obj.get("kind") if kind == "old" else obj.extra.get("kind")
            to = obj.get("to") if kind == "old" else obj.extra.get("to")
            clip = f"🎞{nframes}格 " if nframes else ""
            posetxt = f"{POSES[pose]}→{POSES[to]}" if ckind == "trans" and to else POSES[pose]
            it.setText(f"{clip}{posetxt} · 頭朝{'←' if facing == 'left' else '→'}{warn}"
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
            name = self.save_only()
            if not name:
                return
            subprocess.Popen([pythonw(), str(ROOT / "app" / "pet.py"), "--summon", name], cwd=str(ROOT))
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
            jobs = aifill.missing_jobs(name)
            if not jobs:
                QMessageBox.information(self, "動作都齊了", "走路、坐、趴的循環動作和轉場都已經有了，不需要 AI 補。")
                return
            st = load_settings()
            dlg = QDialog(self)
            dlg.setWindowTitle("🪄 AI 補齊動作")
            v = QVBoxLayout(dlg)
            v.addWidget(QLabel("AI 會用你的照片當「開始畫面」和「結束畫面」，生成中間的自然動作。\n"
                               "這隻寵物還缺這些（勾選要生成的）："))
            checks = []
            for j in jobs:
                c = QCheckBox(j.label)
                c.setChecked(True)
                v.addWidget(c)
                checks.append((c, j))
            v.addWidget(QLabel("<b>用哪個 AI？</b>"))
            hf = QRadioButton("Hugging Face（免費，每天有額度，約可做 2～4 段；尖峰時段要排隊）")
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
                cost.setText(f"預估費用：約 US${aifill.Fal.cost(n):.2f}（{n} 段 × {aifill.CLIP_SECS} 秒 × US${aifill.FAL_PRICE_PER_SEC}/秒）"
                             if fal.isChecked() else f"共 {n} 段，免費（用你的 Hugging Face 每日額度）")
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
            self.worker = AIWorker(name, chosen, backend)
            self.worker.progress.connect(self.on_progress)
            self.worker.result.connect(self.on_result)
            self.worker.finished_all.connect(self.on_ai_done)
            self.go.setEnabled(False)
            self.worker.start()

        def on_ai_done(self, ok, bad):
            self.go.setEnabled(True)
            msg = f"AI 完成 {ok} 段" + (f"，失敗 {bad} 段" if bad else "")
            self.status.setText(msg + "。檢查一下動作（不好的右鍵刪除），再按「儲存並召喚」。"
                                + (f"\n{self.worker.last_error}" if self.worker.last_error else ""))

    app = QApplication(sys.argv)
    w = Trainer()
    w.show()
    app.exec()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli(sys.argv[2:])
    else:
        run_gui()
