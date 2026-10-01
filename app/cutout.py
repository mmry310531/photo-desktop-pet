"""照片 -> 去背、裁切、判斷姿勢與面向。

流程（每張照片）：
  1. 讀檔（含 iPhone HEIC）、依 EXIF 轉正、縮到最長邊 1280
  2. rembg AI 模型去背（isnet-general-use；高品質模式用 BiRefNet）
  3. 只留最大的主體（去掉旁邊雜物的碎片），修邊、緊貼裁切
  4. 依外形判斷姿勢：坐 / 站走 / 躺；並猜牠面向哪一邊
  5. 用感知雜湊去掉重複照片
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone 的 HEIC 照片
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff", ".jfif", ".avif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".gif"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
OUT_HEIGHT = 360  # 存檔高度（顯示時再依面積縮放）


@dataclass
class CutResult:
    source: str
    image: Image.Image | None = None
    pose: str = "walk"
    facing: str = "right"  # 頭朝哪邊
    area: int = 0          # 不透明像素數（用來讓各張看起來一樣大）
    hash: int = 0
    frames: list = field(default_factory=list)   # 影片：[(PIL 圖, 錨點 x)]
    fps: float = 0.0
    warning: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


_sessions: dict = {}

# ---------------------------------------------------------------- 動物偵測（YOLOX-S，Apache-2.0）
# 先找出照片裡的貓狗在哪，只對那一塊去背，旁邊的腳踏車、狗屋、人就不會一起被剪下來。
DETECTOR_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
# COCO 類別：鳥、貓、狗、馬、羊、牛、熊、泰迪熊（毛茸茸的寵物偶爾會被認成它）
PET_CLASSES = {14: "鳥", 15: "貓", 16: "狗", 17: "馬", 18: "羊", 19: "牛", 21: "熊", 77: "毛玩偶"}
_detector = None


def _model_dir() -> Path:
    from common import ROOT

    d = ROOT / "models"
    d.mkdir(exist_ok=True)
    return d


def _dl_callback(label, progress):
    if not progress:
        return None

    def cb(done, total, speed):
        pct = done * 100 / max(total, 1)
        progress(f"下載{label}（只需一次）：{pct:4.1f}%　{done / 1e6:.0f}/{total / 1e6:.0f} MB　{speed / 1e6:.1f} MB/s")
    return cb


def get_detector(progress=None):
    global _detector
    if _detector is None:

        import onnxruntime as ort

        f = _model_dir() / "yolox_s.onnx"
        if not f.exists() or f.stat().st_size < 30_000_000:
            if progress:
                progress("下載動物偵測模型（約 36MB，只需一次）…")
            import fastdl

            urls, _, md5 = fastdl.YOLOX
            fastdl.download(urls, f, md5=md5, label="動物偵測模型", status=progress,
                            callback=_dl_callback("動物偵測模型", progress))
        _detector = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    return _detector


def detect_pet(img: Image.Image, det) -> tuple[tuple[int, int, int, int], str, float] | None:
    """回傳 (x0, y0, x1, y1), 類別名稱, 信心分數；找不到動物回傳 None。"""
    S = 640
    w, h = img.size
    r = min(S / w, S / h)
    rs = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    pad = np.full((S, S, 3), 114, dtype=np.float32)
    arr = np.asarray(rs, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    pad[: arr.shape[0], : arr.shape[1]] = arr
    out = det.run(None, {det.get_inputs()[0].name: pad.transpose(2, 0, 1)[None]})[0][0]
    grids, strides = [], []
    for st in (8, 16, 32):
        n = S // st
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), st))
    grids, strides = np.concatenate(grids), np.concatenate(strides)
    xy = (out[:, :2] + grids) * strides
    wh = np.exp(out[:, 2:4]) * strides
    scores = out[:, 4:5] * out[:, 5:]
    ids = list(PET_CLASSES)
    pet_scores = scores[:, ids]
    best_cls = pet_scores.argmax(1)
    best = pet_scores.max(1)
    if best.max() < 0.25:
        return None
    # 分數高的優先；分數差不多時挑大的
    area = wh[:, 0] * wh[:, 1]
    rank = best * (0.75 + 0.25 * np.sqrt(area / area.max()))
    rank[best < 0.25] = -1
    i = int(rank.argmax())
    cx, cy = xy[i] / r
    bw, bh = wh[i] / r
    box = (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))
    return box, PET_CLASSES[ids[int(best_cls[i])]], float(best[i])


def get_session(high_quality: bool = False, progress=None):
    """載入去背模型。高品質模型（約 900MB）下載失敗時自動退回標準模型，不讓整批失敗。"""
    import fastdl

    # 模型放在專案的 models/ 底下，並用多線程下載（比 rembg 內建的單線程快很多）
    os.environ["U2NET_HOME"] = str(_model_dir() / "u2net")
    if high_quality:
        urls, rel, md5 = fastdl.BIREFNET
        target = _model_dir() / rel
        try:
            if not target.exists():
                fastdl.download(urls, target, md5=md5, label="高品質去背模型", status=progress,
                                callback=_dl_callback("高品質去背模型（約 900MB）", progress))
            name = "birefnet-general"
        except Exception as e:
            if progress:
                progress(f"高品質模型下載失敗（{e}），改用標準模型繼續")
            high_quality = False
    if not high_quality:
        urls, rel, md5 = fastdl.ISNET
        target = _model_dir() / rel
        if not target.exists():
            fastdl.download(urls, target, md5=md5, label="去背模型", status=progress,
                            callback=_dl_callback("去背模型", progress))
        name = "isnet-general-use"

    from rembg import new_session

    if name not in _sessions:
        _sessions[name] = new_session(name)
    return _sessions[name]


def collect_images(paths) -> list[Path]:
    out: list[Path] = []
    from common import ROOT

    def ours(q: Path) -> bool:  # 不要把程式自己的資料夾（寵物包、快取、模型）當成素材
        try:
            q.resolve().relative_to(ROOT)
            return True
        except ValueError:
            return False

    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in MEDIA_EXTS and not ours(q))
        elif p.suffix.lower() in MEDIA_EXTS:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        k = str(p.resolve()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def dhash(img: Image.Image) -> int:
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _largest_component(alpha: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    solid = alpha > 128
    lab, n = ndimage.label(solid)
    if n <= 1:
        return alpha
    sizes = ndimage.sum(solid, lab, range(1, n + 1))
    biggest = sizes.max()
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= biggest * 0.15]
    keep = np.isin(lab, keep_ids)
    # 半透明毛邊也留著：把保留區膨脹幾個像素當遮罩
    keep = ndimage.binary_dilation(keep, iterations=4)
    return np.where(keep, alpha, 0).astype(np.uint8)


def clean_alpha(a: np.ndarray) -> np.ndarray:
    """去掉半透明的「霧」（床單、毯子被半透明地留下來），補起身體中間的洞。"""
    from scipy import ndimage

    a = np.clip((a.astype(np.float32) - 90) / (210 - 90), 0, 1) * 255
    solid = a > 128
    lab, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= sizes.max() * 0.15])
        a = np.where(ndimage.binary_dilation(keep, iterations=3), a, 0)
    filled = ndimage.binary_fill_holes(a > 128)
    a = np.where(filled & (a < 128), 255, a)
    return a.astype(np.uint8)


# ---------------------------------------------------------------- 姿勢辨識（CLIP 零樣本分類）
# 文字端的向量事先算好放在 clip_text.npy，使用者只需要下載圖片端模型（約 89MB）。
CLIP_URLS = ["https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
             "https://hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx"]
CLIP_MD5 = "f4f4e42828171ccd945ef40e183053c0"
_clip = None


def get_clip(progress=None):
    global _clip
    if _clip is None:
        import json

        import onnxruntime as ort

        import fastdl

        f = _model_dir() / "clip" / "vision_model_quantized.onnx"
        if not f.exists():
            fastdl.download(CLIP_URLS, f, md5=CLIP_MD5, label="姿勢辨識模型", status=progress,
                            callback=_dl_callback("姿勢辨識模型（約 89MB）", progress))
        here = Path(__file__).resolve().parent
        _clip = (ort.InferenceSession(str(f), providers=["CPUExecutionProvider"]),
                 np.load(here / "clip_text.npy"), json.load(open(here / "clip_labels.json", encoding="utf-8")))
    return _clip


def clip_pose(img: Image.Image) -> dict | None:
    if _clip is None:
        return None
    sess, T, labels = _clip
    w, h = img.size
    s = max(w, h)
    bg = Image.new("RGB", (s, s), (124, 116, 104))
    bg.paste(img.convert("RGB"), ((s - w) // 2, (s - h) // 2))
    x = np.asarray(bg.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255
    x = (x - [0.4815, 0.4578, 0.4082]) / [0.2686, 0.2613, 0.2758]
    e = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    e = e / np.linalg.norm(e)
    sims = T @ e * 100
    p = np.exp(sims - sims.max())
    p /= p.sum()
    out = {}
    for k in set(labels):  # 每類的機率加總（用北鼻 61 張實拍照片對照過，加總比平均準）
        idx = [i for i, l in enumerate(labels) if l == k]
        out[k] = float(p[idx].sum())
    return out


def classify(alpha: np.ndarray) -> tuple[str, str, dict]:
    """用外形判斷姿勢與面向（簡單規則，訓練器裡可以手動改）。"""
    solid = alpha > 128
    h, w = solid.shape
    ar = w / max(h, 1)
    # 底部 15% 的佔滿程度：站著時只有四隻腳碰地（有空隙），躺著時整片貼地
    band = solid[int(h * 0.85):, :]
    bottom_fill = float(band.any(axis=0).mean()) if band.size else 0.0
    if ar < 0.95:
        pose = "sit"
    elif (bottom_fill >= 0.55 and ar > 1.25) or ar > 2.0:
        pose = "lie"
    else:
        pose = "walk"
    # 面向：頭那一側通常比較高、也比較「厚」
    cols = solid.any(axis=0)
    tops = np.where(cols, solid.argmax(axis=0), h).astype(float)
    third = max(w // 3, 1)
    left_top, right_top = tops[:third].mean(), tops[-third:].mean()
    left_mass, right_mass = solid[:, :third].sum(), solid[:, -third:].sum()
    score = (right_top - left_top) / h + 0.5 * (left_mass - right_mass) / max(solid.sum(), 1)
    facing = "left" if score > 0 else "right"
    return pose, facing, {"aspect": round(ar, 2), "bottom_fill": round(bottom_fill, 2)}


def process(path: Path, session, existing_hashes: list[int] | None = None, detector=None) -> CutResult:
    r = CutResult(source=str(path))
    try:
        from rembg import remove

        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((1280, 1280), Image.LANCZOS)
        found_w, found_h = img.size
        r.hash = dhash(img)
        if existing_hashes and any(hamming(r.hash, h) <= 4 for h in existing_hashes):
            r.error = "重複的照片，已略過"
            return r

        found = detect_pet(img, detector) if detector is not None else None
        if found:
            (bx0, by0, bx1, by1), kind, conf = found
            r.extra["detected"] = f"{kind} {conf:.0%}"
            if (bx1 - bx0) > img.width * 0.9 and (by1 - by0) > img.height * 0.9:
                r.error = "太近的特寫，看不到身體（請用拍到全身的照片）"
                return r
            # 只拿動物附近那一塊去背
            mx, my = int((bx1 - bx0) * 0.12) + 8, int((by1 - by0) * 0.12) + 8
            cx0, cy0 = max(0, bx0 - mx), max(0, by0 - my)
            cx1, cy1 = min(img.width, bx1 + mx), min(img.height, by1 + my)
            img = img.crop((cx0, cy0, cx1, cy1))
            bx0, by0, bx1, by1 = bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0
        cut = remove(img, session=session)  # RGBA
        a = np.asarray(cut.getchannel("A")).copy()
        a[a < 24] = 0
        if found:
            # 偵測框外（稍微放寬）的東西都不要
            ex, ey = int((bx1 - bx0) * 0.04) + 3, int((by1 - by0) * 0.04) + 3
            keep = np.zeros_like(a, dtype=bool)
            keep[max(0, by0 - ey):by1 + ey, max(0, bx0 - ex):bx1 + ex] = True
            a[~keep] = 0
        elif detector is not None:
            r.warning = "沒認出是貓狗（若是兔子、倉鼠等請自行確認）"
        a = _largest_component(a)
        if found and (a > 128).sum() < (bx1 - bx0) * (by1 - by0) * 0.2:
            r.error = "去背失敗（主體和背景太像或太模糊）"
            return r
        if (a > 128).sum() < a.size * 0.02:
            r.error = "找不到明顯的主體（寵物太小或背景太亂）"
            return r

        raw = a.copy()
        a = clean_alpha(a)
        ys, xs = np.where(a > 24)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        H, W = a.shape
        edges = sum([y0 <= 1, x0 <= 1, y1 >= H - 1, x1 >= W - 1])
        if found:  # 用原圖邊界判斷（裁切過的邊不算）
            edges = sum([cy0 + y0 <= 1, cx0 + x0 <= 1, cy0 + y1 >= found_h - 1, cx0 + x1 >= found_w - 1])
        if edges >= 2:
            r.warning = "寵物可能被照片邊緣切到"

        # 品質分數：偵測信心、半透明殘留（背景沒去乾淨）、被切到、形狀是否合理
        solid = a > 128
        haze = float(((raw > 30) & (raw < 220)).sum() / max(solid.sum(), 1))
        conf = found[2] if found else 0.4
        fill = float(solid.sum() / max((bx1 - bx0) * (by1 - by0), 1)) if found else 0.5
        score = conf - 1.5 * haze - 0.25 * edges - 0.5 * abs(fill - 0.55)
        r.extra.update({"score": round(score, 2), "haze": round(haze, 2)})

        # 姿勢：先用 AI（CLIP）看內容判斷，沒有模型時才退回用形狀猜
        probs = clip_pose(img)
        if probs:
            r.extra["clip"] = {k: round(v, 2) for k, v in probs.items()}
            if probs.get("face", 0) > 0.35:
                r.error = "太近的特寫，看不到身體（請用拍到全身的照片）"
                return r
        if score < 0:
            r.error = f"品質太差，自動略過（背景沒去乾淨或被切到，分數 {score:.2f}）"
            return r

        rgba = np.dstack([np.asarray(cut)[:, :, :3], a])[y0:y1, x0:x1]
        out = Image.fromarray(rgba, "RGBA")
        scale = OUT_HEIGHT / out.height
        out = out.resize((max(1, round(out.width * scale)), OUT_HEIGHT), Image.LANCZOS)
        al = np.asarray(out.getchannel("A"))
        r.pose, r.facing, info = classify(al)
        if probs:
            r.pose = max(("walk", "sit", "lie"), key=lambda k: probs.get(k, 0))
        r.extra.update(info)
        r.area = int((al > 128).sum())
        r.image = out
    except Exception as e:  # 壞檔、格式不支援…
        r.error = f"{type(e).__name__}: {e}"
    return r


def match_brightness(images: list[Image.Image], strength: float = 0.5) -> list[Image.Image]:
    """把每張照片的亮度往整組的中位數拉一半，換姿勢時才不會忽明忽暗。"""
    lums = []
    for im in images:
        arr = np.asarray(im).astype(np.float32)
        m = arr[:, :, 3] > 128
        lums.append(arr[:, :, :3][m].mean() if m.any() else 128.0)
    target = float(np.median(lums)) if lums else 128.0
    out = []
    for im, l in zip(images, lums):
        g = 1 + strength * (target / max(l, 1) - 1)
        g = float(np.clip(g, 0.75, 1.35))
        arr = np.asarray(im).astype(np.float32)
        arr[:, :, :3] = np.clip(arr[:, :, :3] * g, 0, 255)
        out.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
    return out


# ================================================================ 影片 -> 連續動畫
VIDEO_FPS = 10        # 每秒取幾格
VIDEO_MAX_SECS = 6    # 最多處理幾秒
VIDEO_W = 640


def read_video_frames(path: Path, fps=VIDEO_FPS, max_secs=VIDEO_MAX_SECS):
    """用 ffmpeg 取出影格（手機直拍影片會自動轉正）。

    注意：不能讓 ffmpeg 縮放，imageio_ffmpeg 是用原始尺寸切每一格的，縮放會讓畫面錯位。
    """
    import imageio_ffmpeg

    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24",
                                     output_params=["-vf", f"fps={fps}", "-t", str(max_secs)])
    meta = next(gen)
    w, h = meta["size"]
    frames = []
    for raw in gen:
        im = Image.frombytes("RGB", (w, h), bytes(raw))
        if im.width > VIDEO_W:
            im = im.resize((VIDEO_W, round(im.height * VIDEO_W / im.width)), Image.LANCZOS)
        frames.append(im)
    return frames


def _cut_frame(img, session, detector, last_box):
    """單一影格去背；偵測不到時沿用上一格的位置。回傳 (alpha, box, 原圖 RGBA) 或 None。"""
    from rembg import remove

    found = detect_pet(img, detector) if detector is not None else None
    box = found[0] if found else last_box
    if box is None:
        return None
    bx0, by0, bx1, by1 = box
    mx, my = int((bx1 - bx0) * 0.15) + 8, int((by1 - by0) * 0.15) + 8
    cx0, cy0 = max(0, bx0 - mx), max(0, by0 - my)
    cx1, cy1 = min(img.width, bx1 + mx), min(img.height, by1 + my)
    crop = img.crop((cx0, cy0, cx1, cy1))
    cut = remove(crop, session=session)
    a = np.asarray(cut.getchannel("A")).copy()
    a[a < 24] = 0
    a = _largest_component(a)
    if (a > 128).sum() < (bx1 - bx0) * (by1 - by0) * 0.2:
        return None
    rgba = np.dstack([np.asarray(cut)[:, :, :3], a])
    return a, box, rgba


def _best_loop(masks, min_len=6):
    """找出頭尾最像的一段，讓動畫循環播放時接得順。"""
    n = len(masks)
    if n <= min_len:
        return 0, n
    best, bi, bj = -1.0, 0, n
    for i in range(n):
        for j in range(i + min_len, n + 1):
            k = j % n if j == n else j
            if j == n:
                continue
            inter = np.logical_and(masks[i], masks[k]).sum()
            union = np.logical_or(masks[i], masks[k]).sum() or 1
            score = inter / union + 0.006 * (j - i)
            if score > best:
                best, bi, bj = score, i, j
    return bi, bj


def _pose_of(frames_alpha):
    """幾格畫面的多數決姿勢。"""
    votes = [classify(a)[0] for a in frames_alpha]
    return max(set(votes), key=votes.count)


def process_video(path: Path, session, detector, progress=None, force: dict | None = None,
                  max_secs=VIDEO_MAX_SECS) -> CutResult:
    """把一段寵物影片變成連續動畫。

    自動判斷是「循環動作」（走路、坐著搖尾巴、睡覺呼吸）還是「轉場動作」（坐下、趴下、起身）：
    開頭和結尾的姿勢不同就是轉場。force 可以直接指定（AI 生成的影片用）。
    """
    r = CutResult(source=str(path))
    try:
        frames = read_video_frames(path, max_secs=max_secs)
        if len(frames) < 4:
            r.error = "影片太短"
            return r
        cuts, boxes = [], []
        last = None
        for i, fr in enumerate(frames):
            if progress:
                progress(f"處理影片 {Path(path).name}：第 {i + 1}/{len(frames)} 格")
            c = _cut_frame(fr, session, detector, last)
            if c is None:
                continue
            a, box, rgba = c
            last = box
            cuts.append((a, rgba))
            boxes.append(box)
        if len(cuts) < 4:
            r.error = "影片裡找不到清楚的寵物"
            return r

        tight = []
        for a, rgba in cuts:
            ys, xs = np.where(a > 24)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            tight.append(rgba[y0:y1, x0:x1])
        areas = np.array([(t[:, :, 3] > 128).sum() for t in tight], dtype=float)
        # 去掉被擋住、去背失敗的格（跟前後格比，不跟整段比——轉場時面積本來就會變）
        keep = np.ones(len(tight), dtype=bool)
        for i in range(len(tight)):
            nb = areas[max(0, i - 3):i + 4]
            keep[i] = abs(areas[i] - np.median(nb)) < np.median(nb) * 0.4
        tight = [t for t, k in zip(tight, keep) if k]
        boxes = [bx for bx, k in zip(boxes, keep) if k]
        if len(tight) < 4:
            r.error = "影片裡寵物被擋住或去背失敗太多格"
            return r

        # 整段用同一個縮放比例（不逐格縮放，坐下、趴下的高度變化才會保留）
        sc = OUT_HEIGHT / max(t.shape[0] for t in tight)
        out_frames, masks, alphas = [], [], []
        for t in tight:
            im = Image.fromarray(t, "RGBA")
            im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
            al = np.asarray(im.getchannel("A"))
            solid = al > 128
            # 錨點用身體的質心，比框的中心穩（尾巴擺動不會讓整隻左右抖）
            ax = float(np.where(solid)[1].mean()) if solid.any() else im.width / 2
            out_frames.append((im, ax))
            alphas.append(al)
            m = Image.fromarray((solid * 255).astype(np.uint8)).resize((32, 32))
            masks.append(np.asarray(m) > 127)

        n = len(out_frames)
        cxs = np.array([(bx[0] + bx[2]) / 2 for bx in boxes])
        widths = np.array([bx[2] - bx[0] for bx in boxes])
        travel = (cxs[-1] - cxs[0]) / max(np.median(widths), 1)
        start_pose = _pose_of(alphas[: min(3, n)])
        end_pose = _pose_of(alphas[-min(3, n):])
        _, facing, info = classify(alphas[n // 2])
        r.extra.update(info)
        r.extra["travel"] = round(float(travel), 2)

        if force:
            kind = force["kind"]
            start_pose, end_pose = force.get("from", start_pose), force.get("to", end_pose)
            facing = force.get("facing", facing)
        elif abs(travel) > 0.5:  # 在畫面裡移動了 → 走路循環
            kind, start_pose = "loop", "walk"
            end_pose = "walk"
            facing = "right" if travel > 0 else "left"
        elif start_pose != end_pose:
            kind = "trans"
        else:
            kind = "loop"

        if kind == "loop":
            if force and force.get("seamless"):
                i, j = 0, n - 1  # AI 用同一張圖當頭尾，本身就接得起來；最後一格跟第一格重複
            else:
                i, j = _best_loop(masks)
            r.frames = out_frames[i:j]
            r.pose = start_pose
            if len(r.frames) < n * 0.5:
                r.warning = "找到的循環段較短，動作可能不夠完整"
        else:
            r.frames = out_frames
            r.pose = start_pose
            r.extra["to"] = end_pose
        r.extra["kind"] = kind
        r.fps = VIDEO_FPS
        r.facing = facing
        r.image = r.frames[0][0]
        r.area = int(np.median([(np.asarray(f.getchannel("A")) > 128).sum() for f, _ in r.frames]))
        r.hash = dhash(out_frames[n // 2][0].convert("RGB"))
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
    return r


def process_any(path: Path, session, hashes, detector, progress=None) -> CutResult:
    if Path(path).suffix.lower() in VIDEO_EXTS:
        return process_video(path, session, detector, progress)
    return process(path, session, hashes, detector)
