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
from PIL import Image, ImageFilter, ImageOps

try:  # iPhone 的 HEIC 照片
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}
OUT_HEIGHT = 360  # 存檔高度（顯示時再依面積縮放）


@dataclass
class CutResult:
    source: str
    image: Image.Image | None = None
    pose: str = "walk"
    facing: str = "right"  # 頭朝哪邊
    area: int = 0          # 不透明像素數（用來讓各張看起來一樣大）
    hash: int = 0
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


def get_detector(progress=None):
    global _detector
    if _detector is None:
        import urllib.request

        import onnxruntime as ort

        f = _model_dir() / "yolox_s.onnx"
        if not f.exists() or f.stat().st_size < 30_000_000:
            if progress:
                progress("下載動物偵測模型（約 36MB，只需一次）…")
            try:
                import fastdl

                fastdl.download(DETECTOR_URL, f)
            except Exception:
                tmp = f.with_suffix(".part")
                urllib.request.urlretrieve(DETECTOR_URL, tmp)
                tmp.replace(f)
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
    import fastdl

    # 模型放在專案的 models/ 底下，並用多線程下載（比 rembg 內建的單線程快很多）
    os.environ["U2NET_HOME"] = str(_model_dir() / "u2net")
    name = "birefnet-general" if high_quality else "isnet-general-use"
    url, rel, md5 = fastdl.BIREFNET if high_quality else fastdl.MODELS[0]
    target = _model_dir() / rel
    if not target.exists():
        if progress:
            progress(f"下載去背模型 {target.name}（只需一次）…")
        try:
            fastdl.download(url, target, md5=md5)
        except Exception:
            pass  # 交給 rembg 自己下載

    from rembg import new_session

    if name not in _sessions:
        _sessions[name] = new_session(name)
    return _sessions[name]


def collect_images(paths) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
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

        ys, xs = np.where(a > 24)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        H, W = a.shape
        edges = sum([y0 <= 1, x0 <= 1, y1 >= H - 1, x1 >= W - 1])
        if found:  # 用原圖邊界判斷（裁切過的邊不算）
            edges = sum([cy0 + y0 <= 1, cx0 + x0 <= 1, cy0 + y1 >= found_h - 1, cx0 + x1 >= found_w - 1])
        if edges >= 2:
            r.warning = "寵物可能被照片邊緣切到"

        rgba = np.dstack([np.asarray(cut)[:, :, :3], a])[y0:y1, x0:x1]
        out = Image.fromarray(rgba, "RGBA")
        # 輕微柔化邊緣避免鋸齒
        alpha = out.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
        out.putalpha(alpha)

        scale = OUT_HEIGHT / out.height
        out = out.resize((max(1, round(out.width * scale)), OUT_HEIGHT), Image.LANCZOS)
        al = np.asarray(out.getchannel("A"))
        r.pose, r.facing, info = classify(al)
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
