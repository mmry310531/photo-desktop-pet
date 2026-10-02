"""照片 -> 去背、裁切、判斷姿勢與面向。

流程（每張照片）：
  1. 讀檔（含 iPhone HEIC）、依 EXIF 轉正、縮到最長邊 1280
  2. rembg AI 模型去背（isnet-general-use），再用 SAM 依偵測框圈出整隻動物修輪廓
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
FRAME_CUT_MAX = 0.12  # 身體貼著照片邊緣的長度超過這個比例 → 被照片切到，不要


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


def get_session(fast: bool = False, progress=None):
    """載入去背模型（IS-Net）＋輪廓模型（SAM）。

    照片和影片的每一格都會用 SAM 修輪廓（影片裡貓的頭常常貼著深色背景或雜物，只靠 IS-Net 會缺頭缺腳）。
    fast=True：影片不精修（快很多，品質較粗）。照片一律精修。
    以前的高品質選項是 BiRefNet，但它在一般電腦（只用 CPU）要吃 6GB 以上記憶體，實測會直接當掉，已移除。
    """
    global _refine_video
    import fastdl

    # 模型放在專案的 models/ 底下，並用多線程下載（比 rembg 內建的單線程快很多）
    os.environ["U2NET_HOME"] = str(_model_dir() / "u2net")
    urls, rel, md5 = fastdl.ISNET
    target = _model_dir() / rel
    if not target.exists():
        fastdl.download(urls, target, md5=md5, label="去背模型", status=progress,
                        callback=_dl_callback("去背模型", progress))
    name = "isnet-general-use"
    get_sam(progress)
    _refine_video = not fast

    from rembg import new_session

    if name not in _sessions:
        _sessions[name] = new_session(name)
    return _sessions[name]


# ---------------------------------------------------------------- 輪廓精修（Segment Anything）
# 去背模型（IS-Net）毛邊很細，但它只看「顯不顯眼」：黑毛貼著深色背景會被當成背景挖掉，
# 顏色跟寵物很像的毯子、衣服又會被一起留下。SAM 是「給一個框，把框裡那個東西整個圈出來」的模型，
# 對「哪些是同一隻動物」判斷得準很多。做法（類似 Matte Anything 的流程）：
#   偵測框 + IS-Net 的結果 當提示 → SAM 圈出整隻 → 身體內部一律保留，邊緣一圈用 IS-Net 的細毛邊，外面全部去掉
SAM_CANVAS = (684, 1024)  # 送進 SAM 的畫布（高, 寬）；跟 rembg 一樣，實測比正方形穩
SAM_MIN_FREE_MB = 3800  # SAM 跑一張約需 3.3GB 記憶體；可用記憶體不夠時自動改用原本的方式
_sam = None
_sam_state = ""  # "", "ok", "lowmem", "fail"
_refine_video = False


def available_mb() -> int | None:
    """目前可用的實體記憶體（MB）；拿不到時回傳 None。"""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullAvailPhys // 2**20)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def get_sam(progress=None):
    """載入 SAM（約 125MB，只下載一次）。記憶體不夠或下載失敗時回傳 None，流程照常（只是不精修）。"""
    global _sam, _sam_state
    if _sam is not None:
        return _sam
    if _sam_state == "fail":
        return None
    free = available_mb()
    if free is not None and free < SAM_MIN_FREE_MB:
        if _sam_state != "lowmem" and progress:
            progress(f"可用記憶體只剩 {free / 1024:.1f}GB，先不用輪廓精修（關掉一些程式再處理效果會更好）")
        _sam_state = "lowmem"
        return None
    try:
        import onnxruntime as ort

        import fastdl

        paths = []
        for (urls, rel, md5), label in ((fastdl.SAM_ENC, "輪廓模型"), (fastdl.SAM_DEC, "輪廓模型（2/2）")):
            f = _model_dir() / rel
            if not f.exists():
                fastdl.download(urls, f, md5=md5, label=label, status=progress,
                                callback=_dl_callback(label + "（約 125MB）", progress))
            paths.append(str(f))
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # 跑完就把那 3GB 還給系統，不要一直佔著
        _sam = tuple(ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for p in paths)
        _sam_state = "ok"
    except Exception as e:
        _sam_state = "fail"
        if progress:
            progress(f"輪廓模型載入失敗（{type(e).__name__}），改用基本去背繼續")
        return None
    return _sam


def _keep_big(m: np.ndarray, frac: float) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sz = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, v in enumerate(sz) if v >= sz.max() * frac])


def sam_mask(img: Image.Image, box, raw: np.ndarray) -> np.ndarray | None:
    """回傳 SAM 認為是寵物的機率圖（0～1，跟 img 一樣大）；不能用時回傳 None。"""
    if _sam is None and get_sam() is None:
        return None
    enc, dec = _sam
    free = available_mb()
    if free is not None and free < SAM_MIN_FREE_MB - 600:
        return None
    w, h = img.size
    CH, CW = SAM_CANVAS
    s = min(CW / w, CH / h)
    sw, sh = round(w * s), round(h * s)
    can = np.zeros((CH, CW, 3), np.float32)
    can[:sh, :sw] = np.asarray(img.convert("RGB").resize((sw, sh), Image.BILINEAR), np.float32)
    emb = enc.run(None, {enc.get_inputs()[0].name: can})[0]
    # 提示：偵測框（左上、右下兩個角）＋ IS-Net 的結果（低解析度遮罩）
    b = np.asarray(box, np.float32) * s
    coords = np.array([[[b[0], b[1]], [b[2], b[3]], [0, 0]]], np.float32)
    labels = np.array([[2, 3, -1]], np.float32)
    lw, lh = max(1, round(sw / 4)), max(1, round(sh / 4))
    prior = np.full((1, 1, 256, 256), -8, np.float32)
    r = np.asarray(Image.fromarray(raw).resize((lw, lh), Image.BILINEAR), np.float32) / 255
    prior[0, 0, :lh, :lw] = (r - 0.5) * 16
    _, _, low = dec.run(None, {"image_embeddings": emb, "point_coords": coords, "point_labels": labels,
                               "mask_input": prior, "has_mask_input": np.ones(1, np.float32),
                               "orig_im_size": np.array([CH, CW], np.float32)})
    # 用低解析度的原始輸出自己放大（模型內建的放大會出現格子狀雜點）
    L = low[0, 0][:lh, :lw].astype(np.float32)
    L = np.asarray(Image.fromarray(L, "F").resize((w, h), Image.BILINEAR))
    return 1 / (1 + np.exp(-np.clip(L, -20, 20)))


def refine_alpha(raw: np.ndarray, p: np.ndarray) -> np.ndarray:
    """IS-Net 的透明度（毛邊細）＋ SAM 的整體輪廓（不漏身體、不帶毯子）合成最後的透明度。"""
    from scipy import ndimage

    H, W = raw.shape
    d = max(H, W)
    p = ndimage.gaussian_filter(p, sigma=d / 256 * 1.2)
    M = ndimage.binary_fill_holes(_keep_big(p > 0.5, 0.2))
    r = max(2, int(0.015 * d))
    core = ndimage.binary_erosion(M, iterations=r)        # 身體內部：一定保留（黑毛不會再被挖掉）
    band = ndimage.binary_dilation(M, iterations=r)       # 輪廓外一小圈：只留 IS-Net 有把握的細毛
    rawf = raw.astype(np.float32)
    lo = np.clip((rawf - 35) / (150 - 35), 0, 1)
    soft = np.clip((p - 0.5) * 4, 0, 1)
    a = np.where(core, 1.0, np.where(M, np.maximum(lo, soft), np.where(band, lo * (rawf > 60), 0.0)))
    a = (a * 255).astype(np.uint8)
    solid = ndimage.binary_fill_holes(_keep_big(a > 128, 0.15))
    a = np.where(ndimage.binary_dilation(solid, iterations=2), a, 0).astype(np.uint8)
    a[solid & (a < 128)] = 255
    return a


def matte(img: Image.Image, raw: np.ndarray, box, refine: bool = True) -> tuple[np.ndarray, bool]:
    """最後的透明度。有 SAM 就精修，沒有就用原本的整理方式。回傳 (alpha, 有沒有精修)。"""
    if refine and box is not None:
        try:
            p = sam_mask(img, box, raw)
            if p is not None and (p > 0.5).sum() > raw.size * 0.02:
                return refine_alpha(raw, p), True
        except Exception:  # 精修失敗不影響整批
            pass
    return clean_alpha(raw), False


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
    """整理去背的透明度：
    1. 身體附近用低門檻（深色毛、白色毛邊模型常常只給一半把握，不能丟）
       離身體遠的地方用高門檻（去掉床單、毯子被半透明留下來的「霧」）
    2. 只留主體（和夠大的部分）
    3. 補洞，並補回輪廓上凹進去、模型有一點把握的缺口（黑毛貼著深色背景時常被吃掉一塊）
    """
    from scipy import ndimage

    raw = a.astype(np.float32)
    core = raw > 200
    lab, n = ndimage.label(core)
    if n > 1:
        sz = ndimage.sum(core, lab, range(1, n + 1))
        core = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= sz.max() * 0.1])
    H, W = raw.shape
    r = max(3, int(0.035 * max(H, W)))
    near = ndimage.binary_dilation(core, iterations=r) if core.any() else np.zeros_like(core)
    lo = np.clip((raw - 35) / (150 - 35), 0, 1)
    hi = np.clip((raw - 110) / (220 - 110), 0, 1)
    out = np.where(near, lo, hi) * 255
    solid = out > 128
    lab, n = ndimage.label(solid)
    if n > 1:
        sz = ndimage.sum(solid, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= sz.max() * 0.15])
        out = np.where(ndimage.binary_dilation(keep, iterations=3), out, 0)
    filled = ndimage.binary_fill_holes(out > 128)
    out = np.where(filled & (out < 128), 255, out)
    # 輪廓缺口
    solid = out > 128
    rr = max(4, int(0.04 * max(H, W)))
    closed = ndimage.binary_closing(np.pad(solid, rr), structure=np.ones((3, 3)), iterations=rr)[rr:-rr, rr:-rr]
    add = closed & ~solid & (raw > 8)
    out[add] = 255
    return out.astype(np.uint8)


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
        a, refined = matte(img, a, (bx0, by0, bx1, by1) if found else None)
        r.extra["refined"] = refined
        if refined:
            raw = a  # 背景的霧已經被輪廓去掉，品質分數改看最後結果
        ys, xs = np.where(a > 24)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        H, W = a.shape
        edges = sum([y0 <= 1, x0 <= 1, y1 >= H - 1, x1 >= W - 1])
        if found:  # 用原圖邊界判斷（裁切過的邊不算）
            edges = sum([cy0 + y0 <= 1, cx0 + x0 <= 1, cy0 + y1 >= found_h - 1, cx0 + x1 >= found_w - 1])
        if edges >= 2:
            r.warning = "寵物可能被照片邊緣切到"
        # 身體貼著「照片邊緣」多長：頭、屁股、腳被照片切掉時很長（這種去背再好也補不回來）
        sl = a > 128
        side = []
        if (cx0 if found else 0) == 0:
            side.append(sl[:, 0].sum() / H)
        if (cy0 if found else 0) == 0:
            side.append(sl[0, :].sum() / W)
        if (cx1 if found else W) >= found_w:
            side.append(sl[:, -1].sum() / H)
        if (cy1 if found else H) >= found_h:
            side.append(sl[-1, :].sum() / W)
        cut_by_frame = max(side) if side else 0.0
        r.extra["frame_cut"] = round(float(cut_by_frame), 3)

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
            if probs.get("face", 0) > 0.2:
                r.error = "太近的特寫，看不到身體（請用拍到全身的照片）"
                return r
        if cut_by_frame > FRAME_CUT_MAX:
            r.error = "寵物有一部分在照片外面（身體不完整，放到桌面上會像被切掉一塊）"
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
    if _refine_video:
        a, _ = matte(crop, a, (bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0))
    else:
        a = clean_alpha(a)  # 去掉背景殘留的半透明「霧」、補回黑毛
    rgba = np.dstack([np.asarray(cut)[:, :, :3], a])
    # 身體貼著「畫面邊緣」的長度（頭或身體跑出畫面外時會很長；只有尾巴尖碰到時很短）
    H, W = a.shape
    solid = a > 128
    touch = max([solid[:, 0].sum() / H if cx0 == 0 else 0, solid[:, -1].sum() / H if cx1 == img.width else 0,
                 solid[0, :].sum() / W if cy0 == 0 else 0, solid[-1, :].sum() / W if cy1 == img.height else 0])
    return a, box, rgba, touch


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


def _cutout_pose(imgs, default: str) -> str:
    """用去背後的圖判斷姿勢：放在素色背景上給 CLIP 看「是不是趴／躺」（這點它很準），
    坐和站則看外形比例（CLIP 常把坐著看成站著；站著走路的貓明顯比較長）。"""
    if not imgs:
        return default
    acc, ars = {}, []
    for im in imgs:
        a = np.asarray(im.getchannel("A")) > 128
        ys, xs = np.where(a)
        if len(xs):
            ars.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        bg = Image.new("RGB", im.size, (124, 116, 104))
        bg.paste(im, (0, 0), im)
        p = clip_pose(bg)
        for kk, v in (p or {}).items():
            acc[kk] = acc.get(kk, 0) + v
    ar = float(np.median(ars)) if ars else 1.5
    if acc and acc.get("lie", 0) >= max(acc.get("walk", 0), acc.get("sit", 0)):
        return "lie"
    if not acc and ar > 2.1:
        return "lie"
    return "sit" if ar < 1.4 else "walk"


def process_video(path: Path, session, detector, progress=None, force: dict | None = None,
                  max_secs=VIDEO_MAX_SECS, frames: list | None = None) -> CutResult:
    """把一段寵物影片變成連續動畫。

    自動判斷是「循環動作」（走路、坐著搖尾巴、睡覺呼吸）還是「轉場動作」（坐下、趴下、起身）：
    開頭和結尾的姿勢不同就是轉場。force 可以直接指定（AI 生成的影片用）。
    """
    r = CutResult(source=str(path))
    try:
        force_keep = bool(force and force.get("seamless"))  # AI 影片：構圖是程式排的，不會出界
        if frames is None:
            frames = read_video_frames(path, max_secs=max_secs)
        if len(frames) < 4:
            r.error = "影片太短"
            return r
        cuts, boxes, idx = [], [], []
        last = None
        for i, fr in enumerate(frames):
            if progress:
                progress(f"處理影片 {Path(path).name}：第 {i + 1}/{len(frames)} 格")
            c = _cut_frame(fr, session, detector, last)
            if c is None:
                continue
            a, box, rgba, touch = c
            last = box
            if touch > 0.12 and not force_keep:
                continue  # 頭或身體有一塊在畫面外，這格不要
            cuts.append((a, rgba))
            boxes.append(box)
            idx.append(i)
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
        # 形狀跟前後兩格都差很多的（去背突然失敗、被手擋住）也去掉，不然播放時會閃一下
        shapes = [np.asarray(Image.fromarray(((t[:, :, 3] > 128) * 255).astype(np.uint8)).resize((32, 32))) > 127
                  for t in tight]

        def iou(p, q):
            return np.logical_and(p, q).sum() / max(np.logical_or(p, q).sum(), 1)
        for i in range(len(tight)):
            # 跟前後幾格的「多數形狀」比：連續好幾格都缺頭時，只跟前後一格比會抓不到
            nb = [shapes[j] for j in range(max(0, i - 4), min(len(tight), i + 5)) if j != i and keep[j]]
            if len(nb) >= 2:
                consensus = np.mean(nb, axis=0) > 0.5
                if iou(shapes[i], consensus) < 0.72:
                    keep[i] = False
        tight = [t for t, k in zip(tight, keep) if k]
        boxes = [bx for bx, k in zip(boxes, keep) if k]
        idx = [x for x, k in zip(idx, keep) if k]
        # 丟掉壞格後只留最長的「連續」一段（中間最多跳 1 格），播放時才不會突然跳一下
        if not force_keep and len(idx) > 1:
            runs, st = [], 0
            for k in range(1, len(idx) + 1):
                if k == len(idx) or idx[k] - idx[k - 1] > 2:
                    runs.append((st, k))
                    st = k
            a0, a1 = max(runs, key=lambda ab: ab[1] - ab[0])
            tight, boxes = tight[a0:a1], boxes[a0:a1]
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
            if force.get("mined"):  # 手機影片：去背後再判斷一次姿勢（掃描時背景雜亂，常判錯）
                if kind == "loop":
                    start_pose = end_pose = _cutout_pose([f for f, _ in out_frames[::3]], start_pose)
                else:  # 轉場：頭尾各看幾格；兩邊判成一樣時（例如趴低身子走路）保留掃描時的判斷
                    k = max(2, min(5, n // 3))
                    a = _cutout_pose([f for f, _ in out_frames[:k]], start_pose)
                    b = _cutout_pose([f for f, _ in out_frames[-k:]], end_pose)
                    if a != b:
                        start_pose, end_pose = a, b
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


# ---------------------------------------------------------------- 從手機影片裡挑出能用的片段
# 一般人拍的影片是手拿、鏡頭跟著貓移動、貓一下走近一下走遠，常常一段影片裡有好幾個動作。
# 所以不再只取前 6 秒當一整段，而是先快速掃過整段影片（只做偵測＋姿勢判斷，很快），
# 找出「姿勢穩定、大小穩定、全身都在畫面裡」的片段，再只對那幾段去背：
#   - 同一個姿勢持續 1.5 秒以上 → 循環動作（坐著待機、趴著、走路）
#   - 姿勢從 A 換成 B（例如趴著 → 起身走路）→ 轉場動作
SCAN_MAX_SECS = 30
LOOP_MIN, LOOP_MAX, TRANS_PAD = 15, 40, 8   # 以格數計（每秒 10 格）
MAX_CLIPS_PER_VIDEO = 4


def _frame_pose(probs: dict | None, box) -> str:
    """影片單格的姿勢。CLIP 分辨「趴／躺」很準，但常把坐著看成站著；坐和站改用外框比例判斷。"""
    w, h = box[2] - box[0], box[3] - box[1]
    if probs and probs.get("lie", 0) > max(probs.get("walk", 0), probs.get("sit", 0)) + 0.05:
        return "lie"
    if probs is None and w / max(h, 1) > 2.1:
        return "lie"
    return "sit" if w / max(h, 1) < 1.3 else "walk"


def scan_video(frames, detector, progress=None, name=""):
    """快速掃描：每格的框、姿勢、能不能用。"""
    info = []
    for i, im in enumerate(frames):
        if progress and i % 10 == 0:
            progress(f"掃描影片 {name}：{i}/{len(frames)} 格")
        d = detect_pet(im, detector) if detector is not None else None
        if not d:
            info.append(None)
            continue
        (x0, y0, x1, y1), _, conf = d
        mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        crop = im.crop((max(0, x0 - mx), max(0, y0 - my), min(im.width, x1 + mx), min(im.height, y1 + my)))
        probs = clip_pose(crop)
        # 框完全貼死畫面邊緣才算出界；只是尾巴尖碰到邊的，等去背後再看身體有多少貼著邊
        edge = x0 <= 1 or y0 <= 1 or x1 >= im.width - 1 or y1 >= im.height - 1
        info.append({"box": (x0, y0, x1, y1), "h": y1 - y0, "pose": _frame_pose(probs, (x0, y0, x1, y1)),
                     "ok": conf > 0.4 and not edge and (y1 - y0) > im.height * 0.12})
    # 姿勢做時間上的平滑（前後 7 格多數決），避免一兩格判錯就切斷
    raw = [x["pose"] if x else None for x in info]
    for i, x in enumerate(info):
        if x:
            win = [p for p in raw[max(0, i - 3):i + 4] if p]
            x["pose"] = max(set(win), key=win.count)
    return info


def find_segments(info) -> list[dict]:
    """把掃描結果切成「姿勢相同、大小穩定、連續可用」的段落。"""
    segs, cur = [], None
    for i, x in enumerate(info):
        good = x is not None and x["ok"]
        if good and cur and x["pose"] == cur["pose"] and abs(np.log(x["h"] / np.median(cur["hs"]))) < 0.22:
            cur["end"] = i + 1
            cur["hs"].append(x["h"])
            continue
        if cur:
            segs.append(cur)
        cur = {"pose": x["pose"], "start": i, "end": i + 1, "hs": [x["h"]]} if good else None
    if cur:
        segs.append(cur)
    return segs


def plan_clips(info) -> list[dict]:
    """挑出要做成動畫的片段：每種姿勢最長的循環段，加上姿勢之間的轉場。"""
    segs = find_segments(info)
    clips = []
    for a, b in zip(segs, segs[1:]):  # 轉場：A 段結尾 → B 段開頭，中間空檔不能太長、大小要接得上
        gap = b["start"] - a["end"]
        if (a["pose"] != b["pose"] and gap <= 10 and a["end"] - a["start"] >= 5 and b["end"] - b["start"] >= 5
                and abs(np.log(np.median(b["hs"][:5]) / np.median(a["hs"][-5:]))) < 0.35):
            clips.append({"kind": "trans", "from": a["pose"], "to": b["pose"],
                          "start": max(a["start"], a["end"] - TRANS_PAD), "end": min(b["end"], b["start"] + TRANS_PAD),
                          "score": 100})
    best = {}
    for sgm in segs:
        n = sgm["end"] - sgm["start"]
        if n >= LOOP_MIN and n > best.get(sgm["pose"], {}).get("n", 0):
            best[sgm["pose"]] = {"kind": "loop", "from": sgm["pose"], "to": sgm["pose"], "start": sgm["start"],
                                 "end": sgm["end"], "n": n, "score": n}
    clips += best.values()
    clips.sort(key=lambda c: -c["score"])
    chosen = []
    for c in clips:  # 不重疊（轉場可以和循環段的頭尾共用幾格）
        if all(c["end"] <= o["start"] + 4 or c["start"] >= o["end"] - 4 for o in chosen):
            chosen.append(c)
        if len(chosen) >= MAX_CLIPS_PER_VIDEO:
            break
    return sorted(chosen, key=lambda c: c["start"])


def process_video_clips(path: Path, session, detector, progress=None) -> list[CutResult]:
    """一段手機影片 → 0～4 個動畫片段（循環或轉場）。"""
    name = Path(path).name
    try:
        frames = read_video_frames(path, max_secs=SCAN_MAX_SECS)
    except Exception as e:
        return [CutResult(source=str(path), error=f"讀不到影片：{type(e).__name__}: {e}")]
    if len(frames) < 4:
        return [CutResult(source=str(path), error="影片太短")]
    info = scan_video(frames, detector, progress, name)
    plans = plan_clips(info)
    if not plans:
        return [CutResult(source=str(path), error="影片裡找不到姿勢穩定、全身入鏡的片段（鏡頭盡量固定、拍到全身）")]
    out = []
    for k, c in enumerate(plans, 1):
        sub = frames[c["start"]:c["end"]]
        if c["kind"] == "loop" and len(sub) > LOOP_MAX:
            sub = sub[:LOOP_MAX + 10]  # 循環段太長只取前面一段，_best_loop 會在裡面找頭尾最接的
        r = process_video(path, session, detector,
                          (lambda m, k=k: progress(f"{m}（片段 {k}/{len(plans)}）")) if progress else None,
                          force={"kind": c["kind"], "from": c["from"], "to": c["to"], "mined": True}, frames=sub)
        r.source = f"{path}#{c['start'] / VIDEO_FPS:.1f}-{c['end'] / VIDEO_FPS:.1f}s"
        r.extra["segment"] = [round(c["start"] / VIDEO_FPS, 1), round(c["end"] / VIDEO_FPS, 1)]
        out.append(r)
    return out


def process_any(path: Path, session, hashes, detector, progress=None) -> list[CutResult]:
    """照片 → 1 個結果；影片 → 可能好幾個片段。"""
    if Path(path).suffix.lower() in VIDEO_EXTS:
        return process_video_clips(path, session, detector, progress)
    return [process(path, session, hashes, detector)]
