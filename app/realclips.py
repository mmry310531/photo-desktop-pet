"""寫實動作：用「接點畫面」串起來的 AI 影片，讓寵物的每個動作都是連續的真實影片。

為什麼會真：
  1. 每個動作都是一整段連續影片（毛、呼吸、重心移動都是真的），不是照片加晃動。
  2. 所有片段的開頭／結尾都落在同一組「接點畫面」上，切換動作時看不到跳格。

接點畫面（hub）：從寵物照片挑最好的「趴」和「坐」各一張，放在同一個畫布、同一個比例。
需要生成的 4 段（首尾幀影片 AI）：
  sleep   趴 → 趴    趴著呼吸（循環）
  sit     坐 → 坐    坐著待機（循環）
  liesit  趴 → 坐    坐起來；倒放就是趴下
  walk    坐 → 空    站起來、走出畫面：前段是「站起來」（倒放是坐下），後段切出走路循環
所有片段保留畫布座標、共用同一個縮放比例和地板線，所以銜接時大小、位置完全一致。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT, load_pack, pack_dir

CANVAS = (832, 480)
BG = (236, 236, 236)
FLOOR = int(CANVAS[1] * 0.88)  # 地板線（寵物腳底）
FPS = 16
STYLE = ("static camera, locked-off tripod shot, plain light gray studio background, soft natural shadow, "
         "photorealistic, smooth realistic natural animal motion, same animal, same fur colors and markings, "
         "the whole body stays visible")
CLIPS = {
    # 名稱: (開始, 結束, 秒數, 提示詞, 中文)
    "sleep": ("lie", "lie", 4.0, "A {a} lying down calmly, sleeping, slow gentle breathing, ears twitching slightly, {s}",
              "趴著睡覺（循環）"),
    "sit": ("sit", "sit", 4.0, "A {a} sitting calmly, breathing, blinking, looking around, tail moving slightly, {s}",
            "坐著待機（循環）"),
    "liesit": ("lie", "sit", 3.0, "A {a} lying down slowly gets up and sits upright in place, {s}", "坐起來／趴下"),
    "walk": ("sit", "empty", 4.5, "A sitting {a} stands up and walks calmly out of the frame to the right, "
             "side view, natural walking gait, {s}", "站起來／走路"),
}


# ---------------------------------------------------------------- 接點畫面
def _hub_candidates(name: str, pose: str):
    pack = load_pack(name)
    d = pack_dir(name)
    out = []
    for im in pack["images"]:
        if im.get("frames") or im["pose"] != pose:
            continue
        img = Image.open(d / im["file"]).convert("RGBA")
        ar = img.width / img.height
        clip = im.get("clip") or {}
        # 形狀硬條件：趴著要橫的、坐著要直的（姿勢辨識偶爾會錯，接點畫面不能錯）
        ok = (1.25 < ar < 2.8) if pose == "lie" else (0.55 < ar < 0.95)
        s = im.get("score", 0.3) + clip.get(pose, 0.3) + (1.0 if ok else 0)
        out.append((s, img, im))
    out.sort(key=lambda c: -c[0])
    return out


def pick_hubs(name: str) -> dict:
    hubs = {}
    for pose in ("lie", "sit"):
        c = _hub_candidates(name, pose)
        if c:
            hubs[pose] = (c[0][1], c[0][2])
    return hubs


def compose(cut: Image.Image, facing: str, height_frac: float) -> Image.Image:
    """放在畫布中間、腳底在地板線、朝右。height_frac 決定大小（坐和趴各自設定，讓體型一致）。"""
    if facing == "left":
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    s = CANVAS[1] * height_frac / cut.height
    s = min(s, CANVAS[0] * 0.5 / cut.width)
    cut = cut.resize((max(1, int(cut.width * s)), max(1, int(cut.height * s))), Image.LANCZOS)
    bg = Image.new("RGBA", CANVAS, BG + (255,))
    bg.alpha_composite(cut, ((CANVAS[0] - cut.width) // 2, FLOOR - cut.height))
    return bg.convert("RGB")


def hub_frames(name: str, work: Path) -> dict[str, Path]:
    hubs = pick_hubs(name)
    if not hubs:
        raise RuntimeError("找不到合適的趴姿或坐姿照片當接點")
    # 趴著約佔畫面高 30%、坐著約 50%（同一隻貓兩種姿勢的大致比例）
    frac = {"lie": 0.30, "sit": 0.50}
    paths = {}
    for pose, (img, im) in hubs.items():
        p = work / f"hub_{pose}.png"
        compose(img, im.get("facing", "right"), frac[pose]).save(p)
        paths[pose] = p
    p = work / "hub_empty.png"
    Image.new("RGB", CANVAS, BG).save(p)
    paths["empty"] = p
    return paths


def plan(name: str) -> list[str]:
    """這隻寵物可以生成哪些片段（缺坐姿照片就做不了需要坐姿的片段）。"""
    hubs = pick_hubs(name)
    return [c for c, (a, b, *_rest) in CLIPS.items() if all(x == "empty" or x in hubs for x in (a, b))]


def done_clips(name: str) -> set:
    pack = load_pack(name)
    return {im.get("real") for im in pack["images"] if im.get("real")}


# ---------------------------------------------------------------- 影片 → 保留畫布座標的去背影格
def read_frames(video: Path) -> list[Image.Image]:
    import imageio_ffmpeg

    gen = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    meta = next(gen)
    w, h = meta["size"]
    frames = [Image.frombytes("RGB", (w, h), bytes(r)).resize(CANVAS, Image.LANCZOS) for r in gen]
    src = meta.get("fps") or FPS
    if abs(src - FPS) > 0.5 and len(frames) > 2:
        idx = np.linspace(0, len(frames) - 1, max(2, round(len(frames) * FPS / src))).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


def matte(frames, session, say=None) -> list[np.ndarray]:
    """逐格去背 + 前後格中位數平滑（去掉邊緣閃爍）。回傳 RGBA 陣列（畫布大小）。"""
    from rembg import remove

    from cutout import clean_alpha

    alphas = []
    for i, f in enumerate(frames):
        if say:
            say(f"逐格去背 {i + 1}/{len(frames)}")
        alphas.append(np.asarray(remove(f, session=session).getchannel("A")).astype(np.float32))
    out = []
    for i, f in enumerate(frames):
        a = np.median(np.stack(alphas[max(0, i - 1):i + 2]), axis=0) if len(alphas) > 2 else alphas[i]
        a = clean_alpha(a.astype(np.uint8))
        a[FLOOR + 6:] = 0  # 地板線以下只有影子／地面
        if (a > 128).sum() < a.size * 0.003:
            a[:] = 0  # 寵物不在畫面裡
        out.append(np.dstack([np.asarray(f), a]))
    return out


def _centroid_x(rgba):
    m = rgba[:, :, 3] > 128
    return float(np.where(m)[1].mean()) if m.any() else None


def _cut(frames, x_center=None, half_w=None, centroid=False):
    """裁成一段動畫格：地板線對齊；錨點=畫布中心（原地動作）或質心（走路循環）。"""
    tops = [np.where(f[:, :, 3] > 24)[0].min() for f in frames if (f[:, :, 3] > 24).any()]
    top = max(0, min(tops) - 4) if tops else 0
    bottom = FLOOR + 6
    out = []
    for f in frames:
        cx = _centroid_x(f) if centroid else x_center
        if cx is None:
            continue
        x0 = int(max(0, cx - half_w))
        x1 = int(min(CANVAS[0], cx + half_w))
        crop = f[top:bottom, x0:x1]
        out.append((Image.fromarray(crop, "RGBA"), float(cx - x0)))
    return out


def split_walk(frames):
    """「站起來走出去」→（站起來的過程, 走路循環）。用質心開始明顯移動的時間點切開。"""
    xs = [_centroid_x(f) for f in frames]
    valid = [i for i, x in enumerate(xs) if x is not None]
    if len(valid) < 8:
        return frames, []
    x0 = xs[valid[0]]
    start = next((i for i in valid if xs[i] - x0 > CANVAS[0] * 0.04), None)
    if start is None:
        return frames, []
    # 走到碰到畫面右緣之前
    end = start
    for i in valid:
        if i < start:
            continue
        m = frames[i][:, :, 3] > 128
        if np.where(m)[1].max() >= CANVAS[0] - 3:
            break
        end = i
    return frames[:start + 1], frames[start:end + 1]


def _best_loop(frames, min_len=6):
    from cutout import _best_loop as bl

    masks = []
    for im, _ in frames:
        m = Image.fromarray(((np.asarray(im)[:, :, 3] > 128) * 255).astype(np.uint8)).resize((32, 32))
        masks.append(np.asarray(m) > 127)
    i, j = bl(masks, min_len)
    return frames[i:j]


# ---------------------------------------------------------------- 存進寵物包
def save_unit(name: str, key: str, frames, pose: str, kind: str, to: str | None, facing="right", extra=None):
    pack = load_pack(name)
    d = pack_dir(name)
    # 同一個 key 的舊版本先移除
    keep = []
    for im in pack["images"]:
        if im.get("real") == key:
            for fr in im.get("frames", []):
                (d / fr["file"]).unlink(missing_ok=True)
            continue
        keep.append(im)
    pack["images"] = keep
    stamp = int(time.time())
    files = []
    for i, (im, ax) in enumerate(frames):
        fn = f"real_{key}_{stamp}_{i:03d}.png"
        im.save(d / fn, optimize=True)
        files.append({"file": fn, "ax": round(ax, 1)})
    entry = {"file": files[0]["file"], "frames": files, "fps": FPS, "pose": pose, "kind": kind,
             "facing": facing, "real": key, "ai": True, "area": 0, "hash": "0", "source": f"AI：{key}"}
    if to:
        entry["to"] = to
    entry.update(extra or {})
    pack["images"].append(entry)
    pack["real_scale"] = True  # 寫實片段共用同一個縮放比例
    (d / "pet.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")


def process_clip(name: str, clip: str, video: Path, session, say=None):
    frames = matte(read_frames(video), session, say)
    half = CANVAS[0] * 0.30
    cx = CANVAS[0] / 2
    if clip == "sleep":
        save_unit(name, "sleep", _cut(frames, cx, half), "lie", "loop", None)
    elif clip == "sit":
        save_unit(name, "sit", _cut(frames, cx, half), "sit", "loop", None)
    elif clip == "liesit":
        save_unit(name, "liesit", _cut(frames, cx, half), "lie", "trans", "sit")
    elif clip == "walk":
        stand, walk = split_walk(frames)
        save_unit(name, "situp", _cut(stand, cx, half), "sit", "trans", "walk")
        if len(walk) >= 8:
            xs = [x for x in (_centroid_x(f) for f in walk) if x is not None]
            speed = (xs[-1] - xs[0]) / max(len(walk) / FPS, 0.1)  # 影片裡實際的步行速度（畫布像素/秒）
            cyc = _best_loop(_cut(walk, half_w=half * 0.8, centroid=True))
            save_unit(name, "walk", cyc, "walk", "loop", None, extra={"px_per_s": round(speed, 1)})


def generate(name: str, backend, session, which=None, status=None, animal="cat"):
    """依序生成；每段做完就存（免費額度中途用完，已完成的不會不見）。回傳完成的片段。"""
    work = ROOT / ".cache" / "real" / name
    work.mkdir(parents=True, exist_ok=True)
    hubs = hub_frames(name, work)
    todo = which or [j.key for j in missing(name)]
    done = []
    for clip in todo:
        a, b, secs, prompt, label = CLIPS[clip]
        say = (lambda m, lb=label: status and status(f"{lb}：{m}"))
        say("AI 生成影片中…")
        vid = work / f"{clip}_{int(time.time())}.mp4"
        backend.generate(hubs[a], hubs[b], prompt.format(a=animal, s=STYLE), vid, say, secs=secs)
        process_clip(name, clip, vid, session, say)
        done.append(clip)
    return done


class Job:
    def __init__(self, clip):
        self.key, self.label = clip, CLIPS[clip][4]


def missing(name: str) -> list[Job]:
    done = done_clips(name)
    if "situp" in done:
        done.add("walk")
    return [Job(c) for c in plan(name) if c not in done]
