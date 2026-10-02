"""AI 補齊動作：用寵物照片，讓 AI 生成「中間的動作影片」，再切成連續動畫。

原理：給影片 AI 一張「開始畫面」和一張「結束畫面」，它會生成中間的自然動作
（例如 開始=坐著、結束=趴著 → 生成「趴下」的過程）。開始和結束用同一張，
就是一段可以無縫循環的待機／走路動作。生成的影片再用跟真實影片一樣的流程
（逐格去背）變成桌寵的動畫。

兩種 AI 來源：
  - Hugging Face（免費）：每個帳號每天有免費 GPU 額度（約幾分鐘），用公開的 Wan 2.2 首尾幀 Space
  - fal.ai（付費）：Wan 2.2，480p 每秒影片 US$0.04，一段約 US$0.1，最快最穩
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT, load_pack, pack_dir

HF_SPACE_DEFAULT = "multimodalart/wan-2-2-first-last-frame"
FAL_ENDPOINT = "fal-ai/wan/v2.2-a14b/image-to-video"
FAL_PRICE_PER_SEC = 0.04  # 480p
CLIP_SECS = 2.5
CANVAS = (832, 480)
BG = (236, 236, 236)

SPECIES = {"貓": "cat", "狗": "dog", "鳥": "bird", "馬": "horse", "羊": "sheep", "牛": "cow",
           "熊": "bear", "毛玩偶": "fluffy pet"}

STYLE = ("static camera, locked-off shot, the whole body always fully visible, plain light gray studio background, "
         "realistic natural animal motion, same animal, same fur colors and markings")
NEGATIVE = ("camera movement, zoom, pan, cut, extra animals, extra legs, deformed, blurry, text, watermark, "
            "different animal, color change, background change")


@dataclass
class Job:
    key: str          # 例如 loop:walk、trans:walk>sit
    kind: str         # loop / trans
    pose: str         # 開始姿勢
    to: str           # 結束姿勢
    label: str        # 給人看的名稱
    prompt: str


def _animal(pack) -> str:
    votes = [im.get("species") for im in pack["images"] if im.get("species")]
    return max(set(votes), key=votes.count) if votes else "pet"


def pack_entries(pack) -> list[dict]:
    return [{"pose": im["pose"], "kind": im.get("kind") or ("loop" if im.get("frames") else None),
             "to": im.get("to"), "animated": bool(im.get("frames"))} for im in pack["images"]]


def missing_jobs(name: str) -> list[Job]:
    """看這隻寵物還缺哪些連續動作，而且有照片可以讓 AI 補。"""
    import spec

    pack = load_pack(name)
    a = _animal(pack)
    prompts = {
        "loop:walk": ("走路（循環）", f"A {a} walking in place on a treadmill, side view, legs moving in a smooth natural "
                                      f"walking cycle, the {a} stays in the center of the frame, {STYLE}"),
        "loop:sit": ("坐著待機（循環）", f"A {a} sitting calmly, breathing, blinking, small head movements, "
                                        f"tail gently moving, {STYLE}"),
        "loop:lie": ("趴著睡覺（循環）", f"A {a} lying down asleep, slow gentle breathing, eyes closed, {STYLE}"),
        "trans:walk>sit": ("坐下／站起來（轉場）", f"A standing {a} slowly sits down in place, {STYLE}"),
        "trans:sit>lie": ("趴下／起身（轉場）", f"A sitting {a} slowly lies down in place, {STYLE}"),
    }
    jobs = []
    for code in spec.ai_fillable(spec.coverage(pack_entries(pack))):
        kind, rest = code.split(":")
        poses = rest.split(">")
        label, prompt = prompts[code]
        jobs.append(Job(code, kind, poses[0], poses[-1], label, prompt))
    return jobs


# ---------------------------------------------------------------- 準備開始／結束畫面
def _ref(pack, d: Path, pose: str):
    """挑一張最有代表性的姿勢圖（靜止照片優先，其次是影片的某一格）。"""
    stills = [im for im in pack["images"] if im["pose"] == pose and not im.get("frames")]
    if stills:
        areas = [im.get("area", 0) for im in stills]
        im = stills[int(np.argsort(areas)[len(areas) // 2])]  # 面積中位數那張，避免極端值
        return Image.open(d / im["file"]).convert("RGBA"), im.get("facing", "right")
    for im in pack["images"]:
        fr = im.get("frames")
        if not fr:
            continue
        if im["pose"] == pose:
            return Image.open(d / fr[0]["file"]).convert("RGBA"), im.get("facing", "right")
        if im.get("to") == pose:
            return Image.open(d / fr[-1]["file"]).convert("RGBA"), im.get("facing", "right")
    return None, None


def _compose(cut: Image.Image, facing: str, area_target: float) -> Image.Image:
    """把去背的寵物放到素色背景中央，統一朝右、統一大小、腳底對齊。"""
    if facing == "left":
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    area = (np.asarray(cut.getchannel("A")) > 128).sum()
    s = (area_target / max(area, 1)) ** 0.5
    s = min(s, CANVAS[1] * 0.78 / cut.height, CANVAS[0] * 0.7 / cut.width)
    cut = cut.resize((max(1, int(cut.width * s)), max(1, int(cut.height * s))), Image.LANCZOS)
    bg = Image.new("RGBA", CANVAS, BG + (255,))
    x = (CANVAS[0] - cut.width) // 2
    y = int(CANVAS[1] * 0.9) - cut.height
    bg.alpha_composite(cut, (x, max(0, y)))
    return bg.convert("RGB")


def build_frames(name: str, job: Job) -> tuple[Image.Image, Image.Image]:
    pack = load_pack(name)
    d = pack_dir(name)
    a, fa = _ref(pack, d, job.pose)
    b, fb = _ref(pack, d, job.to)
    if a is None or b is None:
        raise RuntimeError(f"缺少「{job.pose}」或「{job.to}」姿勢的照片")
    area_target = CANVAS[0] * CANVAS[1] * 0.12
    return _compose(a, fa, area_target), _compose(b, fb, area_target)


# ---------------------------------------------------------------- 兩種 AI 來源
class HuggingFace:
    """免費：Hugging Face ZeroGPU Space（每個帳號每天有免費 GPU 額度）。"""

    def __init__(self, token: str = "", space: str = HF_SPACE_DEFAULT):
        self.token = token.strip() or None
        self.space = space.strip() or HF_SPACE_DEFAULT
        self._client = None

    def client(self):
        if self._client is None:
            from gradio_client import Client

            dl = str(ROOT / ".cache" / "gradio")
            try:
                self._client = Client(self.space, token=self.token, download_files=dl, verbose=False)
            except TypeError:  # 舊版 gradio_client 參數名稱不同
                self._client = Client(self.space, hf_token=self.token, download_files=dl, verbose=False)
        return self._client

    def generate(self, start: Path, end: Path, prompt: str, out: Path, status=None, secs: float = CLIP_SECS) -> Path:
        from gradio_client import handle_file

        if status:
            status("排隊等 Hugging Face 免費 GPU（尖峰時段可能要等幾分鐘）…")
        try:
            res = self.client().predict(
                handle_file(str(start)), handle_file(str(end)), prompt, NEGATIVE,
                secs, 8, 1, 1, 42, True, api_name="/generate_video")
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "GPU" in msg:
                raise RuntimeError("Hugging Face 今天的免費 GPU 額度用完了（每天會重置）。"
                                   "可以明天再按一次繼續，或改用 fal.ai。\n原始訊息：" + msg[:300])
            raise
        video = res[0] if isinstance(res, (list, tuple)) else res
        if isinstance(video, dict):
            video = video.get("video") or video.get("path") or video.get("url")
        src = Path(video)
        out.write_bytes(src.read_bytes())
        return out


class Fal:
    """付費：fal.ai（Wan 2.2，480p 每秒 US$0.04）。"""

    def __init__(self, key: str):
        self.key = key.strip()

    @staticmethod
    def cost(n_clips: int) -> float:
        return n_clips * CLIP_SECS * FAL_PRICE_PER_SEC

    def _req(self, url, data=None, method=None):
        req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                     method=method or ("POST" if data is not None else "GET"),
                                     headers={"Authorization": f"Key {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _data_uri(p: Path) -> str:
        buf = io.BytesIO()
        Image.open(p).convert("RGB").save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, start: Path, end: Path, prompt: str, out: Path, status=None, secs: float = CLIP_SECS) -> Path:
        body = {
            "image_url": self._data_uri(start), "end_image_url": self._data_uri(end),
            "prompt": prompt, "negative_prompt": NEGATIVE, "resolution": "480p",
            "num_frames": min(161, int(secs * 16) + 1), "frames_per_second": 16, "aspect_ratio": "16:9",
        }
        sub = self._req(f"https://queue.fal.run/{FAL_ENDPOINT}", body)
        t0 = time.time()
        while True:
            st = self._req(sub["status_url"])
            s = st.get("status")
            if status:
                status(f"fal.ai 生成中（{s}，已 {int(time.time() - t0)} 秒）…")
            if s == "COMPLETED":
                break
            if time.time() - t0 > 900:
                raise RuntimeError("fal.ai 超過 15 分鐘沒完成")
            time.sleep(3)
        res = self._req(sub["response_url"])
        url = res["video"]["url"]
        import fastdl

        fastdl.download(url, out, label="AI 影片")
        return out


# ---------------------------------------------------------------- 主流程
def run_job(name: str, job: Job, backend, session, detector, status=None):
    """生成一段 → 逐格去背 → 回傳 CutResult（還沒存檔，給訓練器檢查）。"""
    from cutout import process_video

    work = ROOT / ".cache" / "ai" / name
    work.mkdir(parents=True, exist_ok=True)
    a, b = build_frames(name, job)
    sa, sb = work / f"{job.key.replace(':', '_').replace('>', '-')}_start.png", work / f"{job.key.replace(':', '_').replace('>', '-')}_end.png"
    a.save(sa)
    b.save(sb)
    vid = work / (sa.stem.replace("_start", "") + f"_{int(time.time())}.mp4")
    backend.generate(sa, sb, job.prompt, vid, status)
    if status:
        status(f"AI 影片完成，逐格去背中（{job.label}）…")
    r = process_video(vid, session, detector, progress=status,
                      force={"kind": job.kind, "from": job.pose, "to": job.to, "facing": "right",
                             "seamless": job.kind == "loop"})
    r.extra["ai"] = True
    r.source = f"AI：{job.label}"
    return r
