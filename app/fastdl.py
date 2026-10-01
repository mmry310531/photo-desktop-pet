"""多線程分段下載器（只用 Python 標準函式庫，安裝套件前就能跑）。

為什麼需要：有些網路環境「單一連線」被限速（例如到 PyPI 的 CDN 只剩 50KB/s），
但總頻寬其實很大。把一個檔案切成好幾段、同時開多條連線下載，速度就能疊加上去
——跟 IDM、aria2 的原理一樣。

用法：
  python fastdl.py install         分段下載所有 Python 套件並安裝（安裝程式用）
  python fastdl.py models          預先下載 AI 模型
  python fastdl.py URL 目的檔      下載單一檔案
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "photo-desktop-pet/1.0 (pip-compatible)"}
SEGMENTS = 8
MIN_SEG = 2 * 1024 * 1024  # 小於 2MB 的段不再切

INDEXES = [
    "https://pypi.org/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://repo.huaweicloud.com/repository/pypi/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
]

BIREFNET = ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
            "u2net/birefnet-general.onnx", "7a35a0141cbbc80de11d9c9a28f52697")
MODELS = [
    # (網址, 存放位置（相對 models/）, md5)
    ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
     "u2net/isnet-general-use.onnx", "fc16ebd8b0c10d971d3513d564d01e29"),
    ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
     "yolox_s.onnx", None),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


class Progress:
    def __init__(self, total: int, label: str):
        self.total, self.label, self.done = total, label, 0
        self.t0 = self.last = time.time()
        self.lock = threading.Lock()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last > 0.5 or self.done >= self.total:
                self.last = now
                sp = self.done / max(now - self.t0, 0.01)
                pct = self.done * 100 / max(self.total, 1)
                sys.stdout.write(f"\r  {self.label[:42]:<42} {pct:5.1f}%  {self.done / 1e6:7.1f}/{self.total / 1e6:.1f} MB"
                                 f"  {sp / 1e6:5.1f} MB/s   ")
                sys.stdout.flush()


def _probe(url: str) -> tuple[str, int, bool]:
    """回傳 (跟隨轉址後的最終網址, 檔案大小, 是否支援分段)。"""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range")
        if r.status == 206 and cr and "/" in cr:
            return final, int(cr.split("/")[-1]), True
        return final, int(r.headers.get("Content-Length") or 0), False


def _get_range(url: str, start: int, end: int, f, lock, prog: Progress | None) -> None:
    pos = start
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                while pos <= end:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    with lock:
                        f.seek(pos)
                        f.write(chunk)
                    pos += len(chunk)
                    if prog:
                        prog.add(len(chunk))
            if pos > end:
                return
        except Exception:
            time.sleep(1 + attempt)
    raise IOError(f"分段下載失敗：{url} bytes {start}-{end}")


def download(url: str, dest: Path, segments: int = SEGMENTS, md5: str | None = None,
             sha256: str | None = None, label: str | None = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest.name
    if dest.exists() and _verify(dest, md5, sha256):
        _log(f"  {label}：已下載過，略過")
        return dest
    final, size, ranged = _probe(url)
    tmp = dest.with_name(dest.name + ".part")
    prog = Progress(size, label)
    if not ranged or size < MIN_SEG:
        with urllib.request.urlopen(urllib.request.Request(final, headers=UA), timeout=30) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                prog.add(len(chunk))
    else:
        n = max(1, min(segments, size // MIN_SEG))
        bounds = [(i * size // n, (i + 1) * size // n - 1) for i in range(n)]
        with open(tmp, "wb") as f:
            f.truncate(size)
            lock = threading.Lock()
            with ThreadPoolExecutor(n) as ex:
                futs = [ex.submit(_get_range, final, a, b, f, lock, prog) for a, b in bounds]
                for fu in futs:
                    fu.result()
    sys.stdout.write("\n")
    if not _verify(tmp, md5, sha256):
        tmp.unlink(missing_ok=True)
        raise IOError(f"{label} 檔案校驗失敗（下載不完整），請重試")
    tmp.replace(dest)
    return dest


def _verify(p: Path, md5: str | None, sha256: str | None) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    if not md5 and not sha256:
        return True
    h = hashlib.md5() if md5 else hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == (md5 or sha256)


# ---------------------------------------------------------------- 選最快的套件來源
def _wheel_url(index: str, project: str = "pyside6-essentials") -> str | None:
    req = urllib.request.Request(f"{index}/{project}/", headers=UA)
    with urllib.request.urlopen(req, timeout=6) as r:
        html = r.read().decode("utf-8", "ignore")
    links = re.findall(r'href="([^"]+?\.whl)[^"]*"', html)
    links = [l for l in links if "win_amd64" in l] or links
    if not links:
        return None
    return urllib.request.urljoin(f"{index}/{project}/", links[-1])


def _speed(index: str, secs: float = 3.5) -> float:
    """實際下載一段真正的套件檔（不是只抓目錄頁）來量速度，用 4 條連線。"""
    try:
        url = _wheel_url(index)
        if not url:
            return 0.0
        got = [0]
        stop = time.time() + secs

        def worker(i):
            start = i * 8 * 1024 * 1024
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{start + 8 * 1024 * 1024}"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    while time.time() < stop:
                        c = r.read(65536)
                        if not c:
                            break
                        got[0] += len(c)
            except Exception:
                pass

        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(worker, range(4)))
        return got[0] / max(time.time() - t0, 0.1)
    except Exception:
        return 0.0


def pick_index() -> str:
    _log("測試各套件來源的實際下載速度（約 20 秒）…")
    res = {}
    for i in INDEXES:
        res[i] = _speed(i)
        _log(f"  {res[i] / 1e6:6.2f} MB/s  {i}")
    best = max(res, key=res.get)
    if res[INDEXES[0]] >= res[best] * 0.6:  # 官方不差太多就用官方
        best = INDEXES[0]
    _log(f"→ 使用 {best}")
    return best


# ---------------------------------------------------------------- 安裝套件
def install(requirements: Path = ROOT / "requirements.txt") -> None:
    py = sys.executable
    index = pick_index()
    wheels = ROOT / ".cache" / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    report = wheels / "report.json"
    _log("分析需要哪些套件…")
    subprocess.check_call([py, "-m", "pip", "install", "--dry-run", "--quiet", "-i", index, "-r", str(requirements), "--report", str(report)])
    items = json.loads(report.read_text(encoding="utf-8"))["install"]
    todo = []
    for it in items:
        di = it["download_info"]
        url = di["url"]
        hashes = di.get("archive_info", {}).get("hashes", {})
        name = url.split("/")[-1].split("#")[0]
        todo.append((url, wheels / name, hashes.get("sha256")))
    if not todo:
        _log("套件都已安裝")
        return
    _log(f"共 {len(todo)} 個套件，分段平行下載中…")
    errors = []

    def one(t):
        url, dest, sha = t
        try:
            download(url, dest, sha256=sha)
        except Exception as e:
            errors.append(f"{dest.name}: {e}")

    for t in todo:  # 一次一個檔案、每檔 8 條連線
        one(t)
    if errors:
        _log("下列檔案下載失敗，改用 pip 直接下載：\n  " + "\n  ".join(errors))
    _log("安裝中…")
    r = subprocess.call([py, "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
                         "-r", str(requirements)])
    if r != 0:  # 有檔案沒下到：剩下的交給 pip 從網路補
        subprocess.check_call([py, "-m", "pip", "install", "--find-links", str(wheels), "-i", index,
                               "--timeout", "60", "--retries", "8", "-r", str(requirements)])
    _log("套件安裝完成")


def models() -> None:
    _log("預先下載 AI 模型（之後訓練就不用等）…")
    for url, rel, md5 in MODELS:
        try:
            download(url, ROOT / "models" / rel, md5=md5)
        except Exception as e:
            _log(f"  模型下載失敗（第一次訓練時會再試）：{e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["install"]:
        install()
    elif args[:1] == ["models"]:
        models()
    elif len(args) == 2:
        download(args[0], Path(args[1]))
    else:
        print(__doc__)
