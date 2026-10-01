"""共用：路徑、寵物包讀寫、設定檔。"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PETS_DIR = ROOT / "pets"
SETTINGS_FILE = ROOT / "settings.json"

# 姿勢代碼 -> 顯示名稱
POSES = {
    "walk": "站／走",
    "sit": "坐",
    "lie": "躺／睡",
}

DEFAULT_SETTINGS = {
    "size": 1.0,            # 整體縮放
    "base_px": 130,         # 寵物在螢幕上的基準大小（邏輯像素）
    "active_pets": [],      # 啟動時要召喚的寵物包名稱
    "climb_windows": True,  # 能不能跳上視窗頂端
    "quiet": False,         # 安靜模式：不亂跑
}


def safe_name(name: str) -> str:
    name = name.strip() or "我的寵物"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)[:40]


def pack_dir(name: str) -> Path:
    return PETS_DIR / safe_name(name)


def load_pack(name: str) -> dict:
    d = pack_dir(name)
    f = d / "pet.json"
    if not f.exists():
        return {"name": name, "images": [], "phrases": ["喵～", "♪", "…zz", "!"]}
    data = json.loads(f.read_text(encoding="utf-8"))
    data.setdefault("images", [])
    data.setdefault("phrases", ["喵～", "♪", "!"])
    # 濾掉已不存在的檔案
    data["images"] = [im for im in data["images"] if (d / im["file"]).exists()]
    return data


def save_pack(name: str, data: dict) -> Path:
    d = pack_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pet.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def list_packs() -> list[str]:
    if not PETS_DIR.exists():
        return []
    out = []
    for p in sorted(PETS_DIR.iterdir()):
        if (p / "pet.json").exists():
            try:
                if load_pack(p.name)["images"]:
                    out.append(p.name)
            except Exception:
                pass
    return out


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def pythonw() -> str:
    """回傳不跳黑色主控台視窗的 python 執行檔（Windows 用 pythonw.exe）。"""
    exe = Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


# ---------------------------------------------------------------- 匯出／匯入寵物（.petpack = zip）
PACK_EXT = ".petpack"


def export_pack(name: str, dest) -> Path:
    import zipfile

    d = pack_dir(name)
    data = load_pack(name)
    files = {"pet.json"}
    for im in data["images"]:
        files.add(im["file"])
        files.update(f["file"] for f in im.get("frames", []))
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            if (d / f).exists():
                z.write(d / f, f)
    return dest


def import_pack(path) -> str:
    """匯入 .petpack，回傳寵物名稱（同名會自動改名）。只接受 pet.json 和圖片，防止惡意壓縮檔。"""
    import zipfile

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "pet.json" not in names:
            raise ValueError("這不是寵物包（找不到 pet.json）")
        data = json.loads(z.read("pet.json").decode("utf-8"))
        base = safe_name(data.get("name") or Path(path).stem)
        name, k = base, 2
        while pack_dir(name).exists():
            name, k = f"{base} ({k})", k + 1
        d = pack_dir(name)
        d.mkdir(parents=True)
        total = 0
        for n in names:
            if n == "pet.json":
                continue
            if "/" in n or "\\" in n or ".." in n or not n.lower().endswith(".png"):
                continue  # 只收同一層的 png
            info = z.getinfo(n)
            total += info.file_size
            if total > 500 * 1024 * 1024:
                raise ValueError("寵物包太大（超過 500MB）")
            (d / n).write_bytes(z.read(n))
        def ok(fn):
            return isinstance(fn, str) and "/" not in fn and "\\" not in fn and ".." not in fn
        data["images"] = [im for im in data.get("images", []) if ok(im.get("file"))
                          and all(ok(f.get("file")) for f in im.get("frames", []))]
        data["name"] = name
        save_pack(name, data)
    if not load_pack(name)["images"]:
        raise ValueError("寵物包裡沒有可用的圖片")
    return name
