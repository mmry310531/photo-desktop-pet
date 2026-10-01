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
