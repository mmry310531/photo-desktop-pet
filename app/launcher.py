"""exe 版的入口：同一個 PhotoDesktopPet.exe 依參數開啟桌面寵物或訓練器。

  PhotoDesktopPet.exe              → 有寵物就放到桌面，沒有就打開訓練器
  PhotoDesktopPet.exe --trainer    → 訓練器
  PhotoDesktopPet.exe 某隻.petpack  → 打開訓練器並匯入
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def selftest() -> int:
    """打包後的自我檢查（CI 用）：所有模組都載得進來、ffmpeg 在、Qt 能建立視窗。"""
    out = []
    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import importlib

        for m in ("aifill", "cutout", "fastdl", "pet", "spec", "trainer", "winenv", "rembg"):
            importlib.import_module(m)
        import common
        import imageio_ffmpeg
        import onnxruntime
        from PySide6.QtWidgets import QApplication

        out.append("ffmpeg=" + imageio_ffmpeg.get_ffmpeg_exe())
        out.append("onnxruntime=" + onnxruntime.__version__)
        app = QApplication([])
        out.append("qt=ok " + str(app.platformName()))
        out.append("root=" + str(common.ROOT))
        ok = True
    except Exception as e:
        import traceback

        out.append("FAIL " + "".join(traceback.format_exception(e)))
        ok = False
    Path(sys.executable).with_name("selftest.txt").write_text("\n".join(out), encoding="utf-8")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args[:1] == ["--selftest"]:
        sys.exit(selftest())
    which = "pet"
    if args and args[0] in ("--pet", "--trainer"):
        which = args.pop(0)[2:]
    packs = [a for a in args if a.lower().endswith(".petpack")]
    if packs:
        which = "trainer"
    sys.argv = [sys.argv[0]] + args
    if which == "trainer":
        import trainer

        if args[:1] == ["--cli"]:
            trainer.run_cli(args[1:])
        else:
            if packs:
                from common import import_pack

                for p in packs:
                    try:
                        import_pack(p)
                    except Exception as e:
                        print("匯入失敗", p, e)
            trainer.run_gui()
    else:
        import pet

        pet.main()


if __name__ == "__main__":
    main()
