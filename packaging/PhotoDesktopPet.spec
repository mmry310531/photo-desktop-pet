# PyInstaller 打包設定：python -m PyInstaller packaging/PhotoDesktopPet.spec
# 產出 dist/PhotoDesktopPet/（資料夾版，啟動快；寵物、模型、設定都存在 exe 旁邊）
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("imageio_ffmpeg", "gradio_client", "rembg", "onnxruntime", "pillow_heif"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports = [h for h in hiddenimports if not h.startswith(("rembg.commands", "rembg.cli", "onnxruntime.quantization", "onnxruntime.tools"))]
hiddenimports += ["pymatting.alpha.estimate_alpha_cf", "pymatting.foreground.estimate_foreground_ml", "pymatting.util.util",
                  "pet", "trainer", "cutout", "aifill", "fastdl", "winenv", "common"]

a = Analysis(
    ["../app/launcher.py"],
    pathex=["stubs", "../app"],  # stubs：用空殼代替 pymatting（省掉 numba/llvmlite 約 170MB）
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["numba", "llvmlite", "pandas", "sklearn", "lxml", "rembg.commands", "rembg.cli",
              "onnxruntime.quantization", "onnxruntime.tools", "tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "torch", "tensorflow",
              "IPython", "jupyter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PhotoDesktopPet",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PhotoDesktopPet")
