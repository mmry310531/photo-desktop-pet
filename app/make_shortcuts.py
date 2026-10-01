"""在 Windows 桌面建立「桌面寵物」「寵物訓練器」捷徑（安裝程式會呼叫）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYW = Path(sys.executable).with_name("pythonw.exe")


def ps_quote(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


lines = ["$d=[Environment]::GetFolderPath('Desktop')", "$w=New-Object -ComObject WScript.Shell"]
for title, script in (("桌面寵物", "pet.py"), ("寵物訓練器", "trainer.py")):
    lines += [
        f"$s=$w.CreateShortcut((Join-Path $d {ps_quote(title + '.lnk')}))",
        f"$s.TargetPath={ps_quote(PYW)}",
        f"$s.Arguments={ps_quote(chr(34) + str(ROOT / 'app' / script) + chr(34))}",
        f"$s.WorkingDirectory={ps_quote(ROOT)}",
        "$s.Save()",
    ]
r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "; ".join(lines)])
print("桌面捷徑已建立" if r.returncode == 0 else "（建立桌面捷徑失敗，可以直接用資料夾裡的 .bat 啟動）")
