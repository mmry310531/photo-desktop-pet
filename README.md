# 照片桌寵 Photo Desktop Pet 🐾

把你家寵物的照片「餵」給它，牠就會活在你的 Windows 桌面上。

- 在工作列上散步、坐著發呆、累了趴下睡覺（頭上冒 z）
- **會跳上其他視窗的頂端**，視窗拖走牠會跟著移動，視窗關掉牠就掉下來
- 走到視窗邊緣有時會直接走下去，摔得重還會彈一下
- 用滑鼠**拎起來丟出去**、點牠會冒愛心、滑鼠在牠身上來回摸會呼嚕
- 你離開電腦太久牠會睡著，你一回來牠會醒來說「你回來了！」
- 可以同時養好幾隻、調整大小、安靜模式、開機自動啟動
- 全部在你自己的電腦上跑，**照片不會上傳到任何地方**

## 安裝（Windows 10 / 11）

1. 右上角綠色 **Code → Download ZIP**，解壓縮到任何資料夾（路徑盡量不要有空格以外的奇怪符號）
2. 雙擊 **`1-安裝.bat`**
   - 沒有 Python 會自動用 winget 裝 Python 3.12
   - 接著自動挑最快的套件來源並安裝（約 250MB），完成後桌面會出現「寵物訓練器」「桌面寵物」兩個捷徑
3. 訓練器會自動打開 → 把寵物照片（或整個資料夾）拖進去 → 按 **開始訓練**
4. 檢查結果，按 **儲存並召喚到桌面** 🐾

之後想再餵新照片，打開「寵物訓練器」選同一個名字繼續加就好，按儲存後桌上那隻會即時更新。

## 「訓練」實際在做什麼

每張照片會經過：

1. **找動物**：YOLOX 物件偵測先找出照片裡的貓／狗在哪（旁邊的人、腳踏車、狗屋不會被剪進來）
2. **去背**：IS-Net（勾選「高品質毛邊」改用 BiRefNet）把寵物從背景剪下來
3. **分類姿勢**：依外形判斷是「坐」「站／走」還是「躺／睡」，並猜牠頭朝哪邊
4. **整理**：去掉重複照片、過濾太近的特寫或去背失敗的照片、統一大小與亮度

桌面上的動作（走路的晃動、呼吸、被拎起來甩、落地壓扁）是程式即時算出來的，所以**照片越多、姿勢越多樣，牠就越生動**。

## 拍照建議

| ✅ 好照片 | ❌ 不好的照片 |
|---|---|
| 拍到全身（含腳） | 只有臉的特寫 |
| 側面站著 → 用來走路 | 被家具、人擋住一半 |
| 坐著、趴著、睡覺各來幾張 | 很暗、很模糊 |
| 背景單純一點更好 | 寵物和背景顏色幾乎一樣 |

每種姿勢有 3～10 張最理想。判斷錯的話在訓練器裡**右鍵**就能改姿勢、翻轉面向或刪除。

## 操作

| 動作 | 效果 |
|---|---|
| 左鍵點牠 | 冒愛心、說話；睡著時會被叫醒 |
| 拖曳 | 拎起來，放開會照你甩的方向飛出去 |
| 滑鼠在牠身上來回移動 | 摸摸，會呼嚕 |
| 雙擊 | 跳一下 |
| 右鍵（或系統匣圖示） | 叫牠過來、讓牠睡、再召喚一隻、大小、安靜模式、開機啟動、結束 |
| 點系統匣圖示 | 所有寵物跑到滑鼠旁邊 |

想改牠會說的話：編輯 `pets/<名字>/pet.json` 裡的 `"phrases"`。

## 常見問題

- **第一次訓練很慢**：要下載去背模型（約 170MB）和偵測模型（約 36MB），之後就快了。一張照片在一般筆電上約 1～3 秒。
- **兔子、倉鼠、鳥等**：偵測模型認得的動物有限，認不出來時會改用整張去背並標 ⚠，請自己檢查一下。
- **牠倒退著走**：訓練器裡對那張照片右鍵 →「左右翻轉面向」。
- **不想讓牠爬到視窗上**：右鍵 → 取消勾選「可以跳到視窗上」。
- **手動用指令訓練**：`.venv\Scripts\python app\trainer.py --cli 名字 照片資料夾 [--hq]`

## 專案結構

```
app/trainer.py   訓練器（GUI + 指令列）
app/cutout.py    偵測、去背、姿勢分類
app/pet.py       桌面寵物本體（物理、行為、互動、系統匣）
app/winenv.py    Windows 視窗頂邊與閒置時間偵測
pets/            你訓練好的寵物（不會上傳到 Git）
```

macOS / Linux 也能跑（`pip install -r requirements.txt` 後執行 `python app/trainer.py`），但跳上視窗與閒置偵測只支援 Windows。

---

## English

Feed photos of your pet and it comes to life on your Windows desktop: it walks along the taskbar, sits, naps, jumps onto other windows (and rides along when you move them), can be picked up and thrown, and purrs when you pet it with the mouse. Everything runs locally.

**Install:** download the ZIP, run `1-安裝.bat` (installs Python 3.12 via winget if needed), drag photos into the trainer, click *Start*, then *Save & summon*.

**Pipeline:** YOLOX detection → IS-Net / BiRefNet background removal → shape-based pose classification (sit / walk / lie) and facing → dedupe & size/brightness normalization. Motion is procedural.

## 授權 / Credits

本專案 MIT License。使用的開源元件：
[rembg](https://github.com/danielgatis/rembg)（MIT）、
[IS-Net / DIS](https://github.com/xuebinqin/DIS)（Apache-2.0）、
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)（MIT）、
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0）、
[PySide6 / Qt](https://www.qt.io/qt-for-python)（LGPL-3.0）。
