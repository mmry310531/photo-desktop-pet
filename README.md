# 照片桌寵 Photo Desktop Pet 🐾

把你家寵物的照片和影片交給這個程式，牠就會變成一隻住在 Windows 桌面上的小動物。

- 在工作列上走來走去、坐著發呆、累了趴下睡覺
- 會跳到其他視窗上面；你拖動視窗，牠會跟著移動
- 可以用滑鼠把牠拎起來丟出去、點牠、摸牠
- 你離開電腦太久牠會睡著，你回來牠會醒來
- 全部在你自己的電腦上處理，**照片不會上傳到任何地方**（除非你自己選擇用線上 AI 補動作）

> **先說清楚目前的程度**
> 程式可以自動做完整個流程，但「看起來像不像真的」幾乎完全取決於你給的素材。
> 隨手拍的生活照和手持影片，常會出現「動作不連貫」或「身體缺一塊」的情況。
> 原因和解決方法寫在下面〈四、為什麼有時候看起來不自然〉，建議先讀一下再開始拍。

---

## 一、快速開始

1. 下載：在這個網頁右上角按綠色的 **Code → Download ZIP**，解壓縮到任何資料夾。
2. 安裝：雙擊 **`1-安裝.bat`**。
   第一次會自動安裝需要的東西（約 580MB），完成後桌面會出現「寵物訓練器」和「桌面寵物」兩個捷徑。
3. 打開「寵物訓練器」→ 輸入寵物名字 → 把照片、影片（或整個資料夾）拖進視窗 → 按 **一鍵產出 ▶**。
4. 處理完，寵物就會出現在你的桌面上。

需要：Windows 10 或 11，記憶體建議 8GB 以上。

---

## 二、要準備什麼素材？（最重要的一段）

### 好素材和壞素材

| ✅ 好素材 | ❌ 壞素材 | 為什麼 |
|---|---|---|
| 整隻都在畫面裡，連腳和尾巴 | 身體有一部分在照片外 | 照片外的部分補不回來，會像被切掉一塊 |
| 從側面、跟寵物差不多高的角度拍 | 從正上方往下拍 | 桌面上的寵物是「側面看」，俯拍的照片放上去會很怪 |
| 背景單純（地板、牆壁） | 窩在棉被、衣服、紙箱裡 | 跟寵物貼在一起的布料很難分開，容易一起被剪進來 |
| 清楚、光線夠 | 很暗、晃到 | 毛的邊緣會糊掉 |
| 同一時期拍的 | 小時候和長大後混在一起 | 換姿勢時會像換了一隻 |

### 照片：要幾張？

**10～20 張「好照片」就夠了，比 100 張普通照片有用。** 三種姿勢各 3～6 張：

| 姿勢 | 拍什麼 |
|---|---|
| 站／走 | 側面站著，四隻腳都拍到 |
| 坐 | 坐著（正面或側面都可以） |
| 趴／睡 | 趴著、蜷成一團睡覺 |

可以一次丟 30～50 張進去，程式會自動丟掉不合格的（太近的特寫、身體被切到、模糊、重複）。
用一般生活照實測，大約會剩下三分之一到一半。

### 影片：怎麼拍最好？

照片只能做出「一張圖在晃」；**影片才能讓牠真的動腳走路、坐下、趴下**。最理想的拍法：

- 手機**橫拿、放著不動**（或靠在東西上），不要跟著寵物移動
- **從側面拍**，寵物從畫面一邊走到另一邊
- 整隻一直在畫面裡，尾巴也要
- 每段 3～5 秒就夠，一段只拍一個動作

建議拍這幾段：

| 影片內容 | 變成桌面上的什麼 |
|---|---|
| 側面走過畫面 | 走路（最重要） |
| 站著 → 坐下 | 坐下、站起來 |
| 坐著 → 趴下 | 趴下、起身 |
| 坐著不動、搖尾巴 | 坐著發呆 |
| 趴著或睡覺 | 睡覺時的呼吸 |

隨手拍的手持影片也能用：程式會自己從影片裡找出「姿勢穩定、整隻都在畫面裡」的片段（一段影片最多挑 4 段），但能用的部分通常很少、很短。

---

## 三、程式會怎麼處理你的素材

按下「一鍵產出」後，程式會依序做這幾件事，全部自動：

1. **找到寵物在哪**：先認出畫面中哪裡是貓或狗，旁邊的人和雜物不會被剪進來。
2. **把寵物剪下來（去背）**：兩個 AI 模型合作，一個負責毛的細邊，一個負責「整隻動物的輪廓」，避免黑毛貼著深色背景時被挖掉。
3. **自動品管**：身體被照片切到、太近的特寫、剪壞的、重複的，直接丟掉。影片裡剪壞的那幾格也會丟掉，只留最完整的連續一段。
4. **判斷姿勢**：站、坐、趴，還有頭朝哪邊。
5. **統一大小和亮度**，再放到桌面上。

判斷錯的地方，可以在訓練器裡對那張圖**按右鍵**改姿勢、翻轉方向或刪除。

處理時間（一般電腦）：照片每張約 3～10 秒；影片每一格約 5～10 秒，一段影片約 3～8 分鐘。
嫌慢可以勾「**快速模式**」，影片會快很多，但偶爾會缺頭缺腳。

---

## 四、為什麼有時候看起來不自然？

我們拿真實的家貓照片和手機影片實際測試過，遇到的問題整理如下。

### 1. 為什麼會「破圖」（身體缺一塊、帶著棉被）

破圖分兩種：

| 種類 | 例子 | 能不能修 |
|---|---|---|
| 程式判斷錯 | 黑毛貼著深色背景被挖掉、圈錯地方 | 大部分已經修好 |
| 素材本身就不完整 | 貓的一半在照片外、貓埋在棉被裡 | **修不了**，照片裡沒有的部分，沒有任何方法能補回來 |

第二種只能靠「不要用這種照片」。程式會自動丟掉身體被切到的照片，但「貓跟棉被黏在一起」還是可能漏網，請在訓練器裡手動刪掉。

### 2. 為什麼動作不自然

就算每張都剪得很乾淨，混在一起播放還是可能很怪，原因是：

- **角度一直換**：這張是側面、下一張是從上往下拍，看起來像換了一隻貓。
- **走路不是真的在走**：如果沒有「側面走路」的影片，程式只能拿站著的圖在螢幕上滑動，腳沒在動。
- **姿勢之間接不起來**：從站著直接跳到趴著，中間沒有「趴下」的過程，只能用淡入淡出帶過。

**結論：自然感來自「一致」，不是來自「素材多」。**
同一個角度、同一個大小、同一種光線，加上真的在走路的影片，比一堆不同角度的照片有用得多。

---

## 五、別人是怎麼做到「像真的」？

我們查了市面上的桌寵、動畫素材和相關研究。做得自然的作品都有一個共通點：
**整套動作是同一隻、同一個視角、同一個大小，而且背景本來就乾淨**，不是拿生活照硬剪出來的。

| 做法 | 怎麼做 | 優點 | 缺點 |
|---|---|---|---|
| **AI 生成整套動作** | 用一張清楚的側面照當起點，讓影片 AI 在單色背景上生成走路、坐下、趴下、睡覺 | 最像真的，而且是你家那隻的長相 | 要付費或排隊；AI 偶爾會畫錯（多一條腿、臉變了） |
| **3D 模型＋骨架動畫**（遊戲的做法） | 照片轉成 3D 模型，綁上骨架，套用現成的走路、待機動作 | 動作最順、最穩定 | 看起來比較像遊戲畫面；坐、趴的現成動作比較少 |
| **在固定條件下拍攝** | 固定鏡頭、側面、單色背景拍真實影片 | 完全真實 | 寵物不一定配合 |
| 從影片重建可動的 3D 動物（研究中） | 從多段手機影片學出一隻能動的 3D 動物 | 理論上最好 | 需要很強的顯示卡，還不是一般人能用的工具 |

本專案接下來的方向是第一種：用一張最清楚的側面照，讓 AI 在**純色背景**上生成整套動作。
純色背景可以用「去綠幕」的方式剪出來（跟拍電影一樣），不需要 AI 去猜哪裡是寵物，就不會有破圖。

參考資料：
[Cat Gatekeeper](https://www.fastcompany.com/91532809/chrome-browser-extension-cat-gatekeeping)、
[AI 生成的透明背景貓咪循環素材](https://booth.pm/en/items/8012391)、
[Tripo 四腳動物自動綁骨架](https://www.tripo3d.ai/blog/ai-character-animations-animals)、
[Meshy Animate](https://docs.meshy.ai/en/webapp/guides/animate)、
[Animal Avatars（ECCV 2024）](https://arxiv.org/abs/2403.17103)、
[BANMo](https://arxiv.org/abs/2112.12761)、
[Motion Graphs](https://hhoppe.com/motiongraph.pdf)

---

## 六、用 AI 補沒拍到的動作（選用）

在訓練器下方按 **「🪄 AI 補齊動作…」**，程式會列出還缺哪些動作，用你的照片當「開始」和「結束」畫面，讓影片 AI 生成中間的動作。

| AI 來源 | 費用 | 說明 |
|---|---|---|
| Hugging Face | 免費 | 額度很少，實測每天大約只能做 1 段，整套要好幾天 |
| fal.ai | 付費 | 一次做完；每段約 US$0.1～0.5，依選用的模型而定 |

- Hugging Face：到 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 免費註冊，建立一個 **Read** 權限的 token 貼進程式
- fal.ai：到 [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) 建立金鑰（要先儲值）
- 金鑰只存在你自己電腦的 `settings.json`
- 選了線上 AI 時，用來當起點的照片會上傳到該服務
- AI 偶爾會畫錯，生成完會出現在訓練器裡讓你檢查，不好的直接刪掉
- 起點照片越好（側面、整隻、背景單純），生成結果越好

---

## 七、操作方式

| 動作 | 效果 |
|---|---|
| 點牠 | 冒愛心、說話；睡著時會被叫醒 |
| 拖曳 | 拎起來，放開會照你甩的方向飛出去 |
| 滑鼠在牠身上來回移動 | 摸摸，會呼嚕 |
| 雙擊 | 跳一下 |
| 右鍵（或右下角系統匣圖示） | 叫牠過來、讓牠睡、再召喚一隻、調大小、安靜模式、開機自動啟動、結束 |

想改牠會說的話：編輯 `pets/<名字>/pet.json` 裡的 `"phrases"`。

---

## 八、分享給朋友

訓練器裡按 **「匯出這隻…」**，會存成一個 `.petpack` 檔。
朋友在他的訓練器按 **「匯入寵物包…」**（或直接把檔案拖進去），就能直接放到桌面上，不用重新處理。

---

## 九、常見問題

**第一次處理很慢？**
第一次要下載 AI 模型（共約 330MB），之後就不用了。

**牠倒退著走？**
訓練器裡對那張圖按右鍵 →「左右翻轉面向」。

**不想讓牠爬到視窗上？**
在牠身上按右鍵 → 取消勾選「可以跳到視窗上」。

**兔子、倉鼠、鳥可以嗎？**
可以試，但程式最熟悉的是貓和狗。認不出來時會標 ⚠，請自己檢查剪得好不好。

**電腦記憶體不夠？**
輪廓精修每次約需 3.3GB 記憶體。可用記憶體不夠時，程式會自動改用比較簡單的去背，不會當掉，只是品質差一點。

---

## 十、給開發者

### 專案結構

```
app/trainer.py    訓練器（視窗介面 + 指令列）
app/cutout.py     找動物、去背、品管、判斷姿勢、影片片段挑選
app/realclips.py  用 AI 生成整套連續動作
app/aifill.py     AI 影片服務（Hugging Face / fal.ai）
app/pet.py        桌面寵物本體（物理、行為、互動、系統匣）
app/winenv.py     Windows 視窗位置與閒置時間偵測
app/fastdl.py     多連線下載、續傳、校驗
pets/             訓練好的寵物（不會上傳到 Git）
```

### 指令列

```
.venv\Scripts\python app\trainer.py --cli 名字 照片或影片資料夾 [--fast]
```

### 處理流程（技術版）

YOLOX-S 偵測 → IS-Net 去背 → SAM ViT-B（以偵測框和 IS-Net 結果當提示）輪廓精修，並與 IS-Net 毛邊合成 → 品管（身體被畫面切到、特寫、品質分數、重複）→ CLIP 零樣本姿勢判斷 → 影片：整段掃描、找姿勢穩定的片段、逐格去背、丟掉與前後格形狀不一致的格、找最佳循環點。

macOS / Linux 也能跑（`pip install -r requirements.txt` 後執行 `python app/trainer.py`），但跳上視窗與閒置偵測只支援 Windows。

---

## English

Feed photos and videos of your pet and it comes to life on your Windows desktop: it walks along the taskbar, sits, naps, jumps onto other windows, and can be picked up and thrown. Everything runs locally unless you opt in to an online AI video service.

**Honest status:** the pipeline is fully automatic, but realism depends almost entirely on the input. Casual photos (top-down shots, pet half out of frame, pet buried in blankets) and handheld videos produce inconsistent, sometimes broken sprites. What makes desktop pets look real is consistency: one viewpoint, one scale, one lighting, a clean background, and real walk cycles. The planned direction is generating a full, consistent set of motions from one clean side-view photo on a solid-color background, then chroma-keying it (no segmentation guesswork).

**Pipeline:** YOLOX detection → IS-Net matting → SAM (ViT-B, box + IS-Net mask prompt) silhouette refinement fused with IS-Net edges → quality gate (frame cut, close-up, score, dedupe) → CLIP zero-shot pose → videos: scan, mine stable-pose segments, per-frame matting, temporal consistency filter, best loop point.

## 授權 / Credits

本專案採 MIT License。使用的開源元件：
[rembg](https://github.com/danielgatis/rembg)（MIT）、
[IS-Net / DIS](https://github.com/xuebinqin/DIS)（Apache-2.0）、
[Segment Anything](https://github.com/facebookresearch/segment-anything)（Apache-2.0）、
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0）、
[CLIP](https://github.com/openai/CLIP)（MIT）、
[PySide6 / Qt](https://www.qt.io/qt-for-python)（LGPL-3.0）。
