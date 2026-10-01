"""素材規格：需要哪些照片／影片、目前湊齊多少、能做到哪個等級。

訓練器上方的檢查表就是用這裡的定義產生的。
"""
from __future__ import annotations

# (代碼, 名稱, 需要的素材, 說明, 等級, AI 能不能補)
ITEMS = [
    ("photo:walk", "站姿照片", "照片", "側面站著、四隻腳都拍到", 1, False),
    ("photo:sit", "坐姿照片", "照片", "坐著，全身入鏡", 1, False),
    ("photo:lie", "趴／睡照片", "照片", "趴著或蜷著睡覺", 1, False),
    ("loop:walk", "走路動畫", "影片", "側面走過畫面 3～5 秒", 2, True),
    ("trans:walk>sit", "坐下／站起", "影片", "從站著到坐下 3～5 秒", 3, True),
    ("trans:sit>lie", "趴下／起身", "影片", "從坐著到趴下 3～5 秒", 3, True),
    ("loop:sit", "坐著待機", "影片", "坐著不動：轉頭、搖尾巴 3～5 秒", 3, True),
    ("loop:lie", "睡覺呼吸", "影片", "趴著或睡覺：呼吸起伏 3～5 秒", 3, True),
]

LEVELS = {
    0: ("還不能用", "至少要有一張照片"),
    1: ("① 基本", "靜止圖＋晃動，換姿勢淡入淡出"),
    2: ("② 會走路", "走路時腳真的會動"),
    3: ("③ 像活著", "走、坐、趴、睡全部是連續的真實動作"),
}

SPEC_TEXT = (
    "<b>照片</b>：全身入鏡、寵物佔畫面 1/3 以上、清楚不晃、背景越單純越好；JPG／PNG／HEIC 都可以。"
    "<br><b>影片</b>：每段 3～5 秒（只取前 6 秒）、手機橫拿、<u>鏡頭固定不動</u>、全身一直在畫面裡、只拍一隻；"
    "MP4／MOV 都可以，手機原檔即可。"
)


def coverage(entries) -> dict[str, int]:
    """entries：每個素材一個 dict（pose, kind, to, animated）。回傳每個檢查項目湊到幾份。"""
    have = {code: 0 for code, *_ in ITEMS}
    for e in entries:
        pose, kind, to, anim = e.get("pose"), e.get("kind"), e.get("to"), e.get("animated")
        if not anim:
            if f"photo:{pose}" in have:
                have[f"photo:{pose}"] += 1
            continue
        if kind == "trans" and to:
            for a, b in (("walk", "sit"), ("sit", "lie")):
                if {pose, to} == {a, b}:
                    have[f"trans:{a}>{b}"] += 1
            # 轉場的頭尾也算是那個姿勢的畫面
            for p in (pose, to):
                if f"photo:{p}" in have:
                    have[f"photo:{p}"] += 1
        else:
            if f"loop:{pose}" in have:
                have[f"loop:{pose}"] += 1
            if f"photo:{pose}" in have:
                have[f"photo:{pose}"] += 1
    return have


def level(have: dict[str, int]) -> int:
    if not any(have.values()):
        return 0
    lv = 1
    if have["loop:walk"]:
        lv = 2
        if all(have[c] for c, *_ in ITEMS if c.startswith(("trans:", "loop:"))):
            lv = 3
    return lv


def ai_fillable(have: dict[str, int]) -> list[str]:
    """缺的、而且 AI 有材料可以補的項目（需要開始／結束姿勢的照片）。"""
    out = []
    for code, _, _, _, _, can_ai in ITEMS:
        if have[code] or not can_ai:
            continue
        kind, rest = code.split(":")
        poses = rest.split(">")
        if all(have.get(f"photo:{p}") for p in poses):
            out.append(code)
    return out


def html_table(have: dict[str, int]) -> str:
    lv = level(have)
    fill = set(ai_fillable(have))
    rows = []
    for code, name, kind, how, lvl, _ in ITEMS:
        n = have[code]
        mark = "✅" if n else ("🪄" if code in fill else "⬜")
        state = f"{n} 份" if n else ("缺（可 AI 補）" if code in fill else "缺")
        rows.append(f"<tr><td>{mark}</td><td><b>{name}</b></td><td>{kind}</td><td>{how}</td>"
                    f"<td align=center>{'①②③'[lvl - 1]}</td><td>{state}</td></tr>")
    nxt = ""
    if lv < 3:
        need = [name for code, name, _, _, l, _ in ITEMS if not have[code] and l == lv + 1] or \
               [name for code, name, _, _, l, _ in ITEMS if not have[code]]
        nxt = f"　→ 再補：{'、'.join(need)} 就能升到 {LEVELS[lv + 1][0]}"
    if fill and lv < 3:
        nxt += "（或勾選「缺的動作用 AI 補」）"
    return (f"<b>目前等級：{LEVELS[lv][0]}</b>　{LEVELS[lv][1]}{nxt}"
            "<table cellspacing=0 cellpadding=3 style='margin-top:4px'>"
            "<tr><th></th><th align=left>項目</th><th align=left>素材</th><th align=left>怎麼拍</th><th>等級</th><th align=left>狀態</th></tr>"
            + "".join(rows) + "</table>")
