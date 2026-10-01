"""安裝前測一下哪個 Python 套件來源最快（只用標準函式庫）。

印出最快來源的 index URL，給安裝程式的 pip -i 使用。
官方 PyPI 在某些網路環境（公司網路、部分 ISP）會慢到只剩幾十 KB/s，
這時改用鏡像站通常能快幾十倍。
"""
import time
import urllib.request

INDEXES = [
    "https://pypi.org/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://repo.huaweicloud.com/repository/pypi/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
]
PROBE = "/numpy/"  # 頁面約數百 KB，夠用來估速度


def speed(index: str) -> float:
    t0 = time.time()
    got = 0
    try:
        req = urllib.request.Request(index + PROBE, headers={"User-Agent": "pip/24"})
        with urllib.request.urlopen(req, timeout=5) as r:
            while time.time() - t0 < 4:
                chunk = r.read(65536)
                if not chunk:
                    break
                got += len(chunk)
    except Exception:
        return 0.0
    return got / max(time.time() - t0, 0.05) if got > 20_000 else 0.0


if __name__ == "__main__":
    import sys

    print("測試各套件來源速度（約 20 秒）…", file=sys.stderr)
    results = {i: speed(i) for i in INDEXES}  # 依序測，避免互搶頻寬

    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {v / 1024:8.0f} KB/s  {k}", file=sys.stderr)
    best = max(results, key=results.get)
    # 官方來源只要不是慢很多就優先用官方
    if results["https://pypi.org/simple"] >= results[best] * 0.5:
        best = "https://pypi.org/simple"
    print(best)
