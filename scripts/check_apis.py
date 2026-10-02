# -*- coding: utf-8 -*-
"""API 连通性自检：DeepSeek（回答） + 智谱（向量化）。

运行：python scripts/check_apis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
import config  # noqa: E402
from core import embedding  # noqa: E402


def check_deepseek():
    print("== DeepSeek 回答模型 ==")
    try:
        url = config.DEEPSEEK_BASE_URL.rstrip('/') + "/chat/completions"
        r = requests.post(url, headers={
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": config.DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": "请回复两个字：正常"}],
            "max_tokens": 10,
            "stream": False,
        }, timeout=60)
        r.raise_for_status()
        content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"  ✅ 可用，模型={config.DEEPSEEK_MODEL}，返回：{content}")
    except Exception as e:
        print(f"  ❌ 失败：{e}")


def check_zhipu():
    print("== 智谱 向量化 ==")
    ok, dim, msg = embedding.check()
    mark = "✅" if ok else "⚠️"
    print(f"  {mark} {msg}")
    if ok:
        v = embedding.embed("测试向量")
        print(f"  向量维度={dim}，示例前5维={[round(x, 4) for x in v[:5]]}")


if __name__ == "__main__":
    check_deepseek()
    print()
    check_zhipu()
