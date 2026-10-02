# -*- coding: utf-8 -*-
"""向量化：智谱 embedding API（主）+ 本地字符哈希（降级兜底）。

- 主：调用智谱 open.bigmodel.cn 的 /embeddings 接口（默认 embedding-2，1024 维）。
- 兜底：当 API 未配置/不可用时，退回本地字符 n-gram 哈希向量（保证离线仍可跑）。

入库用 embed_many() 批量调用；查询用 embed() 单条调用（带内存缓存）。
"""
import hashlib
import re

import requests

import config

_cache = {}


def _normalize(text: str) -> str:
    return re.sub(r'\s+', '', (text or '').lower())


def _hash_sign(token, seed, dim):
    h = hashlib.md5(f"{seed}:{token}".encode('utf-8')).digest()
    idx = int.from_bytes(h[:4], 'big') % dim
    sign = 1.0 if (h[4] & 1) else -1.0
    return idx, sign


def _hash_embed(text: str, dim: int = 1024) -> list:
    """本地哈希降级向量（无模型、离线可用）。"""
    t = _normalize(text)
    vec = [0.0] * dim
    if t:
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                idx, sign = _hash_sign(t[i:i + n], n, dim)
                vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _zhipu_embed_batch(texts):
    """调用智谱 embedding 接口，返回 list[list[float]]；失败抛异常。"""
    if not config.ZHIPU_API_KEY:
        raise RuntimeError("未配置 ZHIPU_API_KEY")
    headers = {
        "Authorization": f"Bearer {config.ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": config.ZHIPU_EMBED_MODEL, "input": texts}
    r = requests.post(config.ZHIPU_EMBED_BASE_URL, json=payload,
                      headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    vecs = [it["embedding"] for it in items]
    if len(vecs) != len(texts):
        raise RuntimeError(f"智谱返回向量数不符: {len(vecs)} != {len(texts)}")
    return vecs


def embed(text: str) -> list:
    """单条向量化（查询用），带内存缓存。"""
    if text in _cache:
        return _cache[text]
    try:
        vecs = _zhipu_embed_batch([text])
        vec = vecs[0]
    except Exception:
        vec = _hash_embed(text, config.EMBED_DIM)
    _cache[text] = vec
    return vec


def embed_many(texts, batch_size=8):
    """批量向量化（入库用）。逐批调用智谱，失败的批次降级为哈希。"""
    out = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i:i + batch_size]
        try:
            out.extend(_zhipu_embed_batch(sub))
        except Exception:
            out.extend(_hash_embed(t, config.EMBED_DIM) for t in sub)
    return out


def check():
    """检测智谱 embedding 是否可用，返回 (ok, dim, message)。"""
    try:
        vecs = _zhipu_embed_batch(["测试"])
        dim = len(vecs[0])
        return True, dim, f"智谱 {config.ZHIPU_EMBED_MODEL} 可用，向量维度 {dim}"
    except Exception as e:
        return False, config.EMBED_DIM, f"智谱不可用({type(e).__name__})，将使用本地哈希降级"
