# -*- coding: utf-8 -*-
"""RAG 核心：简历分块入库、向量检索（智谱）、大模型回答（DeepSeek）。

检索采用“向量检索 + BM25”混合，判据为二者取并集：
    - 智谱向量 + Milvus 余弦相似度 >= SCORE_THRESHOLD，或
    - BM25 精确匹配 >= BM25_THRESHOLD
命中后把相关简历片段作为上下文，交给 DeepSeek 生成自然语言答案；
DeepSeek 不可用时回退为直接返回命中的简历原文（仍然“如实”）。
"""
import os
import re

import requests

import config
from core.pdf_extract import extract_text
from core import embedding
from core.milvus_client import MilvusClient
from core.bm25 import BM25Index
from core import redis_store

# 内存缓存：从 Milvus 载入的分块 + BM25 索引（查询时惰性构建）
_chunks_cache = None
_bm25_cache = None


def _extract_timeline_entries(text: str):
    """抽取所有“名称 角色 起止时间”条目，返回 [(name, role, start, end)]，按开始时间升序。"""
    pat = re.compile(
        r'(.{2,40}?)\s+(架构师|项目经理|维护工程师|java开发|java|运维)'
        r'\s*(\d{4}\.\d{1,2})\s*[-—~至到]\s*(\d{4}\.\d{1,2}|至今|现在)',
        re.IGNORECASE)
    entries, seen = [], set()
    for m in pat.finditer(text):
        name = m.group(1).strip()
        role, start, end = m.group(2), m.group(3), m.group(4)
        key = name + start
        if key in seen:
            continue
        seen.add(key)
        entries.append((name, role, start, end))
    entries.sort(key=lambda p: "9999.99" if p[2] in ("至今", "现在") else p[2])
    return entries


def _extract_project_timeline(text: str):
    """项目时间线：排除公司/教育。"""
    exclude = ("公司", "大学", "学院", "学校", "动力")
    return [(n, r, s, e) for n, r, s, e in _extract_timeline_entries(text)
            if not any(k in n for k in exclude)]


def _extract_work_timeline(text: str):
    """工作经历时间线：公司条目（含“公司/动力”），排除教育。"""
    include = ("公司", "动力")
    exclude = ("大学", "学院", "学校")
    return [(n, r, s, e) for n, r, s, e in _extract_timeline_entries(text)
            if any(k in n for k in include) and not any(k in n for k in exclude)]


def _annotate(text: str):
    """返回 (标注后文本, 时间线独立分块列表)。

    时间线作为独立分块（不并入正文），避免被正文分块拆散，
    保证“第N份工作/第N个项目”能检索到完整的时间线。
    """
    extra = []
    phones = sorted(set(re.findall(r'1[3-9]\d{9}', text)))
    if phones:
        extra.append("联系电话 手机号 联系方式：" + "、".join(phones))
    emails = sorted(set(re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)))
    if emails:
        extra.append("邮箱 电子邮箱 email：" + "、".join(emails))
    ages = sorted(set(re.findall(r'(\d+)\s*岁', text)))
    if ages:
        extra.append("年龄 岁数 几岁：" + "、".join(a + "岁" for a in ages))
    gm = re.search(r'([男女])\s*[|｜]', text[:500])
    if gm:
        extra.append("性别：" + gm.group(1))
    if extra:
        text = text + "\n" + "\n".join(extra)

    timeline_chunks = []
    project_tl = _extract_project_timeline(text)
    if project_tl:
        first = project_tl[0]
        lines = [
            "项目经历时间线（按开始时间从早到晚排序）：",
            f"项目总数：{len(project_tl)}个",
            f"第一个项目/最早的项目是：{first[0]}（{first[2]}-{first[3]}）",
        ]
        for i, (name, role, start, end) in enumerate(project_tl, 1):
            lines.append(f"{i}. {name}（{start}-{end}）")
        timeline_chunks.append("\n".join(lines))
    work_tl = _extract_work_timeline(text)
    if work_tl:
        lines = [
            "工作经历时间线（按开始时间从早到晚排序）：",
            f"工作总数：{len(work_tl)}份",
        ]
        for i, (name, role, start, end) in enumerate(work_tl, 1):
            lines.append(f"第{i}份工作：{name}（{role}，{start}-{end}）")
        timeline_chunks.append("\n".join(lines))
    # 在线人数/活跃用户/流量指标标注（把“日活/月活/QPS”等与“在线人数”概念打通）
    metric_lines = [l.strip() for l in text.splitlines()
                    if re.search(r'(日活|月活|DAU|MAU|QPS|在线人数|活跃用户|用户量)', l, re.IGNORECASE)]
    if metric_lines:
        timeline_chunks.append("项目在线人数/活跃用户/流量数据：\n" + "\n".join(metric_lines))
    return text, timeline_chunks


def chunk_text(text: str, size: int = None, overlap: int = None) -> list:
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    chunks = []
    cur = ""
    for line in lines:
        if cur and len(cur) + 1 + len(line) > size:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap > 0 else ""
        cur = (cur + " " + line).strip() if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def _client() -> MilvusClient:
    return MilvusClient(config.MILVUS_URI)


def _read_resume_text(path: str) -> str:
    """读取简历文本：PDF 走解析器；txt/md 直接读（自动尝试多种编码）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue
        return ""
    return extract_text(path)


def invalidate_cache():
    global _chunks_cache, _bm25_cache
    _chunks_cache = None
    _bm25_cache = None


def ensure_collection(force: bool = False, dim: int = None):
    c = _client()
    dim = dim or config.EMBED_DIM
    if force and c.has_collection(config.COLLECTION_NAME):
        c.drop_collection(config.COLLECTION_NAME)
    c.create_collection(config.COLLECTION_NAME, dim, config.METRIC_TYPE)
    return c


def ingest(files=None, force=True):
    """解析简历 -> 分块 -> 智谱向量化 -> 写入 Milvus。返回统计信息。"""
    files = files or config.collect_resume_files()
    records = []  # (source, chunk_idx, text)
    detail = []
    for f in files:
        if not os.path.isfile(f):
            continue
        text = _read_resume_text(f)
        if not text.strip():
            detail.append({"file": os.path.basename(f), "chunks": 0, "warning": "未提取到文本"})
            continue
        text, timeline_chunks = _annotate(text)
        chunks = chunk_text(text)
        chunks.extend(timeline_chunks)
        for i, ch in enumerate(chunks):
            records.append((os.path.basename(f), i, ch))
        detail.append({"file": os.path.basename(f),
                       "chunks": sum(1 for s, _, _ in records if s == os.path.basename(f))})

    if not records:
        return {"chunks": 0, "detail": detail, "collection": config.COLLECTION_NAME}

    # 向量化（智谱，失败自动降级哈希）
    vecs = embedding.embed_many([r[2] for r in records])
    dim = len(vecs[0]) if vecs else config.EMBED_DIM

    c = ensure_collection(force=force, dim=dim)
    rows = [{"vector": vecs[i], "text": records[i][2],
             "source": records[i][0], "chunk_idx": records[i][1]}
            for i in range(len(records))]
    total = 0
    for i in range(0, len(rows), 100):
        batch = rows[i:i + 100]
        c.insert(config.COLLECTION_NAME, batch)
        total += len(batch)
    invalidate_cache()
    redis_store.bump_cache_version()  # 使旧答案缓存失效
    return {"chunks": total, "detail": detail, "collection": config.COLLECTION_NAME,
            "embed_dim": dim}


def load_chunks():
    """从 Milvus 载入全部分块，并构建 BM25 索引（惰性 + 缓存）。"""
    global _chunks_cache, _bm25_cache
    if _chunks_cache is not None:
        return _chunks_cache, _bm25_cache
    c = _client()
    rows = []
    try:
        rows = c.query(config.COLLECTION_NAME, expr="id >= 0",
                       output_fields=["text", "source"], limit=16384) or []
    except Exception:
        rows = []
    _chunks_cache = [{"text": (r.get("text") or "").strip(),
                      "source": r.get("source") or ""} for r in rows]
    texts = [r["text"] for r in _chunks_cache]
    _bm25_cache = BM25Index(texts) if texts else None
    return _chunks_cache, _bm25_cache


def search(query: str, top_k: int = None):
    """智谱向量 + Milvus 余弦检索，返回命中列表（含 distance/text/source/id）。"""
    top_k = top_k or config.TOP_K
    c = _client()
    vec = embedding.embed(query)
    try:
        return c.search(config.COLLECTION_NAME, vec, top_k=top_k,
                        output_fields=["id", "text", "source"]) or []
    except Exception:
        return []


def _retrieve(query: str, bm25):
    """混合检索，返回 (vec_hits, bm25_hits, cosine_top, bm25_top)。"""
    vec_hits = search(query, top_k=config.TOP_K)
    cosine_top = float(vec_hits[0].get("distance", 0) or 0) if vec_hits else 0.0
    bm25_hits = bm25.top(query, k=config.TOP_K)
    bm25_top = bm25_hits[0]["score"] if bm25_hits else 0.0
    return vec_hits, bm25_hits, cosine_top, bm25_top


def _is_answerable(cosine_top: float, bm25_top: float) -> bool:
    strong = (cosine_top >= config.SCORE_THRESHOLD) or (bm25_top >= config.BM25_THRESHOLD)
    weak = (cosine_top >= config.WEAK_SCORE_THRESHOLD) and (bm25_top >= config.WEAK_BM25_THRESHOLD)
    return strong or weak


def _clearly_unrelated(cosine_top: float, bm25_top: float) -> bool:
    """两个信号都几乎为零 => 明显无关，直接拒绝（不浪费大模型调用）。"""
    return cosine_top < 0.20 and bm25_top < 1.0


def _looks_declined(text: str) -> bool:
    """大模型是否表示“无法回答”。"""
    t = (text or "").strip()
    return (not t) or ("无法回答" in t) or ("没有提供" in t and "无法" in t)


def _merge_context(vec_hits, bm25_hits, top_n=4):
    """合并向量与 BM25 命中，按综合相关度排序去重，返回前 top_n 条文本。"""
    max_bm25 = max((h["score"] for h in bm25_hits), default=0) or 1.0
    merged = {}
    for h in vec_hits:
        t = (h.get("text") or "").strip()
        if t:
            merged[t] = max(merged.get(t, 0.0), float(h.get("distance", 0) or 0))
    for h in bm25_hits:
        t = h["text"].strip()
        if t:
            merged[t] = max(merged.get(t, 0.0), h["score"] / max_bm25)
    return [t for t, _ in sorted(merged.items(), key=lambda kv: -kv[1])][:top_n]


def _rewrite_query(query: str, history=None) -> str:
    """用大模型理解用户真实意图，把问题改写为自包含、适合检索的查询。

    用于直接检索失败时（或存在历史需解析指代时）重试一次。
    失败返回原查询。
    """
    if not config.DEEPSEEK_API_KEY:
        return query
    try:
        url = config.DEEPSEEK_BASE_URL.rstrip('/') + "/chat/completions"
        headers = {"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                   "Content-Type": "application/json"}
        messages = [
            {"role": "system",
             "content": "你是查询改写助手。根据历史对话，把用户问题改写成一句完整、明确、适合检索的查询。"
                        "解析代词指代和隐含意图，例如“第二个工作”改写成“工作经历时间线中按开始时间从早到晚排序的第2份工作”，"
                        "“那他呢”改写成包含具体对象的问题。只输出改写后的查询，不要任何解释或标点包裹。"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"用户问题：{query}"})
        payload = {"model": config.DEEPSEEK_MODEL, "messages": messages,
                   "temperature": 0, "stream": False}
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        rewritten = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        return rewritten or query
    except Exception:
        return query


def _llm_answer(query: str, context: str, history=None) -> str:
    """DeepSeek 依据简历上下文 + 历史对话生成答案；失败返回空串（触发回退）。"""
    if not config.DEEPSEEK_API_KEY:
        return ""
    try:
        url = config.DEEPSEEK_BASE_URL.rstrip('/') + "/chat/completions"
        headers = {"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                   "Content-Type": "application/json"}
        messages = [
            {"role": "system",
             "content": "你是简历智能客服助手。只依据提供的简历内容如实回答，不得编造；"
                        "如果简历中没有相关信息，请直接回答“无法回答”。"
                        "请结合历史对话理解代词指代（如“他”“那”），保持回答连贯。"
                        "关于时间线：简历末尾的“项目经历时间线”和“工作经历时间线”都按开始时间从早到晚排序；"
                        "回答“第一个/最早的项目”以项目时间线第1项为准；"
                        "回答“第二个工作/第N份工作/第几份工作”时，以工作经历时间线第N项为准（从最早开始数）；"
                        "“最后一个工作/最近的工作”指工作经历时间线的最后一项；"
                        "教育经历的时间（如2003.09-2008.07）既不是项目也不是工作，切勿混用。"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user",
                         "content": f"简历内容：\n{context}\n\n问题：{query}\n请简洁、准确地回答。"})
        payload = {
            "model": config.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def _not_answerable(cos: float, bm: float):
    return {"answerable": False, "answer": "", "sources": [],
            "score": round(cos, 4), "bm25_score": round(bm, 4), "hits": []}


def answer(query: str, session_id: str = None):
    """核心问答。

    流程：直接检索 -> 改写重试 -> 大模型兜底（窄上下文答不出则放大到全量简历），
    并带 Redis 答案缓存（键含“查询 + 历史摘要”，入库时自动失效）。
    """
    history = redis_store.get_history(session_id, max_messages=10) if session_id else []
    ctx_digest = "|".join(f"{h['role']}:{h['content']}" for h in history)

    # 命中缓存直接返回
    cached = redis_store.get_cached_answer(query, ctx_digest)
    if cached:
        return cached

    chunks, bm25 = load_chunks()
    if not chunks or bm25 is None:
        return _not_answerable(0.0, 0.0)

    # 1) 直接检索
    vec_hits, bm25_hits, cosine_top, bm25_top = _retrieve(query, bm25)

    # 2) 改写重试（明显无关则跳过）
    if not _is_answerable(cosine_top, bm25_top) and not _clearly_unrelated(cosine_top, bm25_top):
        rewritten = _rewrite_query(query, history)
        if rewritten and rewritten != query:
            vec_hits, bm25_hits, cosine_top, bm25_top = _retrieve(rewritten, bm25)

    # 明显无关 -> 直接拒绝（不浪费大模型调用）
    if _clearly_unrelated(cosine_top, bm25_top):
        return _not_answerable(cosine_top, bm25_top)

    source_of = {c["text"]: c["source"] for c in chunks}
    context_narrow = _merge_context(vec_hits, bm25_hits, top_n=4)

    # 3) 先窄上下文；答不出再放大到全量简历兜底
    answer_text = _llm_answer(query, "\n\n".join(context_narrow), history)
    if _looks_declined(answer_text):
        full = "\n\n".join(c["text"] for c in chunks)
        answer_text = _llm_answer(query, full, history)

    if _looks_declined(answer_text):
        return _not_answerable(cosine_top, bm25_top)

    result = {
        "answerable": True,
        "answer": answer_text,
        "sources": list(dict.fromkeys(source_of.get(t, "") for t in context_narrow if source_of.get(t))),
        "score": round(cosine_top, 4),
        "bm25_score": round(bm25_top, 4),
        "hits": [{"text": t, "source": source_of.get(t, "")} for t in context_narrow[:3]],
    }

    # 4) 记录历史 + 写入缓存
    if session_id:
        redis_store.append_history(session_id, "user", query)
        redis_store.append_history(session_id, "assistant", answer_text)
    redis_store.set_cached_answer(query, result, ctx_digest)
    return result


def stats():
    c = _client()
    return {
        "collection": config.COLLECTION_NAME,
        "count": c.count(config.COLLECTION_NAME),
        "has_collection": c.has_collection(config.COLLECTION_NAME),
    }
