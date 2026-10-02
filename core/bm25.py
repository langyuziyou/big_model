# -*- coding: utf-8 -*-
"""BM25 精确检索（字符 n-gram）+ 常见意图同义词扩展。

对“短查询 vs 长文档”的区分度远好于哈希后的余弦相似度，
用作“该问题是否能在简历中找到答案”的可靠判据。
"""
import re
import math
from collections import Counter

# 常见简历问答意图 -> 同义/相关词扩展（仅当查询含“键”时才扩展）
QUERY_EXPAND = {
    "电话": ["电话", "手机", "手机号", "联系方式", "联系电话", "号码"],
    "手机": ["电话", "手机", "手机号", "联系方式", "联系电话", "号码"],
    "邮箱": ["邮箱", "邮件", "email", "mail"],
    "学历": ["学历", "本科", "学校", "大学", "学院", "毕业", "教育背景"],
    "学校": ["学校", "大学", "学院", "本科", "学历", "毕业", "教育背景"],
    "教育": ["教育", "学历", "学校", "大学", "学院", "本科", "毕业"],
    "技能": ["技能", "技术", "技术栈", "精通", "擅长", "熟练", "掌握"],
    "技术": ["技术", "技术栈", "技能", "精通", "擅长", "熟练"],
    "项目": ["项目", "项目经历", "参与", "负责", "主导"],
    "工作": ["工作", "工作经历", "任职", "公司", "岗位", "职位", "经历"],
    "公司": ["公司", "任职", "工作经历", "单位", "企业"],
    "经历": ["经历", "工作经历", "项目经历", "任职", "公司"],
    "年龄": ["年龄", "岁数", "几岁", "出生"],
    "多大": ["年龄", "岁数", "几岁"],
    "几岁": ["年龄", "岁数", "几岁"],
    "姓名": ["姓名", "名字", "叫什么", "是谁"],
    "名字": ["名字", "姓名", "叫什么", "是谁"],
    "性别": ["性别", "男", "女"],
}


def _normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r'\s+', '', text)


def expand_query(query: str) -> str:
    """命中意图词时，把同义/相关词追加进查询（用于提高召回）。"""
    q = query or ""
    extra = []
    for key, words in QUERY_EXPAND.items():
        if key in q:
            for w in words:
                if w not in q:
                    extra.append(w)
    if extra:
        return q + " " + " ".join(extra)
    return q


def _tokens(text: str):
    """字符 2/3-gram 词条（去掉单字符，降低常见字噪声）。"""
    t = _normalize(text)
    toks = []
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            toks.append(t[i:i + n])
    return toks


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.docs = [_tokens(c) for c in self.chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        self.N = len(self.docs)
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str):
        qt = _tokens(expand_query(query))
        if not qt:
            return []
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for t in qt:
                if t not in self.df:
                    continue
                tfd = tf.get(t, 0)
                if tfd <= 0:
                    continue
                denom = tfd + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += self._idf(t) * (tfd * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores

    def top(self, query: str, k=5):
        res = self.score(query)[:k]
        return [{"idx": i, "score": round(s, 4),
                 "text": self.chunks[i]} for i, s in res]
