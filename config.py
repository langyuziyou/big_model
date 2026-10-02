# -*- coding: utf-8 -*-
"""全局配置。

敏感信息（API Key / Redis 密码）一律通过环境变量或项目根目录的 .env 文件注入，
不硬编码在代码里。.env 已加入 .gitignore，不会提交到版本库；请参考 .env.example。

优先级：系统环境变量 > .env 文件 > 代码内默认值（仅限非敏感项）。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """极简 .env 加载器：读取 KEY=VALUE 行；已存在的环境变量优先（不覆盖）。"""
    path = path or os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


# ---------- DeepSeek：回答大模型（敏感） ----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- 智谱(Zhipu)：向量化（敏感） ----------
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_EMBED_MODEL = os.environ.get("ZHIPU_EMBED_MODEL", "embedding-2")     # embedding-2=1024 维 / embedding-3=2048 维
ZHIPU_EMBED_BASE_URL = os.environ.get("ZHIPU_EMBED_BASE_URL",
                                      "https://open.bigmodel.cn/api/paas/v4/embeddings")

# ---------- Milvus 向量库 ----------
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "resume_kb")
METRIC_TYPE = os.environ.get("METRIC_TYPE", "COSINE")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# ---------- 检索参数 ----------
TOP_K = int(os.environ.get("TOP_K", "5"))
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.40"))
BM25_THRESHOLD = float(os.environ.get("BM25_THRESHOLD", "5.0"))
WEAK_SCORE_THRESHOLD = float(os.environ.get("WEAK_SCORE_THRESHOLD", "0.30"))
WEAK_BM25_THRESHOLD = float(os.environ.get("WEAK_BM25_THRESHOLD", "1.0"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "240"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "40"))

# ---------- 简历来源 ----------
DEFAULT_RESUME_FILE = os.environ.get("DEFAULT_RESUME_FILE",
                                     r"C:\Users\admin\Desktop\袁志杰-Java time.pdf")
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
# 可通过环境变量 RESUME_PATHS 追加文件或目录，用分号分隔
EXTRA_RESUMES = [p for p in os.environ.get("RESUME_PATHS", "").split(";") if p.strip()]


def collect_resume_files():
    """汇总所有简历文件（默认文件 + resumes 目录 + 环境变量追加），支持 pdf/txt/md。"""
    exts = (".pdf", ".txt", ".md")

    def _is_resume(fname: str) -> bool:
        n = fname.lower()
        return n.endswith(exts) and not n.startswith("readme") and not n.startswith("说明")

    files = []
    if os.path.isfile(DEFAULT_RESUME_FILE):
        files.append(DEFAULT_RESUME_FILE)
    if os.path.isdir(RESUME_DIR):
        for f in sorted(os.listdir(RESUME_DIR)):
            if _is_resume(f):
                files.append(os.path.join(RESUME_DIR, f))
    for p in EXTRA_RESUMES:
        p = p.strip()
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if _is_resume(f)]
        elif os.path.isfile(p) and p.lower().endswith(exts):
            files.append(p)
    seen, uniq = set(), []
    for f in files:
        ap = os.path.abspath(f)
        if ap not in seen:
            seen.add(ap)
            uniq.append(f)
    return uniq


# ---------- Redis（对话历史存储，敏感） ----------
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_HISTORY_TTL = int(os.environ.get("REDIS_HISTORY_TTL", str(24 * 3600)))  # 历史保留 24 小时
REDIS_HISTORY_MAX = int(os.environ.get("REDIS_HISTORY_MAX", "20"))            # 每个会话最多保留条数

# ---------- Web 服务 ----------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
