# -*- coding: utf-8 -*-
"""对话历史存储：Redis（极简 RESP 客户端，纯标准库 socket 实现，无需 redis 包）。

历史以 JSON 字符串列表存储：
    key = rag:history:{session_id}
    value = ["{\"role\":\"user\",\"content\":\"...\"}", "{\"role\":\"assistant\",...}", ...]

Redis 不可用/未配置密码时，读写都会静默降级（返回空历史、跳过写入），
保证聊天主流程不受影响。
"""
import hashlib
import json
import socket

import config


class RedisError(Exception):
    pass


class _Resp:
    """极简 Redis RESP 客户端。每个命令独立建连，适合低频读写。"""

    def __init__(self, host, port, password, db=0, timeout=8):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.timeout = timeout

    def _connect(self):
        try:
            return socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as e:
            raise RedisError(f"无法连接 Redis {self.host}:{self.port}: {e}") from e

    @staticmethod
    def _encode(*args):
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            b = a if isinstance(a, bytes) else str(a).encode("utf-8")
            parts.append(f"${len(b)}\r\n".encode())
            parts.append(b)
            parts.append(b"\r\n")
        return b"".join(parts)

    def _read_line(self, f):
        line = f.readline()
        if not line:
            raise RedisError("Redis 连接被关闭")
        return line

    def _read_reply(self, f):
        line = self._read_line(f)
        p = line[:1]
        if p == b"+":
            return line[1:-2].decode("utf-8", "replace")
        if p == b"-":
            raise RedisError(line[1:-2].decode("utf-8", "replace"))
        if p == b":":
            return int(line[1:-2])
        if p == b"$":
            n = int(line[1:-2])
            if n == -1:
                return None
            data = f.read(n)
            f.read(2)  # 末尾 CRLF
            return data
        if p == b"*":
            n = int(line[1:-2])
            if n == -1:
                return None
            return [self._read_reply(f) for _ in range(n)]
        raise RedisError(f"未知 RESP 前缀: {p!r}")

    def command(self, *args):
        s = self._connect()
        try:
            f = s.makefile("rb")
            if self.password:
                s.sendall(self._encode("AUTH", self.password))
                self._read_reply(f)
            if self.db:
                s.sendall(self._encode("SELECT", self.db))
                self._read_reply(f)
            s.sendall(self._encode(*args))
            return self._read_reply(f)
        finally:
            s.close()


def _client():
    return _Resp(config.REDIS_HOST, config.REDIS_PORT,
                 config.REDIS_PASSWORD, config.REDIS_DB)


def is_available():
    try:
        return _client().command("PING") == "PONG"
    except Exception:
        return False


def history_key(session_id):
    return f"rag:history:{session_id}"


def get_history(session_id, max_messages=None):
    """返回 [{role, content}, ...]（按时间先后）。失败返回 []。"""
    try:
        raw = _client().command("LRANGE", history_key(session_id), 0, -1) or []
        items = []
        for x in raw:
            try:
                items.append(json.loads(x.decode("utf-8")))
            except Exception:
                continue
        if max_messages:
            items = items[-max_messages:]
        return items
    except Exception:
        return []


def append_history(session_id, role, content):
    """追加一条消息，并按最大长度裁剪 + 设置过期时间。失败静默忽略。"""
    try:
        c = _client()
        key = history_key(session_id)
        c.command("RPUSH", key, json.dumps({"role": role, "content": content},
                                          ensure_ascii=False))
        c.command("LTRIM", key, -config.REDIS_HISTORY_MAX, -1)
        c.command("EXPIRE", key, config.REDIS_HISTORY_TTL)
        return True
    except Exception:
        return False


def clear_history(session_id):
    try:
        _client().command("DEL", history_key(session_id))
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ 答案缓存
def get_cache_version():
    try:
        v = _client().command("GET", "rag:cache:version")
        return int(v) if v else 0
    except Exception:
        return 0


def bump_cache_version():
    """入库后调用，使旧答案缓存失效。"""
    try:
        return int(_client().command("INCR", "rag:cache:version"))
    except Exception:
        return get_cache_version()


def get_cached_answer(query: str, ctx: str = ""):
    """按 (query + 历史摘要) 读答案缓存；命中返回结果 dict，否则 None。"""
    try:
        h = hashlib.md5((query + "||" + (ctx or "")).encode("utf-8")).hexdigest()[:16]
        raw = _client().command("GET", f"rag:answer:{get_cache_version()}:{h}")
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        return None


def set_cached_answer(query: str, result, ctx: str = "", ttl=3600):
    try:
        h = hashlib.md5((query + "||" + (ctx or "")).encode("utf-8")).hexdigest()[:16]
        _client().command("SET", f"rag:answer:{get_cache_version()}:{h}",
                          json.dumps(result, ensure_ascii=False), "EX", ttl)
        return True
    except Exception:
        return False
