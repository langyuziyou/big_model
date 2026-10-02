# -*- coding: utf-8 -*-
"""Milvus 客户端：通过 Milvus RESTful API（HTTP）交互，无需安装 pymilvus。

仅依赖 requests。覆盖：列出/创建/删除集合、插入、向量检索、条件查询、统计。
"""
import requests


class MilvusError(Exception):
    pass


class MilvusClient:
    def __init__(self, uri: str = "http://localhost:19530", timeout: int = 15):
        self.base = uri.rstrip('/')
        self.timeout = timeout

    # ---------------- 底层 ----------------
    def _post(self, path, payload):
        try:
            r = requests.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise MilvusError(f"连接 Milvus 失败({self.base}): {e}") from e
        try:
            data = r.json()
        except ValueError:
            raise MilvusError(f"Milvus 返回非 JSON (HTTP {r.status_code}): {r.text[:300]}")
        if data.get("code") not in (0, 200):
            raise MilvusError(f"Milvus 错误: {data}")
        return data.get("data", data)

    def _get(self, path, params=None):
        try:
            r = requests.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise MilvusError(f"连接 Milvus 失败({self.base}): {e}") from e
        try:
            data = r.json()
        except ValueError:
            raise MilvusError(f"Milvus 返回非 JSON (HTTP {r.status_code}): {r.text[:300]}")
        if data.get("code") not in (0, 200):
            raise MilvusError(f"Milvus 错误: {data}")
        return data.get("data", data)

    # ---------------- 集合 ----------------
    def list_collections(self):
        return self._get("/v1/vector/collections") or []

    def has_collection(self, name):
        return name in self.list_collections()

    def describe(self, name):
        return self._get("/v1/vector/collections/describe",
                         params={"collectionName": name})

    def create_collection(self, name, dim, metric="COSINE"):
        """创建集合；已存在则直接返回 False。"""
        if self.has_collection(name):
            return False
        payload = {
            "collectionName": name,
            "dimension": dim,
            "metricType": metric,
            "primaryField": "id",
            "vectorField": "vector",
            "autoId": False,
        }
        self._post("/v1/vector/collections/create", payload)
        return True

    def drop_collection(self, name):
        self._post("/v1/vector/collections/drop", {"collectionName": name})
        return True

    # ---------------- 数据 ----------------
    def insert(self, name, rows):
        """rows: list[dict]，每行必须含 id(int) 和 vector(list[float])，
        其余字段（text/source/...）作为动态字段写入。"""
        payload = {"collectionName": name, "data": rows}
        return self._post("/v1/vector/insert", payload)

    def search(self, name, vector, top_k=5, output_fields=None, expr=""):
        payload = {
            "collectionName": name,
            "vector": vector,   # 单个查询向量（扁平 list[float]）
            "annsField": "vector",
            "limit": top_k,
            "outputFields": output_fields or ["*"],
            "filter": expr,
        }
        return self._post("/v1/vector/search", payload)

    def query(self, name, expr="", output_fields=None, limit=100, offset=0):
        payload = {
            "collectionName": name,
            "filter": expr or "id >= 0",
            "outputFields": output_fields or ["*"],
            "limit": limit,
            "offset": offset,
        }
        return self._post("/v1/vector/query", payload)

    def count(self, name):
        """统计集合内实体数量。"""
        try:
            data = self.query(name, expr="id >= 0", output_fields=["id"], limit=16384)
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0
