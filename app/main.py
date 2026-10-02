# -*- coding: utf-8 -*-
"""RAG 智能客服后端：REST + SSE（Server-Sent Events），支持多客服工作台按负载路由。

实时通道用 SSE 实现（普通 HTTP 流式响应，浏览器原生 EventSource，无需 websockets 库）。

路由策略：
    - 客户转人工时，分配给“当前名下客户最少”的在线客服（最少负载）。
    - 每个客服只收到自己名下客户的事件，互不可见。
    - 客服掉线时，其名下客户自动转接给其它在线客服；无人在线则客户排队等待。
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import core.rag as rag
from core import embedding

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# ------------------------------------------------------------------ 会话管理
class CustomerSession:
    def __init__(self, sid: str):
        self.sid = sid
        self.name = f"客户-{sid[:4]}"
        self.manual = False
        self.connected = False
        self.history = []               # [{from, text}]
        self.last_question = ""
        self.agent_id = None            # 分配到的客服 id
        self.queue = asyncio.Queue()    # 发给该客户的事件


class Agent:
    def __init__(self, agent_id: str, name: str = ""):
        self.agent_id = agent_id
        self.name = name or f"客服-{agent_id[:4]}"
        self.queue = asyncio.Queue()    # 发给该客服的事件
        self.customers = set()          # 名下客户 sid 集合


class Hub:
    def __init__(self):
        self.customers = {}             # sid -> CustomerSession
        self.agents = {}                # agent_id -> Agent

    # ---------------- 客服 ----------------
    def pick_agent(self):
        """最少负载路由：选择当前名下客户最少的在线客服。"""
        if not self.agents:
            return None
        return min(self.agents.values(), key=lambda a: len(a.customers))

    async def send_to_agent(self, agent_id, evt):
        a = self.agents.get(agent_id)
        if not a:
            return False
        try:
            await a.queue.put(evt)
            return True
        except Exception:
            return False

    async def push_agent_snapshot(self, agent_id):
        """把“该客服名下客户列表”推给该客服。"""
        a = self.agents.get(agent_id)
        if not a:
            return
        sess = []
        for sid in list(a.customers):
            c = self.customers.get(sid)
            if c and c.manual:
                sess.append({"session_id": sid, "name": c.name,
                             "manual": c.manual, "history": c.history})
        await a.queue.put({"type": "customer_list", "sessions": sess})

    # ---------------- 分配 ----------------
    async def assign_agent(self, sid):
        """把客户 sid 分配给最少负载的客服；无人在线返回 None。"""
        c = self.customers.get(sid)
        if not c:
            return None
        agent = self.pick_agent()
        if not agent:
            c.agent_id = None
            return None
        c.agent_id = agent.agent_id
        agent.customers.add(sid)
        return agent

    async def assign_waiting(self):
        """把排队中的客户（manual 且未分配）分配给在线客服。"""
        waiting = [c for c in self.customers.values()
                   if c.manual and c.agent_id is None]
        for c in waiting:
            agent = await self.assign_agent(c.sid)
            if not agent:
                break
            await c.queue.put({"type": "manual_connected", "agent": agent.name})
            await self.send_to_agent(agent.agent_id, {
                "type": "customer_connected", "session_id": c.sid,
                "name": c.name, "history": c.history})
            await self.push_agent_snapshot(agent.agent_id)

    async def deliver_customer_message(self, c, text):
        """把客户消息投递给其所属客服；客服掉线/未分配则自动重路由。"""
        if c.agent_id:
            ok = await self.send_to_agent(c.agent_id, {
                "type": "message", "session_id": c.sid,
                "from": "customer", "text": text})
            if ok:
                return
            a = self.agents.get(c.agent_id)
            if a:
                a.customers.discard(c.sid)
            c.agent_id = None
        agent = await self.assign_agent(c.sid)
        if agent:
            await c.queue.put({"type": "manual_connected", "agent": agent.name})
            await self.send_to_agent(agent.agent_id, {
                "type": "customer_connected", "session_id": c.sid,
                "name": c.name, "history": c.history})
            await self.send_to_agent(agent.agent_id, {
                "type": "message", "session_id": c.sid,
                "from": "customer", "text": text})
        else:
            await c.queue.put({"type": "manual_waiting"})

    async def remove_agent(self, agent_id):
        """客服掉线：其名下客户自动转接给其它客服，或进入排队。"""
        a = self.agents.pop(agent_id, None)
        if not a:
            return
        sids = list(a.customers)
        for sid in sids:
            c = self.customers.get(sid)
            if c:
                c.agent_id = None
        for sid in sids:
            c = self.customers.get(sid)
            if not c or not c.manual:
                continue
            agent = await self.assign_agent(sid)
            if agent:
                await c.queue.put({"type": "agent_reassigned", "agent": agent.name})
                await self.send_to_agent(agent.agent_id, {
                    "type": "customer_connected", "session_id": sid,
                    "name": c.name, "history": c.history})
                await self.push_agent_snapshot(agent.agent_id)
            else:
                await c.queue.put({"type": "manual_waiting"})

    async def notify_offline(self, sid):
        """客户下线：通知其所属客服，并清理分配关系。"""
        c = self.customers.get(sid)
        if not c or not c.manual:
            return
        if c.agent_id:
            await self.send_to_agent(c.agent_id, {
                "type": "customer_offline", "session_id": sid, "name": c.name})
            await self.push_agent_snapshot(c.agent_id)
        a = self.agents.get(c.agent_id)
        if a:
            a.customers.discard(sid)
        c.agent_id = None

    async def switch_to_manual(self, sid, trigger_text=None):
        """把客户切到人工并路由给客服（自然语言“转人工”与弹框“是”共用）。"""
        c = self.customers.get(sid)
        if not c:
            return False
        c.manual = True
        c.history.append({"from": "system", "text": "已切换到人工服务"})
        if trigger_text:
            c.last_question = trigger_text
            if not any(h.get("from") == "customer" and h.get("text") == trigger_text
                       for h in c.history):
                c.history.append({"from": "customer", "text": trigger_text})
        agent = await self.assign_agent(sid)
        if agent:
            await c.queue.put({"type": "manual_connected", "agent": agent.name})
            await self.send_to_agent(agent.agent_id, {
                "type": "customer_connected", "session_id": sid,
                "name": c.name, "history": c.history})
            await self.push_agent_snapshot(agent.agent_id)
            return True
        await c.queue.put({"type": "manual_waiting"})
        return False


def _is_manual_request(text: str) -> bool:
    """判断用户是否在要求转人工（自然语言）。"""
    t = (text or "").replace(" ", "")
    if "人工智能" in t:
        return False
    return ("人工" in t) or ("真人" in t)


hub = Hub()


# ------------------------------------------------------------------ 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    ok, dim, msg = await asyncio.to_thread(embedding.check)
    print(f"[startup] 向量化: {msg}")
    print(f"[startup] 回答模型: DeepSeek {config.DEEPSEEK_MODEL} "
          f"({'已配置' if config.DEEPSEEK_API_KEY else '未配置'})")
    try:
        st = rag.stats()
        if not st["has_collection"] or st["count"] == 0:
            print("[startup] 集合为空，开始自动入库…")
            result = await asyncio.to_thread(rag.ingest, None, True)
            print(f"[startup] 入库完成: {result['chunks']} 块, 维度 {result.get('embed_dim')}")
    except Exception as e:
        print(f"[startup] 自动入库失败: {e}")
    yield


app = FastAPI(title="RAG 智能客服问答系统", lifespan=lifespan)


# ------------------------------------------------------------------ 基础 REST
@app.get("/api/stats")
async def api_stats():
    try:
        return {"ok": True, **rag.stats()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/resumes")
async def api_resumes():
    return {"ok": True, "files": [str(f) for f in config.collect_resume_files()]}


@app.post("/api/ingest")
async def api_ingest():
    try:
        return {"ok": True, **rag.ingest(force=True)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/chat")
async def api_chat(payload: dict):
    payload = payload or {}
    text = payload.get("message", "").strip()
    if not text:
        return {"ok": False, "error": "消息为空"}
    session_id = payload.get("session_id") or None
    try:
        result = await asyncio.to_thread(rag.answer, text, session_id)
        result["ok"] = True
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------ 页面
@app.get("/")
async def page_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent")
async def page_agent():
    return FileResponse(STATIC_DIR / "agent.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------ 客户侧
@app.post("/api/customer/join")
async def customer_join():
    sid = uuid.uuid4().hex[:8]
    hub.customers[sid] = CustomerSession(sid)
    return {"ok": True, "sid": sid, "name": hub.customers[sid].name}


@app.get("/api/customer/stream")
async def customer_stream(sid: str):
    c = hub.customers.get(sid)
    if not c:
        raise HTTPException(404, "会话不存在，请刷新页面")

    async def gen():
        c.connected = True
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    evt = await asyncio.wait_for(c.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            c.connected = False
            try:
                asyncio.create_task(hub.notify_offline(sid))
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/customer/chat")
async def customer_chat(payload: dict):
    sid = payload.get("sid")
    text = (payload.get("text") or "").strip()
    c = hub.customers.get(sid)
    if not c or not text:
        return {"ok": False, "error": "会话无效"}
    if c.manual:
        c.history.append({"from": "customer", "text": text})
        await hub.deliver_customer_message(c, text)
    elif _is_manual_request(text):
        # 用户直接要求“转人工/人工客服/真人” -> 直接切人工
        await hub.switch_to_manual(sid, trigger_text=text)
    else:
        c.last_question = text
        c.history.append({"from": "customer", "text": text})
        try:
            result = await asyncio.to_thread(rag.answer, text, sid)
        except Exception:
            result = {"answerable": False, "answer": "", "sources": [],
                      "score": 0, "bm25_score": 0}
        if result.get("answerable"):
            c.history.append({"from": "bot", "text": result["answer"]})
            await c.queue.put({
                "type": "answer", "answer": result["answer"],
                "sources": result.get("sources", []),
                "bm25_score": result.get("bm25_score", 0),
            })
        else:
            await c.queue.put({"type": "need_manual", "text": text})
    return {"ok": True}


@app.post("/api/customer/switch_manual")
async def customer_switch_manual(payload: dict):
    sid = payload.get("sid")
    c = hub.customers.get(sid)
    if not c:
        return {"ok": False, "error": "会话无效"}
    await hub.switch_to_manual(sid, trigger_text=c.last_question)
    return {"ok": True}


@app.post("/api/customer/end_manual")
async def customer_end_manual(payload: dict):
    sid = payload.get("sid")
    c = hub.customers.get(sid)
    if not c:
        return {"ok": False, "error": "会话无效"}
    aid = c.agent_id
    c.manual = False
    c.agent_id = None
    if aid:
        a = hub.agents.get(aid)
        if a:
            a.customers.discard(sid)
    await c.queue.put({"type": "manual_ended"})
    if aid:
        await hub.send_to_agent(aid, {
            "type": "customer_offline", "session_id": sid, "name": c.name})
        await hub.push_agent_snapshot(aid)
    return {"ok": True}


# ------------------------------------------------------------------ 客服侧
@app.get("/api/agent/stream")
async def agent_stream(agent_id: str = "", name: str = ""):
    agent_id = agent_id or uuid.uuid4().hex[:8]
    agent = Agent(agent_id, name)
    hub.agents[agent_id] = agent

    async def gen():
        try:
            await agent.queue.put({"type": "hello", "role": "agent",
                                   "agent_id": agent_id, "name": agent.name})
            await hub.push_agent_snapshot(agent_id)
            asyncio.create_task(hub.assign_waiting())
            while True:
                try:
                    evt = await asyncio.wait_for(agent.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            try:
                asyncio.create_task(hub.remove_agent(agent_id))
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/agent/reply")
async def agent_reply(payload: dict):
    agent_id = payload.get("agent_id")
    sid = payload.get("sid")
    text = (payload.get("text") or "").strip()
    c = hub.customers.get(sid)
    if not c:
        return {"ok": False, "error": "该客户已下线"}
    if not text:
        return {"ok": False, "error": "回复为空"}
    if c.agent_id != agent_id:
        # 客户处于人工模式但无人接单（如原客服掉线）时，转接给当前回复的客服
        if c.manual and c.agent_id is None and agent_id in hub.agents:
            a = hub.agents[agent_id]
            c.agent_id = agent_id
            a.customers.add(sid)
            await c.queue.put({"type": "manual_connected", "agent": a.name})
        else:
            return {"ok": False, "error": "该客户不属于你的工作台"}
    c.history.append({"from": "agent", "text": text})
    await c.queue.put({"type": "message", "from": "agent", "text": text})
    return {"ok": True}
