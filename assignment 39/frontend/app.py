"""Streamlit UI cho TechViet Support. Chỉ gọi FastAPI qua HTTP (không import code agent), nên UI và API
deploy tách được (xem docker-compose.yml).

    python -m api.main                      # terminal 1
    streamlit run frontend/app.py           # terminal 2, API_URL mặc định http://localhost:8000
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT_S", "300"))

st.set_page_config(page_title="TechViet Support", page_icon="🛒", layout="wide")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kw)
    except requests.RequestException as e:
        return None, f"Không kết nối được API ({API_URL}): {e}"
    if r.status_code >= 400:
        try:
            body = r.json()
            msg = body.get("detail") or body.get("error") or r.text
        except ValueError:
            msg = r.text
        return None, f"HTTP {r.status_code}: {msg}"
    return r.json(), None


def init_state():
    ss = st.session_state
    ss.setdefault("session_id", None)
    ss.setdefault("messages", [])          # {"role", "content", "meta"}
    ss.setdefault("total_cost", 0.0)
    ss.setdefault("total_turns", 0)


def new_conversation():
    if st.session_state.session_id:
        api("DELETE", f"/sessions/{st.session_state.session_id}")
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.total_cost = 0.0
    st.session_state.total_turns = 0


init_state()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛒 TechViet Support")
    health, err = api("GET", "/health")
    if err:
        st.error(err)
    else:
        st.success("API đang chạy")
        st.caption(f"Router: `{health['router']}`  \nAgents: `{health['agent_model']}`  \n"
                   f"Tools: `{health['tools_backend']}` · Rewrite: `{health['query_rewrite']}`  \n"
                   f"RAG: `{health['rag_mode'] or 'chưa tải'}` · Sessions: {health['sessions']}")

    st.divider()
    st.subheader("Hội thoại")
    st.caption(f"session_id: `{st.session_state.session_id or '(mới)'}`")
    c1, c2 = st.columns(2)
    c1.metric("Lượt", st.session_state.total_turns)
    c2.metric("Chi phí", f"${st.session_state.total_cost:.4f}")
    if st.button("➕ Hội thoại mới", use_container_width=True):
        new_conversation()
        st.rerun()
    if st.session_state.session_id and st.button("🧹 Xoá state (giữ session)", use_container_width=True):
        _, err = api("POST", f"/sessions/{st.session_state.session_id}/reset")
        if err:
            st.error(err)
        else:
            st.session_state.messages = []
            st.rerun()
    if st.session_state.session_id:
        with st.expander("ConversationState"):
            info, err = api("GET", f"/sessions/{st.session_state.session_id}")
            st.json(info["state"] if info else {"error": err})

    with st.expander("A2A Agent cards"):
        cards, err = api("GET", "/agents")
        st.json(cards if cards else {"error": err})

    with st.expander("Admin"):
        token = st.text_input("ADMIN_TOKEN", type="password")
        if st.button("Khôi phục dữ liệu đơn hàng") and token:
            _, err = api("POST", "/admin/reset-data", headers={"X-Admin-Token": token})
            if err:
                st.error(err)
            else:
                st.success("Đã khôi phục dữ liệu")

    st.divider()
    st.caption("Thử: *Màn hình mình mới mua bị điểm chết, muốn đổi* → rồi gõ *DH1009*")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_chat, tab_rag = st.tabs(["💬 Chat", "🔎 RAG search"])


def render_meta(meta: dict):
    m = meta["metrics"]
    cols = st.columns(5)
    cols[0].caption(f"intent **{meta['intent']}**")
    cols[1].caption(f"agents **{', '.join(meta['agents']) or '—'}**")
    cols[2].caption(f"⏱ {m['latency_ms'] / 1000:.1f}s")
    cols[3].caption(f"🔁 {m['llm_calls']} LLM · in {m['input_tokens']} / out {m['output_tokens']}"
                    f" / cache {m['cache_read_tokens']}")
    cols[4].caption(f"💲 {m['cost_usd']:.4f}")
    with st.expander("Chi tiết lượt (plan, tool calls, trace)"):
        if meta["missing_info"]:
            st.warning(f"Thiếu thông tin: {meta['missing_info']} → hỏi lại khách, không gọi agent")
        if meta["plan"]:
            st.markdown("**Plan**")
            st.table([{"agent": p["agent"], "task": p["task"]} for p in meta["plan"]])
        if meta["tool_calls"]:
            st.markdown("**Tool calls**")
            for c in meta["tool_calls"]:
                st.code(f"{c['name']}({c.get('args', {})})", language="python")
        if meta["agent_results"]:
            st.markdown("**Kết quả từng agent (A2A task)**")
            st.json(meta["agent_results"], expanded=False)
        st.caption(f"trace_id `{meta['trace_id']}` — xem trong results/traces/api-*.jsonl hoặc dashboard")


with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                render_meta(msg["meta"])

    if prompt := st.chat_input("Nhập câu hỏi cho TechViet…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý…"):
                data, err = api("POST", "/chat", json={"message": prompt, "session_id": st.session_state.session_id})
            if err:
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err}"})
            else:
                st.session_state.session_id = data["session_id"]
                st.session_state.total_cost += data["metrics"]["cost_usd"]
                st.session_state.total_turns += 1
                st.markdown(data["reply"])
                render_meta(data)
                st.session_state.messages.append({"role": "assistant", "content": data["reply"], "meta": data})
        st.rerun()


with tab_rag:
    st.markdown("Debug RAG: xem câu hỏi được **viết lại** thế nào và chunk nào lọt top (không gọi agent).")
    q = st.text_input("Câu hỏi", value="tai nghe tws bóc seal r đổi đc k? hoàn tiền mất bn ngày")
    c1, c2 = st.columns([1, 1])
    mode = c1.radio("Query rewrite", ["off", "rules", "llm"], index=1, horizontal=True)
    top_k = c2.slider("top_k", 1, 10, 4)
    if st.button("Tìm", type="primary") and q.strip():
        with st.spinner("Đang tìm… (lần đầu phải tải index RAG)"):
            data, err = api("POST", "/rag/search", json={"query": q, "top_k": top_k, "rewrite": mode})
        if err:
            st.error(err)
        else:
            rw = data["rewrite"]
            st.caption(f"RAG mode `{data['rag_mode']}` · {data['latency_ms']:.0f} ms · rewrite source `{rw['source']}`")
            st.markdown("**Truy vấn đã dùng**")
            for i, qq in enumerate(rw["queries"]):
                st.markdown(f"{i + 1}. `{qq}`")
            if rw["keywords"]:
                st.markdown("**Từ khoá BM25:** " + ", ".join(f"`{k}`" for k in rw["keywords"]))
            st.markdown("**Kết quả**")
            for h in data["hits"]:
                score = f"rerank {h['rerank_score']}" if h["rerank_score"] is not None else f"rrf {h['rrf']}"
                with st.container(border=True):
                    st.markdown(f"**{h['doc']} › {h['section']}**  ·  `{h['chunk_id']}`  ·  {score}  ·  "
                                f"bm25 #{h['bm25_rank'] or '—'} · dense #{h['dense_rank'] or '—'}")
                    st.text(h["text"])
