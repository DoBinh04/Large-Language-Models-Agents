"""Quản lý hội thoại cho API: mỗi session_id = một ConversationState riêng.

Tất cả session dùng chung router, backend tool, 3 agent, composer và tracer (các thành phần này không giữ
state hội thoại), chỉ khác `ConversationState`. Nhờ vậy 500 session không tạo 2000 Anthropic client.

- Mỗi session có lock riêng: 2 request cùng session chạy tuần tự (state không bị ghi đè), khác session chạy song song.
- Hết hạn sau SESSION_TTL_S không dùng; vượt MAX_SESSIONS thì bỏ session dùng lâu nhất (LRU).
- Lưu trong RAM: restart container là mất hội thoại. Muốn bền thì serialize `state.to_dict()` ra Redis / DB.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from support.logging_setup import get_logger
from support.orchestrator import Orchestrator
from support.state import ConversationState

log = get_logger(__name__)


@dataclass
class Session:
    id: str
    orch: Orchestrator
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, base: Orchestrator, ttl_s: int, max_sessions: int):
        self.base = base
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _new_orch(self) -> Orchestrator:
        orch = copy.copy(self.base)              # chia sẻ router / agents / backend / tracer
        orch.state = ConversationState()
        return orch

    def _evict(self, now: float) -> None:
        expired = [k for k, s in self._items.items() if now - s.last_used > self.ttl_s]
        for k in expired:
            del self._items[k]
        while len(self._items) > self.max_sessions:
            k, _ = self._items.popitem(last=False)
            expired.append(k)
        if expired:
            log.info("sessions evicted", extra={"count": len(expired), "remaining": len(self._items)})

    def get_or_create(self, session_id: str | None) -> Session:
        now = time.time()
        with self._lock:
            if session_id and session_id in self._items:
                s = self._items[session_id]
                self._items.move_to_end(session_id)
            else:
                sid = session_id or uuid.uuid4().hex[:16]
                s = Session(id=sid, orch=self._new_orch())
                self._items[sid] = s
                log.info("session created", extra={"session_id": sid})
            s.last_used = now
            self._evict(now)          # sau khi thêm: giữ đúng <= max_sessions (session mới nằm cuối, không bị bỏ)
            return s

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._items.get(session_id)

    def reset(self, session_id: str) -> bool:
        s = self.get(session_id)
        if not s:
            return False
        with s.lock:
            # không gọi orch.reset(): hàm đó reset dữ liệu đơn hàng dùng chung cho mọi session
            s.orch.state = ConversationState()
            s.last_used = time.time()
        return True

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
