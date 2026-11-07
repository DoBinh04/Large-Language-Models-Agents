"""Query rewriting cho RAG: biến câu hỏi tự nhiên của khách thành truy vấn hợp với tài liệu chính sách.

Vì sao cần: khách gõ "tai tws bóc seal r đổi đc k", "bh mh bao lâu", "đơn 5tr trả góp dc ko".
BM25 không khớp "bh" với "bảo hành", "tws" với "true wireless"; dense model nhỏ cũng yếu với teencode.
Câu nhiều ý ("đổi ý được mấy ngày? hoàn tiền mất bao lâu?") nên tách thành nhiều truy vấn để mỗi ý
có chunk riêng lọt top, thay vì một chunk "trung bình" cho cả hai.

Ba chế độ (env QUERY_REWRITE, mặc định "rules"):
    off   : giữ nguyên câu hỏi.
    rules : chuẩn hoá viết tắt / teencode / tiền tệ + tách câu nhiều ý. Không tốn API, deterministic,
            nên eval tái lập được. Đủ cho phần lớn câu hỏi vì agent Claude đã tự viết query khi gọi tool.
    llm   : rules trước, rồi Claude (mặc định Haiku 5.5 @ low, structured output) viết lại thành 1–3 truy vấn
            dùng đúng thuật ngữ trong tài liệu (danh sách heading được đưa vào prompt) + từ khoá cho BM25.
            Lỗi API / từ chối -> tự lùi về kết quả rules. Kết quả được cache theo câu đã chuẩn hoá.

Output `RewrittenQuery`:
    queries  : các truy vấn đem đi search (dense + BM25), truy vấn đầu là truy vấn chính.
    keywords : từ khoá chỉ cộng thêm vào nhánh BM25.
    rerank_query : câu dùng cho cross-encoder (câu đã chuẩn hoá đầy đủ, không phải từng mảnh).
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import config
from support.logging_setup import get_logger
from support.tracing import NULL_TRACER, Tracer, current_tracer

log = get_logger(__name__)

MODES = ("off", "rules", "llm")

# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
# Viết tắt / teencode -> thuật ngữ dùng trong data/docs. Khoá viết thường; khớp nguyên từ.
ABBREVIATIONS: dict[str, str] = {
    # chính sách
    "bh": "bảo hành", "bảo hàng": "bảo hành", "bao hanh": "bảo hành",
    "đt": "điện thoại", "dt": "điện thoại", "dthoai": "điện thoại", "đthoại": "điện thoại",
    "sp": "sản phẩm", "nsx": "nhà sản xuất", "rma": "đổi trả",
    "refund": "hoàn tiền", "hoàn lại tiền": "hoàn tiền", "trả lại tiền": "hoàn tiền",
    "đổi trả hàng": "đổi trả", "trả hàng": "đổi trả",
    "ship": "vận chuyển giao hàng", "shipper": "giao hàng", "vc": "vận chuyển",
    "freeship": "miễn phí vận chuyển", "free ship": "miễn phí vận chuyển",
    "cod": "thanh toán khi nhận hàng COD", "ck": "chuyển khoản", "tg": "trả góp",
    "fullbox": "nguyên hộp", "full box": "nguyên hộp",
    "nguyên seal": "nguyên seal", "còn seal": "nguyên seal", "seal": "nguyên seal",
    "bóc seal": "đã mở hộp", "khui hộp": "đã mở hộp", "đập hộp": "đã mở hộp",
    "active": "kích hoạt", "actived": "đã kích hoạt",
    "die": "hỏng", "rớt": "rơi", "vô nước": "vào nước",
    "điểm die": "điểm chết", "dead pixel": "điểm chết",
    "đk": "điều kiện", "dk": "điều kiện",
    # sản phẩm (không nở "tab" / "4K" vì trùng tên sản phẩm "Nova Tab S", "Nova View 27 4K")
    "tws": "true wireless", "tai phone": "tai nghe", "headphone": "tai nghe", "earbuds": "tai nghe true wireless",
    "mh": "màn hình", "lap": "laptop", "lappy": "laptop", "ipad": "máy tính bảng",
    "smartwatch": "đồng hồ",
    # teencode
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hong": "không", "khong": "không",
    "dc": "được", "đc": "được",
    "r": "rồi", "rùi": "rồi", "j": "gì", "gi": "gì", "bn": "bao nhiêu", "bnhiu": "bao nhiêu",
    "ntn": "như thế nào", "sao v": "sao vậy",
    "trc": "trước", "ms": "mới", "mk": "mình",
}


def _build_abbrev_re(table: dict[str, str]) -> re.Pattern:
    keys = sorted(table, key=len, reverse=True)                     # khớp cụm dài trước ("bóc seal" trước "seal")
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?<![\w])({alt})(?![\w])", re.IGNORECASE)


_ABBREV_RE = _build_abbrev_re(ABBREVIATIONS)
# "dh 1005" là mã đơn (support/state.ORDER_RE) -> chỉ nở "đh"/"dh" thành "đơn hàng" khi không có số theo sau.
_ORDER_WORD_RE = re.compile(r"(?<![\w])(đh|dh)(?![\w])(?!\s?-?\d)", re.IGNORECASE)
_MONEY_RE = [
    (re.compile(r"(\d+(?:[.,]\d+)?)\s?(?:tr|triệu|củ)(?![\w])", re.I), r"\1 triệu"),
    (re.compile(r"(\d{2,})\s?(?:k|nghìn|ngàn)(?![\w])", re.I), r"\1 nghìn"),   # >= 2 chữ số để "4K" vẫn là độ phân giải
]
_SPACE_RE = re.compile(r"\s+")
_SPLIT_RE = re.compile(r"(?<=[?!;\n])\s*|\.\s+(?=[A-ZÀ-Ỹa-zà-ỹ])")
_CONTINUATION_RE = re.compile(r"^(với lại|ngoài ra|và)\s+", re.I)


def normalize(text: str) -> str:
    """Chuẩn hoá khoảng trắng, tiền tệ, viết tắt / teencode. Giữ nguyên mã đơn, mã sản phẩm, dấu câu."""
    s = _SPACE_RE.sub(" ", text).strip()
    for pat, rep in _MONEY_RE:
        s = pat.sub(rep, s)
    s = _ORDER_WORD_RE.sub("đơn hàng", s)
    s = _ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], s)
    # "không" lặp do "k ko" -> gộp; dọn khoảng trắng một lần nữa
    s = re.sub(r"\b(không)(\s+\1\b)+", r"\1", s, flags=re.I)
    return _SPACE_RE.sub(" ", s).strip()


def split_subqueries(text: str, min_words: int = 3, max_parts: int = 3) -> list[str]:
    """Tách câu nhiều ý theo ? ! ; xuống dòng và dấu chấm giữa câu. Mảnh quá ngắn gộp vào mảnh trước."""
    parts: list[str] = []
    for raw in _SPLIT_RE.split(text):
        p = raw.strip(" ?!;.,")
        if not p:
            continue
        if parts and len(p.split()) < min_words:
            parts[-1] = f"{parts[-1]} {p}"
            continue
        parts.append(_CONTINUATION_RE.sub("", p))
    return parts[:max_parts]


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = x.strip()
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


@dataclass
class RewrittenQuery:
    original: str
    normalized: str
    queries: list[str]
    keywords: list[str] = field(default_factory=list)
    mode: str = "rules"
    source: str = "rules"            # rules | llm | cache | fallback

    @property
    def rerank_query(self) -> str:
        return self.normalized or self.original

    @property
    def changed(self) -> bool:
        return self.queries != [self.original] or bool(self.keywords)

    def to_dict(self) -> dict:
        return asdict(self) | {"rerank_query": self.rerank_query}


def rewrite_rules(query: str) -> RewrittenQuery:
    norm = normalize(query)
    subs = split_subqueries(norm)
    # truy vấn chính = cả câu đã chuẩn hoá; các mảnh chỉ thêm vào khi câu thật sự có nhiều ý
    queries = _dedupe([norm] + (subs if len(subs) > 1 else []))
    return RewrittenQuery(original=query, normalized=norm, queries=queries or [query], mode="rules")


# --------------------------------------------------------------------------- #
# LLM rewrite
# --------------------------------------------------------------------------- #
REWRITE_SYSTEM = """Bạn viết lại câu hỏi của khách hàng cửa hàng điện tử TechViet thành truy vấn tìm kiếm trên kho tài liệu nội bộ.

Kho tài liệu gồm các mục sau (tên tài liệu > mục):
{headings}

Yêu cầu:
- queries: 1 đến 3 truy vấn tiếng Việt có dấu, mỗi truy vấn MỘT ý. Câu hỏi một ý thì chỉ 1 truy vấn.
  Dùng đúng thuật ngữ của tài liệu (vd. "đổi ý", "lỗi nhà sản xuất", "điểm chết màn hình", "trả góp 0%").
  Giữ nguyên tên sản phẩm, mã sản phẩm (Pxxx), con số, mốc thời gian khách nêu. Không thêm điều kiện khách không nói.
- keywords: tối đa 8 từ khoá / cụm từ ngắn quan trọng nhất để tìm theo từ (BM25).
- Không trả lời câu hỏi. Không bịa thông tin."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries", "keywords"],
    "additionalProperties": False,
}


def _headings() -> str:
    from support.rag import ingest

    seen = OrderedDict()
    for c in ingest():
        seen.setdefault(f"- {c.doc} > {c.section}", None)
    return "\n".join(seen)


class LLMRewriter:
    """Claude structured output. Một instance Claude cho mỗi tracer để span ghi đúng file trace."""

    def __init__(self, model: str | None = None, effort: str | None = None, cache_size: int = 512):
        self.model = model or config.REWRITE_MODEL
        self.effort = effort or config.REWRITE_EFFORT
        self._system: str | None = None
        self._llms: dict[int, object] = {}
        self._cache: OrderedDict[str, tuple[list[str], list[str]]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def _llm(self, tracer: Tracer):
        from support.llm import Claude

        with self._lock:
            key = id(tracer)
            if key not in self._llms:
                self._llms[key] = Claude(model=self.model, effort=self.effort, tracer=tracer,
                                         name="query_rewrite", max_tokens=1000)
            return self._llms[key]

    @property
    def system(self) -> str:
        if self._system is None:
            self._system = REWRITE_SYSTEM.format(headings=_headings())
        return self._system

    def rewrite(self, base: RewrittenQuery, tracer: Tracer) -> RewrittenQuery:
        key = base.normalized.lower()
        with self._lock:
            hit = self._cache.get(key)
            if hit:
                self._cache.move_to_end(key)
        if hit:
            queries, keywords = hit
            return RewrittenQuery(base.original, base.normalized, list(queries), list(keywords), "llm", "cache")

        out = self._llm(tracer).structured(self.system, f"Câu hỏi của khách: {base.normalized}", REWRITE_SCHEMA)
        queries = _dedupe([q for q in out.get("queries", []) if isinstance(q, str)])[:3]
        keywords = _dedupe([k for k in out.get("keywords", []) if isinstance(k, str)])[:8]
        if not queries:
            raise ValueError("rewrite trả về queries rỗng")
        # luôn giữ câu đã chuẩn hoá ở cuối làm lưới an toàn khi LLM diễn giải lệch
        queries = _dedupe(queries + [base.normalized])[:4]
        with self._lock:
            self._cache[key] = (queries, keywords)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return RewrittenQuery(base.original, base.normalized, queries, keywords, "llm", "llm")


_LLM_REWRITER: LLMRewriter | None = None
_INIT_LOCK = threading.Lock()


def _get_llm_rewriter() -> LLMRewriter:
    global _LLM_REWRITER
    with _INIT_LOCK:
        if _LLM_REWRITER is None:
            _LLM_REWRITER = LLMRewriter()
        return _LLM_REWRITER


def current_mode() -> str:
    mode = os.environ.get("QUERY_REWRITE", config.QUERY_REWRITE).strip().lower()
    return mode if mode in MODES else "rules"


def rewrite_query(query: str, mode: str | None = None, tracer: Tracer | None = None) -> RewrittenQuery:
    """Điểm vào duy nhất. mode=None -> đọc env QUERY_REWRITE. tracer=None -> tracer của span đang mở (nếu có)."""
    mode = (mode or current_mode()).lower()
    tracer = tracer or current_tracer() or NULL_TRACER
    with tracer.span("rag.rewrite", kind="rag", mode=mode, query=query) as sp:
        if mode == "off" or not query.strip():
            rq = RewrittenQuery(original=query, normalized=query, queries=[query], mode="off", source="off")
        else:
            rq = rewrite_rules(query)
            if mode == "llm":
                try:
                    rq = _get_llm_rewriter().rewrite(rq, tracer)
                except Exception as e:                       # API lỗi / refusal / JSON hỏng -> dùng rules
                    log.warning("query rewrite LLM lỗi, dùng rules", extra={"error": f"{type(e).__name__}: {e}"})
                    rq.mode, rq.source = "llm", "fallback"
                    sp.set(error=f"{type(e).__name__}: {e}")
        sp.set(queries=rq.queries, keywords=rq.keywords, source=rq.source)
        log.debug("query rewritten", extra={"rewrite": rq.to_dict()})
        return rq
