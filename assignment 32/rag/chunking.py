"""Bước 1b: 3 chiến lược chunking.

- fixed_token_chunks : cửa sổ trượt cố định N token (+ overlap), cắt bất kể câu/đoạn.
- recursive_chunks   : thử tách theo separator "thô" trước (đoạn -> dòng -> câu -> từ),
                       chỉ xuống separator mịn hơn khi mảnh vẫn còn quá dài,
                       rồi gộp các mảnh nhỏ liền kề lại cho tới gần chunk_size.

Mọi chunk đều giữ (start, end) = vị trí ký tự trong tài liệu gốc.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

Span = Tuple[int, int]

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass
class Chunk:
    id: int
    text: str
    start: int          # vị trí ký tự bắt đầu trong tài liệu
    end: int            # vị trí ký tự kết thúc
    n_tokens: int
    strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Chiến lược 1 & 2: fixed-size theo token
# --------------------------------------------------------------------------- #
def fixed_token_spans(text: str, tokenizer, chunk_size: int, overlap: int = 0) -> List[Span]:
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")
    offs = tokenizer.offsets(text)
    spans: List[Span] = []
    step = chunk_size - overlap
    i = 0
    while i < len(offs):
        j = min(i + chunk_size, len(offs))
        spans.append((offs[i][0], offs[j - 1][1]))
        if j == len(offs):
            break
        i += step
    return spans


# --------------------------------------------------------------------------- #
# Chiến lược 3: recursive character/token splitter (tự cài, kiểu LangChain)
# --------------------------------------------------------------------------- #
def _split_on(text: str, start: int, end: int, sep: str) -> List[Span]:
    """Tách [start, end) theo sep; separator được dính vào cuối mảnh trước
    để các mảnh liền mạch (ghép lại = đúng đoạn gốc)."""
    pieces: List[Span] = []
    cur = start
    while True:
        idx = text.find(sep, cur, end)
        if idx == -1:
            break
        cut = idx + len(sep)
        pieces.append((cur, cut))
        cur = cut
    if cur < end:
        pieces.append((cur, end))
    return pieces


def _merge(spans: List[Span], counts: List[int], chunk_size: int) -> List[Span]:
    """Gộp tham lam các mảnh liền kề cho tới khi sắp vượt chunk_size."""
    merged: List[Span] = []
    cur_s, cur_e, cur_n = None, None, 0
    for (s, e), n in zip(spans, counts):
        if cur_s is None:
            cur_s, cur_e, cur_n = s, e, n
        elif cur_n + n <= chunk_size:
            cur_e, cur_n = e, cur_n + n
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e, cur_n = s, e, n
    if cur_s is not None:
        merged.append((cur_s, cur_e))
    return merged


def _recursive(text, start, end, separators, chunk_size, tokenizer) -> List[Span]:
    if tokenizer.count(text[start:end]) <= chunk_size:
        return [(start, end)]

    # Chọn separator "thô" nhất có xuất hiện trong đoạn này
    for i, sep in enumerate(separators):
        if sep == "":
            # Hết cách: cắt cứng theo token
            sub = fixed_token_spans(text[start:end], tokenizer, chunk_size, 0)
            return [(start + s, start + e) for s, e in sub]
        if text.find(sep, start, end) != -1:
            rest = separators[i + 1:]
            break

    good: List[Span] = []
    for s, e in _split_on(text, start, end, sep):
        if tokenizer.count(text[s:e]) <= chunk_size:
            good.append((s, e))
        else:
            good.extend(_recursive(text, s, e, rest, chunk_size, tokenizer))
    counts = [tokenizer.count(text[s:e]) for s, e in good]
    return _merge(good, counts, chunk_size)


def recursive_spans(text: str, tokenizer, chunk_size: int,
                    separators: Sequence[str] = DEFAULT_SEPARATORS) -> List[Span]:
    return _recursive(text, 0, len(text), list(separators), chunk_size, tokenizer)


# --------------------------------------------------------------------------- #
# API chung
# --------------------------------------------------------------------------- #
def _strip(text: str, s: int, e: int) -> Span:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def chunk_document(text: str, tokenizer, strategy_name: str, strategy_cfg: dict) -> List[Chunk]:
    kind = strategy_cfg["type"]
    size = strategy_cfg["chunk_size"]
    if kind == "fixed":
        spans = fixed_token_spans(text, tokenizer, size, strategy_cfg.get("overlap", 0))
    elif kind == "recursive":
        spans = recursive_spans(text, tokenizer, size,
                                strategy_cfg.get("separators", DEFAULT_SEPARATORS))
    else:
        raise ValueError(f"Strategy type không hợp lệ: {kind}")

    chunks: List[Chunk] = []
    for s, e in spans:
        s, e = _strip(text, s, e)
        if e <= s:
            continue
        piece = text[s:e]
        chunks.append(Chunk(id=len(chunks), text=piece, start=s, end=e,
                            n_tokens=tokenizer.count(piece), strategy=strategy_name))
    return chunks
