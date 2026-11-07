"""Bước 1a: đọc tài liệu PDF / TXT / MD thành một chuỗi text sạch."""
from __future__ import annotations

import re
from pathlib import Path


def load_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Không hỗ trợ định dạng {suffix} (chỉ .pdf/.txt/.md)")
    return clean_text(text)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # import lazy: chỉ cần khi đọc PDF

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    # Ngăn cách các trang bằng dòng trống để recursive splitter coi như đoạn mới
    return "\n\n".join(pages)   


def clean_text(text: str) -> str:
    """Làm sạch nhẹ nhưng GIỮ cấu trúc đoạn (\n\n) cho recursive chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ")      #thay các non-breaking space bằng spce thông thường
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # nối từ bị gạch nối cuối dòng (PDF)
    text = re.sub(r"[ \t]+\n", "\n", text)             # bỏ khoảng trắng cuối dòng
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)             # tối đa 1 dòng trống
    return text.strip()
