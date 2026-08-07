"""Load HR/DMS data (CSV rows + PDF text) into searchable text chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
import os

ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(ROOT)))

@dataclass
class Chunk:
    """A single piece of retrievable text plus its provenance."""

    text: str
    source: str  # relative path or logical source name
    type: str  # "csv" | "pdf"
    meta: dict = field(default_factory=dict)


_ID_RE = re.compile(r"(EMP\d{3}|CTR\d{3})", re.IGNORECASE)


def _entity_id(path: Path) -> str | None:
    m = _ID_RE.search(str(path))
    return m.group(1).upper() if m else None


def extract_entity_ids(text: str) -> list[str]:
    """Pull EMP###/CTR### identifiers out of free text (upper-cased, de-duped).

    Used to scope retrieval to the entity a question names, so a query about one
    employee cannot pull in near-identical documents belonging to others.
    """
    seen: dict[str, None] = {}
    for m in _ID_RE.finditer(text or ""):
        seen[m.group(1).upper()] = None
    return list(seen)


def _csv_chunks(data_dir: Path) -> list[Chunk]:
    """One chunk per CSV row, rendered as `header \n col: value` lines."""
    chunks: list[Chunk] = []
    meta_dir = data_dir / "metadata"
    if not meta_dir.is_dir():
        return chunks

    for csv_path in sorted(meta_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as exc:  
            print(f"[ingest] skipping {csv_path.name}: {exc}")
            continue

        rel = csv_path.relative_to(data_dir).as_posix()
        for idx, row in df.iterrows():
            body = "\n".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
            if not body.strip():
                continue
            text = f"[{csv_path.stem}] row {idx + 1}\n{body}"
            eid = None
            for key in ("emp_id", "contract_id"):
                if key in row and str(row[key]).strip():
                    eid = str(row[key]).strip().upper()
                    break
            chunks.append(
                Chunk(text=text, source=rel, type="csv", meta={"entity_id": eid})
            )
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        parts.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return parts


def _pdf_chunks(data_dir: Path) -> list[Chunk]:
    """Extract text from every PDF under contracts/ and employees/."""
    chunks: list[Chunk] = []
    for sub in ("employees"):
        base = data_dir / sub
        if not base.is_dir():
            continue
        for pdf_path in sorted(base.rglob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
                raw = "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest] skipping {pdf_path.name}: {exc}")
                continue

            rel = pdf_path.relative_to(data_dir).as_posix()
            eid = _entity_id(pdf_path)
            for piece in _split(raw, CHUNK_SIZE, CHUNK_OVERLAP):
                header = f"[{rel}]"
                if eid:
                    header += f" ({eid})"
                chunks.append(
                    Chunk(
                        text=f"{header}\n{piece}",
                        source=rel,
                        type="pdf",
                        meta={"entity_id": eid},
                    )
                )
    return chunks


def load_chunks(data_dir: Path | None = None) -> list[Chunk]:
    """Load all CSV + PDF chunks from the data directory."""
    data_dir = Path(data_dir or config.DATA_DIR)
    chunks = _csv_chunks(data_dir) + _pdf_chunks(data_dir)
    print(f"[ingest] loaded {len(chunks)} chunks from {data_dir}")
    return chunks