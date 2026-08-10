from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
import os

from config import DATA_DIR

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "250"))


@dataclass
class Chunk:
    """A single piece of retrievable text plus its provenance."""
    text: str
    source: str
    type: str
    employee_id: str
    employee_name: str
    filename: str
    meta: dict = field(default_factory=dict)


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
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


def _extract_employee_info(path: Path) -> tuple[str, str]:
    """Extract employee ID and name from folder name."""
    folder_name = path.name
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("EMP"):
        return parts[0], parts[1]
    return "", ""


def load_chunks(data_dir: Path | str | None = None) -> list[Chunk]:
    """Load all PDF chunks from nested employee folders."""
    if data_dir is None:
        data_dir = Path(DATA_DIR)
    else:
        data_dir = Path(data_dir)
    
    print(f"[ingest] 📂 Loading PDFs from: {data_dir}")
    
    if not data_dir.exists():
        print(f"[ingest] ⚠️ Directory not found: {data_dir}")
        return []
    
    chunks: list[Chunk] = []
    employee_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("EMP")]
    
    if not employee_folders:
        print(f"[ingest] ⚠️ No employee folders found in {data_dir}")
        return []
    
    print(f"[ingest] 📁 Found {len(employee_folders)} employee folders")
    
    total_pdfs = 0
    
    for emp_folder in employee_folders:
        emp_id, emp_name = _extract_employee_info(emp_folder)
        
        if not emp_id:
            print(f"[ingest] ⚠️ Skipping invalid folder: {emp_folder.name}")
            continue
        
        pdf_files = list(emp_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"[ingest] ⚠️ No PDFs in {emp_folder.name}")
            continue
        
        print(f"[ingest] 📄 {emp_id} ({emp_name}): {len(pdf_files)} PDFs")
        total_pdfs += len(pdf_files)
        
        for pdf_path in pdf_files:
            try:
                reader = PdfReader(str(pdf_path))
                raw = "\n".join((page.extract_text() or "") for page in reader.pages)
                
                if not raw.strip():
                    print(f"[ingest] ⚠️ No text extracted from {pdf_path.name}")
                    continue
                    
            except Exception as exc:
                print(f"[ingest] ⚠️ Skipping {pdf_path.name}: {exc}")
                continue

            rel = pdf_path.relative_to(data_dir).as_posix()
            
            # ─── KEY CHANGE: Split and create enriched text ───
            for piece in _split(raw, CHUNK_SIZE, CHUNK_OVERLAP):
                # ─── DYNAMIC METADATA: Build from available data ───
                
                # 1. Employee info (dynamically extracted from folder)
                employee_info = f"Employee: {emp_name} (ID: {emp_id})"
                
                # 2. File info (dynamically from PDF)
                file_info = f"Document: {pdf_path.name}"
                
                # 3. Document type (dynamically from filename)
                # Remove extension, replace underscores with spaces, title case
                doc_type = pdf_path.stem.replace("_", " ").title()
                doc_info = f"Type: {doc_type}"
                
                # ─── Combine everything dynamically ───
                # No hardcoded values - everything comes from the actual file/folder
                enriched_text = f"""
{employee_info}
{file_info}
{doc_info}

{piece}
"""
                
                chunks.append(
                    Chunk(
                        text=enriched_text,  # ← Now includes dynamic metadata!
                        source=rel,
                        type="pdf",
                        employee_id=emp_id,
                        employee_name=emp_name,
                        filename=pdf_path.name,
                        meta={
                            "employee_id": emp_id,
                            "employee_name": emp_name,
                            "filename": pdf_path.name
                        }
                    )
                )
    
    print(f"[ingest] ✅ Loaded {len(chunks)} chunks from {total_pdfs} PDFs across {len(employee_folders)} employees")
    return chunks