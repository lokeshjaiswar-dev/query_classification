# ============================================
# FILE: ingest.py
# PURPOSE: Load PDF documents from nested employee folders
# ============================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
import os

from config import DATA_DIR

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))


@dataclass
class Chunk:
    """A single piece of retrievable text plus its provenance."""
    text: str
    source: str          # Relative path like "EMP001_Advik_Maharaj/resume.pdf"
    type: str            # "pdf"
    employee_id: str     # "EMP001"
    employee_name: str   # "Advik_Maharaj"
    filename: str        # "resume.pdf"
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
    """
    Extract employee ID and name from folder name.
    Example: "EMP001_Advik_Maharaj" → ("EMP001", "Advik_Maharaj")
    """
    folder_name = path.name
    parts = folder_name.split("_", 1)  # Split on first underscore only
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
    
    # Find all employee folders (any folder starting with "EMP")
    employee_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("EMP")]
    
    if not employee_folders:
        print(f"[ingest] ⚠️ No employee folders found in {data_dir}")
        print(f"[ingest] 💡 Expected structure: employees/EMP001_Name/ *.pdf")
        return []
    
    print(f"[ingest] 📁 Found {len(employee_folders)} employee folders")
    
    total_pdfs = 0
    
    for emp_folder in employee_folders:
        emp_id, emp_name = _extract_employee_info(emp_folder)
        
        if not emp_id:
            print(f"[ingest] ⚠️ Skipping invalid folder: {emp_folder.name}")
            continue
        
        # Find all PDFs in this employee folder
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
            
            # Split into chunks
            for piece in _split(raw, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(
                    Chunk(
                        text=piece,
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