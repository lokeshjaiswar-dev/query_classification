"""
Elasticsearch Ingestion Module
"""
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict
from elasticsearch import Elasticsearch
from pypdf import PdfReader

from config import ES_HOST, ES_PORT, ES_INDEX_MASTER, ES_INDEX_DOCUMENTS, DATA_DIR


class ESIngestor:
    """Handles Elasticsearch ingestion"""

    def __init__(self, host=ES_HOST, port=ES_PORT):
        # ─── ES 8.x Connection ───
        self.es = Elasticsearch(
            [{'host': host, 'port': port, 'scheme': 'http'}],
            request_timeout=60,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        self.master_index = ES_INDEX_MASTER
        self.documents_index = ES_INDEX_DOCUMENTS

        # Test connection
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {host}:{port}")
        print(f"[es_ingest] ✅ Connected to Elasticsearch")

    def create_indices(self):
        """Create indices with mappings"""
        
        master_mapping = {
            "mappings": {
                "properties": {
                    "emp_id": {"type": "keyword"},
                    "emp_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "emp_number": {"type": "keyword"},
                    "first_name": {"type": "text"},
                    "last_name": {"type": "text"},
                    "folder_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "document_count": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }

        documents_mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "emp_id": {"type": "keyword"},
                    "emp_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "employee_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "file_extension": {"type": "keyword"},
                    "file_size": {"type": "long"},
                    "document_type": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "increment_number": {"type": "integer"},
                    "extracted_text": {"type": "text"},
                    "created_at": {"type": "date"}
                }
            }
        }

        for index_name, mapping in [
            (self.master_index, master_mapping),
            (self.documents_index, documents_mapping)
        ]:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                print(f"[es_ingest] 🗑️ Deleted: {index_name}")

            self.es.indices.create(index=index_name, body=mapping)
            print(f"[es_ingest] ✅ Created: {index_name}")

    def extract_metadata(self, filename: str) -> Dict:
        """Extract metadata from filename"""
        metadata = {
            'document_type': 'other',
            'year': None,
            'increment_number': None
        }
        
        filename_lower = filename.lower()
        
        if 'offer_letter' in filename_lower:
            metadata['document_type'] = 'offer_letter'
        elif 'appointment_letter' in filename_lower:
            metadata['document_type'] = 'appointment_letter'
        elif 'increment_letter' in filename_lower:
            metadata['document_type'] = 'increment_letter'
            year_match = re.search(r'20\d{2}', filename)
            if year_match:
                metadata['year'] = int(year_match.group())
        elif 'transfer_order' in filename_lower:
            metadata['document_type'] = 'transfer_order'
        elif 'degree_certificate' in filename_lower:
            metadata['document_type'] = 'degree'
        elif 'aadhaar' in filename_lower:
            metadata['document_type'] = 'aadhaar'
        elif 'pan' in filename_lower:
            metadata['document_type'] = 'pan'
        elif 'document_checklist' in filename_lower:
            metadata['document_type'] = 'checklist'
        
        return metadata



    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text
        except Exception as e:
            print(f"   ⚠️ Error reading {pdf_path.name}: {e}")
            return ""

    def ingest_all(self) -> Dict:
        """Main ingestion"""
        print("\n" + "="*60)
        print("📥 ELASTICSEARCH INGESTION STARTED")
        print("="*60)

        data_dir = Path(DATA_DIR)
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return {'employees': 0, 'documents': 0}

        emp_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("EMP")]
        print(f"\n📁 Found {len(emp_folders)} employee folders")

        employee_count = 0
        document_count = 0

        for emp_folder in emp_folders:
            parts = emp_folder.name.split("_", 1)
            if len(parts) != 2:
                continue
            
            emp_id = parts[0]
            emp_name = parts[1].replace('_', ' ')
            name_parts = emp_name.split()
            first_name = name_parts[0] if name_parts else ''
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
            
            print(f"\n👤 {emp_id}: {emp_name}")
            
            pdf_files = list(emp_folder.glob("*.pdf"))
            print(f"   📄 {len(pdf_files)} PDF files")
            
            # Index Employee
            employee_doc = {
                'emp_id': emp_id,
                'emp_name': emp_name,
                'emp_number': emp_id,
                'first_name': first_name,
                'last_name': last_name,
                'folder_path': str(emp_folder),
                'document_count': len(pdf_files),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            self.es.index(index=self.master_index, id=emp_id, body=employee_doc)
            print(f"   ✅ Employee indexed")
            employee_count += 1
            
            # Index Documents
            for pdf_path in pdf_files:
                metadata = self.extract_metadata(pdf_path.name)
                text = self.extract_text_from_pdf(pdf_path)
                file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
                
                doc_id = f"{emp_id}_{pdf_path.name}"
                
                document_doc = {
                    'doc_id': doc_id,
                    'emp_id': emp_id,
                    'emp_name': emp_name,
                    'employee_name': emp_name,
                    'file_name': pdf_path.name,
                    'file_path': str(pdf_path),
                    'file_extension': pdf_path.name.split('.')[-1] if '.' in pdf_path.name else '',
                    'file_size': file_size,
                    'document_type': metadata['document_type'],
                    'year': metadata['year'],
                    'increment_number': metadata['increment_number'],
                    'extracted_text': text[:10000],
                    'created_at': datetime.now().isoformat()
                }
                
                self.es.index(index=self.documents_index, id=doc_id, body=document_doc)
                document_count += 1
                year_str = f"(Year: {metadata['year']})" if metadata['year'] else ""
                print(f"      📄 {pdf_path.name} → {metadata['document_type']} {year_str}")

        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE")
        print(f"   👤 Employees indexed: {employee_count}")
        print(f"   📄 Documents indexed: {document_count}")
        print("="*60)

        return {
            'employees': employee_count,
            'documents': document_count
        }

    def get_stats(self) -> Dict:
        """Get index statistics"""
        stats = {}
        for index_name in [self.master_index, self.documents_index]:
            if self.es.indices.exists(index=index_name):
                count = self.es.count(index=index_name)['count']
                stats[index_name] = count
            else:
                stats[index_name] = 0
        return stats