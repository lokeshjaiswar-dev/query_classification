#!/usr/bin/env python3
"""
Run Elasticsearch Ingestion
Usage: python run_es_ingest.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from es_ingest import ESIngestor
from config import ES_HOST, ES_PORT, DATA_DIR


def main():
    print("\n" + "="*70)
    print("🚀 ELASTICSEARCH INGESTION SCRIPT")
    print("="*70)
    
    # ─── Check data directory ───
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Please create the 'employees' folder with EMP* subfolders")
        return
    
    # Count employee folders
    emp_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("EMP")]
    if not emp_folders:
        print(f"\n❌ No employee folders found in {data_dir}")
        print("   Expected structure: employees/EMP001_Advik_Maharaj/")
        return
    
    print(f"\n📁 Found {len(emp_folders)} employee folders")
    for folder in emp_folders[:5]:
        print(f"   - {folder.name}")
    if len(emp_folders) > 5:
        print(f"   ... and {len(emp_folders) - 5} more")
    
    # ─── Initialize ingestor ───
    try:
        ingestor = ESIngestor(host=ES_HOST, port=ES_PORT)
    except ConnectionError as e:
        print(f"\n❌ {e}")
        print("   Please start Elasticsearch first:")
        print("   docker-compose up -d")
        return
    
    # ─── Create indices ───
    print("\n🔧 Creating indices...")
    ingestor.create_indices()
    
    # ─── Ingest all data ───
    print("\n📥 Starting ingestion...")
    stats = ingestor.ingest_all()
    
    # ─── Show stats ───
    print("\n📊 INDEX STATISTICS:")
    stats = ingestor.get_stats()
    for index_name, count in stats.items():
        print(f"   {index_name}: {count} documents")
    
    # ─── Show sample data ───
    # ingestor.show_sample_data()
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("   Kibana: http://localhost:5601")
    print("   Elasticsearch: http://localhost:9200")
    print("="*70)


if __name__ == "__main__":
    main()