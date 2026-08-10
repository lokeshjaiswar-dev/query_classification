# ============================================
# FILE: main.py
# PURPOSE: Query Classifier + PDF Vector Search
# ============================================

import json
import os
from config import API_KEY, MODEL, ENDPOINT, CSV_PATH, DATA_DIR
from classification_builder import build_grouped_examples, build_classification_prompt
from llm_client import LLMClient
from ingest import load_chunks
from vector_store import VectorStore


def initialize_vector_store():
    """Load PDFs and build vector store."""
    print("\n🔧 Loading PDFs and building vector store...")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"   ⚠️ Data directory not found: {DATA_DIR}")
        print(f"   ⚠️ Vector search will be disabled.")
        return None
        
    chunks = load_chunks()
    
    if not chunks:
        print("   ⚠️ No chunks created. Vector search will be disabled.")
        return None
    
    # Build vector store
    vector_store = VectorStore()
    vector_store.build(chunks)
    
    # Show employee information
    employees = vector_store.get_employees()
    print(f"\n   👥 Employees loaded:")
    for emp_id, emp_name in employees:
        print(f"      - {emp_id}: {emp_name}")
    
    print(f"   ✅ Vector store built with {vector_store.size} vectors")
    
    return vector_store


def main():
    """Interactive query classification with vector search."""
    print("=" * 60)
    print("QUERY CLASSIFIER + PDF VECTOR SEARCH")
    print("=" * 60)
    
    # ─── Initialize Classifier ───
    print("\n🔧 Initializing classifier...")
    
    # Build grouped examples
    grouped_examples = build_grouped_examples(CSV_PATH, max_examples_per_category=7)
    print("   ✅ Grouped examples built")
    
    # Build the prompt
    system_prompt = build_classification_prompt(grouped_examples)
    print("   ✅ System prompt built")
    
    # Initialize LLM client
    client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
    client.set_system_prompt(system_prompt)
    print("   ✅ LLM client ready")
    
    # ─── Initialize Vector Store ───
    vector_store = initialize_vector_store()
    
    # Set vector store in client
    if vector_store:
        client.set_vector_store(vector_store)
    
    print("\n" + "=" * 60)
    print("✅ READY! Enter your queries below.")
    print("Type 'exit' or 'quit' to stop.")
    print("\n💡 Examples:")
    print("   - 'find resume of Advik'")
    print("   - 'show me documents for EMP001'")
    print("   - 'find increment letters'")
    print("   - 'search offer letters'")
    print("=" * 60)
    
    # ─── Interactive loop ───
    while True:
        query = input("\n🔍 Enter query (or 'exit'): ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # ─── Classify and Search ───
        print("\n⏳ Processing...")
        
        # Step 1: Classify
        classifications = []
        
        # Step 2: Search vector store
        search_results = []
        if vector_store:
            search_results = vector_store.search(query, k=7)
        
        # ─── Display Classification Results ───
        print("\n" + "=" * 50)
        if len(classifications) == 1:
            print("CLASSIFICATION RESULT")
        else:
            print(f"COMPOSITE QUERY - {len(classifications)} PARTS")
        print("=" * 50)
        
        for i, result in enumerate(classifications, 1):
            if len(classifications) > 1:
                print(f"\n📌 Part {i}:")
            else:
                print()
            
            print(f"  Query:           {result.get('text', query)}")
            print(f"  Intent:          {result.get('intent', 'N/A')}")
            print(f"  Spec Category:   {result.get('spec_category', 'N/A')}")
            print(f"  Route:           {result.get('route', 'N/A')}")
            print(f"  ES Index:        {result.get('es_index', 'N/A')}")
            print(f"  Search Strategy: {result.get('search_strategy', 'N/A')}")
        
        # ─── Display Search Results ───
        if search_results:
            print("\n" + "-" * 50)
            print("PDF SEARCH RESULTS")
            print("-" * 50)
            
            for i, (chunk, score) in enumerate(search_results, 1):
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      Employee: {chunk.employee_id} - {chunk.employee_name}")
                print(f"      File: {chunk.filename}")
                print(f"      Source: {chunk.source}")
                preview = chunk.text[:200].replace('\n', ' ')
                print(f"      Preview: {preview}...")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()