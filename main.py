# ============================================
# FILE: main.py
# PURPOSE: Query Classifier + PDF Vector Search + AI Answers
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
    for emp_id, emp_name in employees[:10]:  # Show first 10
        print(f"      - {emp_id}: {emp_name}")
    if len(employees) > 10:
        print(f"      ... and {len(employees) - 10} more")
    
    print(f"   ✅ Vector store built with {vector_store.size} vectors")
    
    return vector_store


def main():
    """Interactive query classification with vector search and AI answers."""
    print("=" * 60)
    print("QUERY CLASSIFIER + PDF VECTOR SEARCH + AI ANSWERS")
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
    print("✅ READY! Ask any question about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("\n💡 Examples:")
    print("   - 'when was Maya offered a job?'")
    print("   - 'show me increment letters for Advik'")
    print("   - 'what is the salary of Maya?'")
    print("   - 'find documents for EMP001'")
    print("=" * 60)
    
    # ─── Interactive loop ───
    while True:
        query = input("\n🔍 Ask a question (or 'exit'): ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n⏳ Searching and generating answer...")
        
        # ─── Step 1: Classify (optional - uncomment if needed) ───
        # classifications = client.classify(query)
        
        # ─── Step 2: Search and Generate Answer ───
        if vector_store:
            result = client.answer_question(query, k=5)
            
            # ─── Display Answer ───
            print("\n" + "=" * 60)
            print("📝 ANSWER")
            print("=" * 60)
            print(f"\n{result['answer']}")
            
            # ─── Display Sources ───
            if result.get('sources'):
                print("\n" + "-" * 60)
                print("📄 SOURCES")
                print("-" * 60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n  [{i}] {source.get('filename', 'Unknown')}")
                    print(f"      Employee: {source.get('employee', 'N/A')}")
                    print(f"      Score: {source.get('score', 0):.4f}")
            
            # ─── Optional: Show raw search results ───
            if result.get('chunks'):
                print("\n" + "-" * 60)
                print("🔍 RAW SEARCH RESULTS (for debugging)")
                print("-" * 60)
                for i, (chunk, score) in enumerate(result['chunks'][:3], 1):
                    print(f"\n  [{i}] Score: {score:.4f}")
                    print(f"      File: {chunk.filename}")
                    preview = chunk.text[:150].replace('\n', ' ')
                    print(f"      Preview: {preview}...")
        
        else:
            # Fallback: Only search results (no LLM)
            search_results = vector_store.search(query, k=7) if vector_store else []
            
            if search_results:
                print("\n" + "-" * 50)
                print("PDF SEARCH RESULTS")
                print("-" * 50)
                
                for i, (chunk, score) in enumerate(search_results, 1):
                    print(f"\n  [{i}] Score: {score:.4f}")
                    print(f"      Employee: {chunk.employee_id} - {chunk.employee_name}")
                    print(f"      File: {chunk.filename}")
                    preview = chunk.text[:200].replace('\n', ' ')
                    print(f"      Preview: {preview}...")
            else:
                print("\n❌ No results found.")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()