# ============================================
# FILE: main.py
# PURPOSE: Query Classifier + ES + AI Answers
# ============================================

import json
import os
from config import API_KEY, MODEL, ENDPOINT, CSV_PATH
from classification_builder import build_grouped_examples, build_classification_prompt
from llm_client import LLMClient
from es_client import ESClient


def initialize_elasticsearch():
    """Initialize Elasticsearch client."""
    print("\n🔧 Initializing Elasticsearch...")
    try:
        es_client = ESClient()
        return es_client
    except Exception as e:
        print(f"   ⚠️ Could not connect to Elasticsearch: {e}")
        return None


def main():
    """Interactive query classification with ES search and AI answers."""
    print("=" * 60)
    print("QUERY CLASSIFIER + ELASTICSEARCH + AI ANSWERS")
    print("=" * 60)
    
    # ─── Initialize Classifier ───
    print("\n🔧 Initializing classifier...")
    
    grouped_examples = build_grouped_examples(CSV_PATH, max_examples_per_category=7)
    print("   ✅ Grouped examples built")
    
    system_prompt = build_classification_prompt(grouped_examples)
    print("   ✅ System prompt built")
    
    client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
    client.set_system_prompt(system_prompt)
    print("   ✅ LLM client ready")
    
    # ─── Initialize Elasticsearch ───
    es_client = initialize_elasticsearch()
    if es_client:
        client.set_es_client(es_client)
    
    print("\n" + "=" * 60)
    print("✅ READY! Ask any question about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("\n💡 Examples:")
    print("   - 'what was Ranbir's salary after his first increment?'")
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
        
        # ─── Classify + Search + Generate Answer ───
        result = client.answer_question(query)

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
                if source.get('year'):
                    print(f"      Year: {source['year']}")
                if source.get('document_type'):
                    print(f"      Type: {source['document_type']}")
        
        # ─── Show Classification ───
        if result.get('classification'):
            print("\n" + "-" * 60)
            print("🏷️ CLASSIFICATION")
            print("-" * 60)
            print(f"  Intent: {result['classification'].get('intent')}")
            print(f"  Spec Category: {result['classification'].get('spec_category')}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()