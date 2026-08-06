# ============================================
# FILE: main.py (LLM-ONLY APPROACH)
# PURPOSE: Pure LLM-based composite query classification
# ============================================

import os
import json
from dotenv import load_dotenv
from classification_builder import (
    build_grouped_examples,
    build_system_prompt_with_examples
)
from llm_client import LLMClient
from composite_analyzer import CompositeQueryProcessor

# ─── LOAD ENVIRONMENT VARIABLES ───
load_dotenv()

# ─── CONFIGURATION ───
CSV_PATH = "query_classifications.csv"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
    exit(1)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def initialize_classifier():
    """
    Initialize the classifier once.
    """
    print("🔧 Initializing classifier...")
    
    grouped_examples = build_grouped_examples(CSV_PATH, max_examples_per_category=10)
    print("✅ Grouped examples built from CSV")
    
    system_prompt = build_system_prompt_with_examples(grouped_examples)
    print("✅ System prompt built")
    
    client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
    client.set_system_prompt(system_prompt)
    print("✅ LLM client initialized")
    
    return client


def main():
    """
    Interactive query classification with LLM-based composite detection.
    """
    print("=" * 60)
    print("INTERACTIVE QUERY CLASSIFIER (LLM-Only Composite Detection)")
    print("=" * 60)
    
    client = initialize_classifier()
    processor = CompositeQueryProcessor(client)
    
    print("\n" + "=" * 60)
    print("READY! Enter your queries below.")
    print("Type 'exit' or 'quit' to stop.")
    print("\n💡 The LLM will automatically detect composite queries:")
    print("  - 'Find my resume and find my contracts'")
    print("  - 'Delete Draft.docx then move it to archive'")
    print("  - 'Compare these files and summarize them'")
    print("=" * 60)
    
    while True:
        query = input("\n🔍 Enter your query: ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            print("⚠️ Please enter a query.")
            continue
        
        try:
            # Process the query (LLM detects composite)
            results = processor.process(query)
            
            # Print results
            print("\n" + "=" * 50)
            if len(results) == 1:
                print("CLASSIFICATION RESULT")
            else:
                print(f"COMPOSITE QUERY - {len(results)} CLASSIFICATIONS")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                if len(results) > 1:
                    print(f"\n📌 Query {i}:")
                else:
                    print()
                print(f"  Intent:           {result.get('intent', 'N/A')}")
                print(f"  Spec Category:    {result.get('spec_category', 'N/A')}")
                print(f"  Route:            {result.get('route', 'N/A')}")
                print(f"  ES Index:         {result.get('es_index', 'N/A')}")
                print(f"  Search Strategy:  {result.get('search_strategy', 'N/A')}")
            
            print("\n" + "=" * 50)
            
            # JSON output
            print("\n📋 JSON Output:")
            print(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()