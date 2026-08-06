# ============================================
# FILE: main.py
# PURPOSE: Simple interactive classifier
# ============================================

import json
from config import API_KEY, MODEL, ENDPOINT, CSV_PATH
from classification_builder import build_grouped_examples, build_classification_prompt
from llm_client import LLMClient


# def print_grouped_structure(grouped_examples):
#     """Print the nested structure for debugging."""
#     print("\n📚 TRAINING DATA STRUCTURE:")
#     print("=" * 50)
#     for intent, spec_categories in grouped_examples.items():
#         print(f"\n{intent}: {{")
#         for spec_category, examples in spec_categories.items():
#             print(f"    {spec_category}: {{")
#             for example in examples:
#                 print(f"        route: \"{example['route']}\"")
#                 print(f"        es_index: {example['es_index']}")
#                 print(f"        search_strategy: {example['search_strategy']}")
#             print(f"    }}")
#         print(f"}}")
#     print("=" * 50)


def main():
    """Interactive query classification."""
    # print("=" * 60)
    print("QUERY CLASSIFIER (Single + Composite Support)")
    # print("=" * 60)
    
    # ─── Initialize ───
    # print("\n🔧 Initializing...")
    
    # Build grouped examples with nested structure
    grouped_examples = build_grouped_examples(CSV_PATH, max_examples_per_category=7)
    # print("✅ Grouped examples built")
    
    # Print the structure for debugging
    # print_grouped_structure(grouped_examples)
    
    # Build the prompt using the nested structure
    system_prompt = build_classification_prompt(grouped_examples)
    # print("\n✅ System prompt built")
    
    # Initialize LLM client
    client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
    client.set_system_prompt(system_prompt)
    # print("✅ Ready!\n")
    
    # ─── Interactive loop ───
    while True:
        query = input("\n🔍 Enter query (or 'exit'): ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # ─── Classify ───
        print("\n⏳ Classifying...")
        results = client.classify(query)
        
        # ─── Display results ───
        # print("\n" + "=" * 50)
        if len(results) == 1:
            print("CLASSIFICATION RESULT")
        else:
            print(f"COMPOSITE QUERY - {len(results)} PARTS")
        # print("=" * 50)
        
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                print(f"\n📌 Part {i}:")
            else:
                print()
            
            print(f"  Query:           {result.get('text', query)}")
            print(f"  Intent:          {result.get('intent', 'N/A')}")
            print(f"  Spec Category:   {result.get('spec_category', 'N/A')}")
            print(f"  Route:           {result.get('route', 'N/A')}")
            print(f"  ES Index:        {result.get('es_index', 'N/A')}")
            print(f"  Search Strategy: {result.get('search_strategy', 'N/A')}")
        
        # ─── JSON output ───
        # print("\n📋 JSON:")
        # print(json.dumps(results, indent=2))
        # print("=" * 50)


if __name__ == "__main__":
    main()