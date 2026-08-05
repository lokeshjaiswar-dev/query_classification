# ============================================
# FILE: main.py
# PURPOSE: Interactive Query Classification
# ============================================

import os
import json
from dotenv import load_dotenv
from classification_builder import (
    build_grouped_examples,
    build_system_prompt_with_examples
)
from llm_client import LLMClient

# ─── LOAD ENVIRONMENT VARIABLES ───
load_dotenv()

# ─── CONFIGURATION ───
CSV_PATH = "query_classifications.csv"

# OpenRouter Configuration
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Get API key from environment variable
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
    print("Please create a .env file with: OPENROUTER_API_KEY=sk-or-v1-...")
    exit(1)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def initialize_classifier():
    """
    Initialize the classifier once.
    Builds context from CSV and sets it in the LLM client.
    """
    print("🔧 Initializing classifier...")
    
    # Step 1: Build grouped examples from CSV
    grouped_examples = build_grouped_examples(CSV_PATH, max_examples_per_category=3)
    print("✅ Grouped examples built from CSV")
    
    # Step 2: Build the system prompt with examples
    system_prompt = build_system_prompt_with_examples(grouped_examples)
    # print("prompt :", system_prompt)
    print("✅ System prompt built")
    
    # Step 3: Create LLM client and set system prompt
    client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
    client.set_system_prompt(system_prompt)
    print("✅ LLM client initialized")
    
    return client


def classify_query(client: LLMClient, query: str) -> dict:
    """
    Classify a single query using the initialized client.
    """
    return client.classify_query(query)


def print_result(result: dict):
    """
    Pretty print the classification result.
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"  Intent:           {result.get('intent', 'N/A')}")
    print(f"  Spec Category:    {result.get('spec_category', 'N/A')}")
    print(f"  Route:            {result.get('route', 'N/A')}")
    print(f"  ES Index:         {result.get('es_index', 'N/A')}")
    print(f"  Search Strategy:  {result.get('search_strategy', 'N/A')}")
    print("=" * 50)


def print_json(result: dict):
    """
    Print the result as JSON.
    """
    print("\n" + "=" * 50)
    print("JSON OUTPUT")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print("=" * 50)


def main():
    """
    Interactive query classification.
    """
    # ─── Initialize once ───
    print("=" * 60)
    print("INTERACTIVE QUERY CLASSIFIER")
    print("=" * 60)
    
    client = initialize_classifier()
    
    print("\n" + "=" * 60)
    print("READY! Enter your queries below.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    
    # ─── Interactive loop ───
    while True:
        # Get user input
        query = input("\n🔍 Enter your query: ").strip()
        
        # Check for exit
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty queries
        if not query:
            print("⚠️ Please enter a query.")
            continue
        
        # Classify the query
        try:
            result = classify_query(client, query)
            
            # Print both formats
            print_result(result)
            # print_json(result)
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()