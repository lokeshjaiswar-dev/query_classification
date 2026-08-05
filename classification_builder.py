# ============================================
# FILE: classification_builder.py
# PURPOSE: Build classification context from CSV
# ============================================

import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List


def build_grouped_examples(
    csv_path: str,
    max_examples_per_category: int = 3
) -> Dict[str, Any]:
    """
    Build grouped examples from CSV.
    
    Args:
        csv_path: Path to the CSV file
        max_examples_per_category: Max examples per spec category
    
    Returns:
        Grouped dictionary with all fields
    """
    # Step 1: Load CSV
    df = pd.read_csv(csv_path)
    
    # Step 2: Create nested dictionary
    grouped = defaultdict(lambda: defaultdict(list))
    
    # Step 3: Loop through each row
    for _, row in df.iterrows():
        intent = row["Intent (intent_analysis)"]
        spec_category = row["Spec Category"]
        
        # Build the example object
        example = {
            "query": row["Query"],
            "route": row["Route / Handler"],
            "es_index": row.get("ES Index (if search)", ""),
            "search_strategy": row.get("Search Strategy", "")
        }
        
        # Clean up empty values
        if example["es_index"] == "-" or example["es_index"] == "":
            example["es_index"] = None
        if example["search_strategy"] == "-" or example["search_strategy"] == "":
            example["search_strategy"] = None
        
        # Add to group (limit per category)
        if len(grouped[intent][spec_category]) < max_examples_per_category:
            grouped[intent][spec_category].append(example)
    
    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in grouped.items()}


def build_system_prompt_with_examples(grouped_examples: Dict) -> str:
    """
    Build the system prompt with grouped examples.
    """
    # Convert grouped examples to a readable string
    examples_text = ""
    
    for intent, spec_categories in grouped_examples.items():
        examples_text += f"\n## INTENT: {intent.upper()}\n"
        
        for spec_category, examples in spec_categories.items():
            examples_text += f"\n### {spec_category}\n"
            
            for example in examples:
                query = example.get("query", "")
                route = example.get("route", "")
                es_index = example.get("es_index")
                search_strategy = example.get("search_strategy")
                
                examples_text += f'  - Query: "{query}"\n'
                examples_text += f"    Route: {route}\n"
                
                if es_index:
                    examples_text += f"    ES Index: {es_index}\n"
                if search_strategy:
                    examples_text += f"    Search Strategy: {search_strategy}\n"
    
    system_prompt = f"""
You are a query classifier for a document management system.

Your job is to classify a user's query based on the examples below.

Here are the classification rules and examples:

{examples_text}

When a user asks a query, return ONLY a JSON object with these fields:
{{
  "intent": "search or rag_content or conversation or action",
  "spec_category": "File Retrieval or Pay Slip or Contracts or etc.",
  "route": "Elasticsearch search or RAG content analysis or etc.",
  "es_index": "document or master or masterrecord or doctype or folder or null",
  "search_strategy": "BFS or DFS or BFS->DFS or null"
}}

Do not add any extra text. Return ONLY the JSON.
"""
    
    return system_prompt

