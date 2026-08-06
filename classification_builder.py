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
            # "query": row["Query"],
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
    
    # ─── 1. Get all unique intents ───
    intents = list(grouped_examples.keys())
    intents_str = ", ".join(intents)
    print(f"Intents found: {intents_str}")
    
    # ─── 2. Build examples text ───
    examples_text = ""
    
    for intent, spec_categories in grouped_examples.items():
        examples_text += f"\n## INTENT: {intent.upper()}\n"
        
        for spec_category, examples in spec_categories.items():
            examples_text += f"\n### {spec_category}\n"
            
            for example in examples:
                # query = example.get("query", "")
                route = example.get("route", "")
                es_index = example.get("es_index")
                search_strategy = example.get("search_strategy")
                
                # examples_text += f'  - Query: "{query}"\n'
                examples_text += f"    Route: {route}\n"
                
                if es_index:
                    examples_text += f"    ES Index: {es_index}\n"
                if search_strategy:
                    examples_text += f"    Search Strategy: {search_strategy}\n"
    
    # ─── 3. Get all unique spec categories ───
    all_spec_categories = set()
    for intent, spec_categories in grouped_examples.items():
        for spec_category in spec_categories.keys():
            all_spec_categories.add(spec_category)
    spec_categories_str = ", ".join(sorted(all_spec_categories))
    
    # ─── 4. Get all unique routes ───
    all_routes = set()
    for intent, spec_categories in grouped_examples.items():
        for spec_category, examples in spec_categories.items():
            for example in examples:
                route = example.get("route", "")
                if route:
                    all_routes.add(route)
    routes_str = ", ".join(sorted(all_routes))
    
    # ─── 5. Get all unique ES indexes ───
    all_es_indexes = set()
    for intent, spec_categories in grouped_examples.items():
        for spec_category, examples in spec_categories.items():
            for example in examples:
                es_index = example.get("es_index")
                # ✅ FIX: Only add if it's a string
                if es_index and isinstance(es_index, str):
                    all_es_indexes.add(es_index)
    es_indexes_str = ", ".join(sorted(all_es_indexes))
    
    # ─── 6. Get all unique search strategies ───
    all_search_strategies = set()
    for intent, spec_categories in grouped_examples.items():
        for spec_category, examples in spec_categories.items():
            for example in examples:
                search_strategy = example.get("search_strategy")
                # ✅ FIX: Only add if it's a string
                if search_strategy and isinstance(search_strategy, str):
                    all_search_strategies.add(search_strategy)
    search_strategies_str = ", ".join(sorted(all_search_strategies))

    print(f"Spec Categories found: {spec_categories_str}")
    print(f"Routes found: {routes_str}")
    print(f"ES Indexes found: {es_indexes_str}")
    print(f"Search Strategies found: {search_strategies_str}")
    
    
    # ─── 7. Build the system prompt ───
    system_prompt = f"""
You are a query classifier for a document management system.

Your job is to classify a user's query based on the examples below.

Here are the classification rules and examples:

{examples_text}

When a user asks a query, return ONLY a JSON object with these fields:
{{
  "intent": "{intents_str}",
  "spec_category": "{spec_categories_str}",
  "route": "{routes_str}",
  "es_index": "{es_indexes_str} or null",
  "search_strategy": "{search_strategies_str} or null"
}}

Do not add any extra text. Return ONLY the JSON object. If you are unsure, return an empty json. 
NEVER respond with text outside the JSON object. If a field is not applicable, return null for that field.
"""
    
    return system_prompt

