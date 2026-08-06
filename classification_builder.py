# ============================================
# FILE: classification_builder.py (FIXED)
# PURPOSE: Fix variable substitution in IMPORTANT RULES
# ============================================

import pandas as pd
from collections import defaultdict
from typing import Dict


def build_grouped_examples(
    csv_path: str,
    max_examples_per_category: int = 2
) -> Dict[str, Dict[str, list]]:
    """
    Build grouped examples with structure:
    {
        "intent_name": {
            "spec_category_name": [
                {
                    "route": "route name",
                    "es_index": "index name or None",
                    "search_strategy": "strategy or None"
                }
            ]
        }
    }
    """
    # ─── Load CSV ───
    df = pd.read_csv(csv_path)
    
    # ─── Create nested structure ───
    grouped = defaultdict(lambda: defaultdict(list))
    
    # ─── Populate with examples ───
    for _, row in df.iterrows():
        intent = row["Intent (intent_analysis)"]
        spec_category = row["Spec Category"]
        
        example = {
            "route": row["Route / Handler"],
            "es_index": row.get("ES Index (if search)", None),
            "search_strategy": row.get("Search Strategy", None)
        }
        
        # Clean up empty values
        if pd.isna(example["es_index"]) or example["es_index"] == "-" or example["es_index"] == "":
            example["es_index"] = None
        if pd.isna(example["search_strategy"]) or example["search_strategy"] == "-" or example["search_strategy"] == "":
            example["search_strategy"] = None
        
        # Add to group (limit per category)
        if len(grouped[intent][spec_category]) < max_examples_per_category:
            grouped[intent][spec_category].append(example)
    
    # Convert to regular dict
    return {k: dict(v) for k, v in grouped.items()}


def build_classification_prompt(grouped_examples: Dict) -> str:
    """
    Build the classification prompt with dynamic intents from CSV.
    """
    # ─── Get all unique intents dynamically from CSV ───
    all_intents = list(grouped_examples.keys())
    intents_str = ", ".join(all_intents)
    
    # ─── Build the examples section ───
    examples_section = ""
    for intent, spec_categories in grouped_examples.items():
        examples_section += f"{intent}: {{\n"
        
        for spec_category, examples in spec_categories.items():
            examples_section += f"    {spec_category}: {{\n"
            
            for example in examples:
                examples_section += f"        route: \"{example['route']}\"\n"
                if example['es_index']:
                    examples_section += f"        es_index: \"{example['es_index']}\"\n"
                else:
                    examples_section += f"        es_index: null\n"
                if example['search_strategy']:
                    examples_section += f"        search_strategy: \"{example['search_strategy']}\"\n"
                else:
                    examples_section += f"        search_strategy: null\n"
            
            examples_section += f"    }}\n"
        
        examples_section += f"}}\n\n"
    
    # ─── Build the complete prompt ───
    prompt = f"""You are a query classifier for a document management system.

## CLASSIFICATION RULES:
1. DETECT COMPOSITE QUERIES: A query is COMPOSITE if it contains MULTIPLE independent actions
2. For COMPOSITE queries, SPLIT into separate sub-queries
3. Classify EACH sub-query with the appropriate intent

## TRAINING EXAMPLES:

{examples_section}

## RESPONSE FORMAT:

For SINGLE query:
{{
    "queries": [
        {{
            "text": "original query",
            "intent": "intent_name",
            "spec_category": "spec_category_name",
            "route": "route_name",
            "es_index": "index_name or null",
            "search_strategy": "strategy_name or null"
        }}
    ]
}}

For COMPOSITE query:
{{
    "queries": [
        {{
            "text": "sub-query 1",
            "intent": "intent_name",
            "spec_category": "spec_category_name",
            "route": "route_name",
            "es_index": "index_name or null",
            "search_strategy": "strategy_name or null"
        }}
    ]
}}

## IMPORTANT RULES:
- Return ONLY valid JSON
- No explanations, no markdown
- The "intent" field MUST be one of: {intents_str}
- The "spec_category" field MUST be from the examples above
- "intent" and "spec_category" are DIFFERENT fields - DO NOT use spec_category values as intent
- Set es_index and search_strategy to null if not applicable
- For composite queries, split naturally based on the query structure
- Each sub-query in a composite query gets its own classification
"""
    # print(f"\n✅ Classification prompt built {prompt}")
    return prompt