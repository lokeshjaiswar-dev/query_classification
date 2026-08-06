import pandas as pd
from collections import defaultdict
from typing import Dict, Any


def build_grouped_examples(
    csv_path: str,
    max_examples_per_category: int = 3
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
        if example["es_index"] == "-" or example["es_index"] == "":
            example["es_index"] = None
        if example["search_strategy"] == "-" or example["search_strategy"] == "":
            example["search_strategy"] = None
        
        # Add to group (limit per category)
        if len(grouped[intent][spec_category]) < max_examples_per_category:
            grouped[intent][spec_category].append(example)
    
    # Convert to regular dict
    return {k: dict(v) for k, v in grouped.items()}


def build_classification_prompt(grouped_examples: Dict) -> str:
    """
    Build the classification prompt with nested structure:
    intent_name: {
        spec_category_name: {
            examples
        }
    }
    """
    # ─── Build the nested prompt structure ───
    prompt = """You are a query classifier for a document management system.

## CLASSIFICATION RULES:
1. DETECT COMPOSITE QUERIES: A query is COMPOSITE if it contains MULTIPLE independent actions separated by:
   - Conjunctions: "and", "also", "then", "after that", "finally", "additionally"
   - Multiple verbs: "find X and get Y", "delete X then summarize Y"
   - Multiple requests: "show me X and also tell me about Y"
   
2. For COMPOSITE queries, SPLIT into separate sub-queries and classify EACH action independently
   - Example: "find my resume and get my contracts" → 2 sub-queries
   - Example: "delete file1, move file2, and rename file3" → 3 sub-queries
   - Example: "get all files and return summary and how are you" → 3 sub-queries

3. For SINGLE queries, return just ONE classification

## TRAINING EXAMPLES (Query patterns and their classifications):

"""
    
    # ─── Add examples in nested structure ───
    for intent, spec_categories in grouped_examples.items():
        prompt += f"{intent}: {{\n"
        
        for spec_category, examples in spec_categories.items():
            prompt += f"    {spec_category}: {{\n"
            
            for example in examples:
                prompt += f"        route: \"{example['route']}\"\n"
                if example['es_index']:
                    prompt += f"        es_index: \"{example['es_index']}\"\n"
                else:
                    prompt += f"        es_index: null\n"
                if example['search_strategy']:
                    prompt += f"        search_strategy: \"{example['search_strategy']}\"\n"
                else:
                    prompt += f"        search_strategy: null\n"
                prompt += f"        spec_category: \"{spec_category}\"\n"
            
            prompt += f"    }}\n"
        
        prompt += f"}}\n\n"
    
    # ─── Add response format instructions ───
    prompt += """## HOW TO RESPOND:

For a SINGLE query, return ONE classification:
{
    "queries": [
        {
            "text": "original query",
            "intent": "intent_name",
            "spec_category": "spec_category_name",
            "route": "route_name",
            "es_index": "index_name or null",
            "search_strategy": "strategy_name or null"
        }
    ]
}

For a COMPOSITE query, return MULTIPLE classifications:
{
    "queries": [
        {
            "text": "sub-query 1",
            "intent": "intent_name",
            "spec_category": "spec_category_name",
            "route": "route_name",
            "es_index": "index_name or null",
            "search_strategy": "strategy_name or null"
        },
        {
            "text": "sub-query 2",
            "intent": "intent_name",
            "spec_category": "spec_category_name",
            "route": "route_name",
            "es_index": "index_name or null",
            "search_strategy": "strategy_name or null"
        }
    ]
}

## IMPORTANT RULES:
- Return ONLY valid JSON
- No explanations
- No markdown
- Use the exact intent and spec_category names from the examples above
- Set es_index and search_strategy to null if not applicable
- Use your judgment to match the query to the most appropriate intent and spec_category
- For composite queries, split naturally based on the query structure
"""
    
    return prompt