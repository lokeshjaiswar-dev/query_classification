# ============================================
# FILE: composite_analyzer.py (SIMPLIFIED & ROBUST)
# PURPOSE: Pure LLM-based composite query detection
# ============================================

import json
import re
from typing import List, Dict


class CompositeQueryAnalyzer:
    """
    Simple LLM-based composite query analyzer.
    """
    
    def __init__(self, llm_client):
        """
        Initialize with the LLM client.
        """
        self.llm_client = llm_client
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze a query and determine if it's composite.
        """
        # Build the analysis prompt - SIMPLER VERSION
        analysis_prompt = f"""
Analyze this query and determine if it contains MULTIPLE separate actions.

Query: "{query}"

Rules:
- COMPOSITE = multiple independent actions (e.g., "find X and find Y", "delete X then move Y")
- SINGLE = one action with multiple objects (e.g., "compare X and Y", "find X and Y")

Return ONLY JSON with this exact format:
{{"is_composite": true/false, "sub_queries": ["query1", "query2"], "reasoning": "explanation"}}

For SINGLE queries, sub_queries should be ["{query}"].
"""
        
        # Save original system prompt
        original_prompt = self.llm_client.system_prompt
        
        try:
            # Temporarily set system prompt for analysis
            self.llm_client.system_prompt = "You are a query analyzer. Return ONLY valid JSON."
            
            # Make the API call
            messages = [
                {"role": "user", "content": analysis_prompt}
            ]
            
            payload = {
                "model": self.llm_client.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 300,
                "top_p": 0.1
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_client.api_key}"
            }
            
            import requests
            response = requests.post(
                self.llm_client.endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"⚠️ Analysis API error: {response.status_code}")
                return self._default_analysis(query)
            
            # Parse response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            print(f"📝 Analysis raw response: {content[:200]}...")  # Debug
            
            # Extract JSON from the response
            result = self._extract_json(content)
            
            if result and "is_composite" in result and "sub_queries" in result:
                return result
            else:
                print("⚠️ Invalid analysis response, using default")
                return self._default_analysis(query)
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return self._default_analysis(query)
        finally:
            # Restore original system prompt
            self.llm_client.system_prompt = original_prompt
    
    def _extract_json(self, content: str) -> Dict:
        """
        Extract JSON from LLM response (handles various formats).
        """
        try:
            # Try to find JSON in the content using regex
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # If no JSON found, try parsing the whole content
            return json.loads(content)
            
        except json.JSONDecodeError:
            # Try to clean and parse again
            try:
                # Remove markdown code blocks
                content = content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content)
            except:
                return None
    
    def _default_analysis(self, query: str) -> Dict:
        """
        Return default analysis when LLM fails.
        """
        return {
            "is_composite": False,
            "sub_queries": [query],
            "reasoning": "Default: treating as single query"
        }


class CompositeQueryProcessor:
    """
    Processes composite queries and returns multiple classifications.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.analyzer = CompositeQueryAnalyzer(llm_client)
    
    def process(self, query: str) -> List[Dict]:
        """
        Process a query and return one or more classifications.
        """
        # Analyze the query
        analysis = self.analyzer.analyze(query)
        
        print(f"\n🔍 Analysis: {analysis.get('reasoning', 'N/A')}")
        print(f"📊 Composite: {analysis.get('is_composite', False)}")
        
        # Get sub-queries
        sub_queries = analysis.get("sub_queries", [query])
        print(f"📝 Sub-queries: {sub_queries}")
        
        # Classify each sub-query
        results = []
        for sub_query in sub_queries:
            print(f"\n📌 Classifying: '{sub_query}'")
            result = self.llm_client.classify_query(sub_query)
            results.append(result)
        
        return results