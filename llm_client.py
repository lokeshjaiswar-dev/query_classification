import json
import requests
from typing import Dict, List


class LLMClient:
    """
    Simple LLM client for query classification.
    """
    
    def __init__(self, model: str, api_key: str, endpoint: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.system_prompt = None
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt once."""
        self.system_prompt = prompt
    
    def classify(self, query: str) -> List[Dict]:
        """
        Classify a query (handles both single and composite).
        Returns a list of classifications.
        """
        if not self.system_prompt:
            raise ValueError("System prompt not set!")
        
        try:

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f'Classify this query: "{query}"'}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.1
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return self._default_classification(query)
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # print(f"\n🔹 LLM Response:\n{content}\n")
            
            return self._parse_response(content, query)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._default_classification(query)
    
    def _parse_response(self, content: str, original_query: str) -> List[Dict]:
        """Parse LLM response into list of classifications."""
        try:

            content = content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content)
            
            if isinstance(result, list):
                queries = result
            elif isinstance(result, dict) and "queries" in result:
                queries = result["queries"]
            else:
                # Single classification
                queries = [result]
            
            # Ensure each has required fields
            for q in queries:
                if "text" not in q:
                    q["text"] = original_query
                q.setdefault("intent", "search")
                q.setdefault("spec_category", "File Retrieval")
                q.setdefault("route", "Elasticsearch search")
                q.setdefault("es_index", None)
                q.setdefault("search_strategy", None)
            
            return queries
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            print(f"Content: {content}")
            return self._default_classification(original_query)
    
    def _default_classification(self, query: str) -> List[Dict]:
        """Default classification when everything fails."""
        return [{
            "text": query,
            "intent": "search",
            "spec_category": "File Retrieval",
            "route": "Elasticsearch search",
            "es_index": None,
            "search_strategy": "BFS"
        }]