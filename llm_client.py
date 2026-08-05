# ============================================
# FILE: llm_client.py
# PURPOSE: LLM Client with Context
# ============================================

import json
import requests
from typing import Dict, Optional


class LLMClient:
    """
    Client for calling OpenRouter LLM API.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name (OpenRouter model)
            api_key: OpenRouter API key
            endpoint: OpenRouter API endpoint
        """
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.system_prompt = None
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt (context) once.
        """
        self.system_prompt = system_prompt
    
    def classify_query(self, query: str) -> Dict:
        """
        Classify a query using the stored system prompt.
        """
        if not self.system_prompt:
            raise ValueError("System prompt not set. Call set_system_prompt() first.")
        
        try:
            # ─── Build the request ───
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f'Classify this query: "{query}"'}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 300,
                "top_p": 0.1
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # ─── Call the API ───
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # ─── Check response ───
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return self._get_default()
            
            # ─── Parse response ───
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            return self._parse_response(content)
            
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return self._get_default()
    
    def _parse_response(self, content: str) -> Dict:
        """
        Parse the LLM response into a dictionary.
        """
        try:
            # Clean the response
            content = content.strip()
            
            # Remove markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Parse JSON
            result = json.loads(content)
            
            # Ensure all fields exist
            return {
                "intent": result.get("intent", "search"),
                "spec_category": result.get("spec_category", "File Retrieval"),
                "route": result.get("route", "Elasticsearch search"),
                "es_index": result.get("es_index"),
                "search_strategy": result.get("search_strategy")
            }
        except Exception as e:
            print(f"❌ Error parsing response: {e}")
            return self._get_default()
    
    def _get_default(self) -> Dict:
        """
        Return default classification when everything fails.
        """
        return {
            "intent": "search",
            "spec_category": "File Retrieval",
            "route": "Elasticsearch search",
            "es_index": "document",
            "search_strategy": "BFS"
        }