import json
import requests
from typing import Dict, List, Optional
from vector_store import VectorStore


class LLMClient:
    """
    LLM client for query classification with vector search.
    """
    
    def __init__(self, model: str, api_key: str, endpoint: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.system_prompt = None
        self.vector_store = None
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt once."""
        self.system_prompt = prompt
    
    def set_vector_store(self, vector_store: VectorStore):
        """Set the vector store for semantic search."""
        self.vector_store = vector_store
    
    def classify(self, query: str) -> List[Dict]:
        """Classify a query."""
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
                return self._default_classification(query)
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            return self._parse_response(content, query)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._default_classification(query)
    
    def answer_question(self, query: str, k: int = 20) -> Dict:
        """
        Search for relevant chunks and generate an answer using LLM.
        """
        if not self.vector_store:
            return {
                "answer": "Vector store not initialized.",
                "sources": []
            }
        

        search_results = self.vector_store.search(query, k=k)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        # ─── Step 2: Build context ───
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(search_results, 1):
            # Use FULL text, NOT truncated
            context_parts.append(f"[Document {i}] From: {chunk.filename}\n{chunk.text}")
            sources.append({
                "filename": chunk.filename,
                "employee": chunk.employee_name,
                "score": score
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided document excerpts.

Instructions:
1. Answer using ONLY the information from the provided documents.
2. If the answer is not in the documents, say "I couldn't find that information."
3. Be concise, clear, and direct.
4. Always cite which document(s) you got the information from.
5. If there are dates, numbers, or specific details, include them exactly as they appear.
6. Look carefully at ALL the documents provided - the answer might be in any of them."""
        
        # ─── Step 4: Generic user prompt ───
        answer_prompt = f"""
Answer the following question based ONLY on the provided document excerpts.

Question: {query}

Document Excerpts:
{context}

Rules:
- Answer ONLY from the documents provided
- If the answer isn't there, say so
- Cite your sources (document name)
- Be direct and specific

Answer:
"""
        
        # ─── Step 5: Send to LLM ───
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": answer_prompt}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1000,
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
                print(f"❌ Answer API Error: {response.status_code}")
                return {
                    "answer": "Sorry, I couldn't generate an answer at this time.",
                    "sources": sources
                }
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": "Sorry, I encountered an error while generating the answer.",
                "sources": sources
            }
    
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
                queries = [result]
            
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