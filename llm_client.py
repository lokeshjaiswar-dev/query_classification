import json
import requests
from typing import Dict, List, Optional
from es_client import ESClient
from es_query_builder import ESQueryBuilder

class LLMClient:
    """
    LLM client for query classification with Elasticsearch.
    """
    
    def __init__(self, model: str, api_key: str, endpoint: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.system_prompt = None
        self.es_client = None
        self.query_builder = ESQueryBuilder()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt once."""
        self.system_prompt = prompt
    
    def set_es_client(self, es_client: ESClient):
        """Set the Elasticsearch client."""
        self.es_client = es_client
    
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
            print(f"\n🤖 LLM Response:\n{content}\n")
            
            return self._parse_response(content, query)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._default_classification(query)
    
    def answer_question(self, query: str) -> Dict:
        """
        Simple flow: Classify → Build ES Query → Get Documents → LLM Answer
        """
        # Step 1: Classify
        classifications = self.classify(query)
        
        if not classifications:
            return {
                "answer": "Failed to classify query.",
                "sources": []
            }
        
        classification = classifications[0]
        print(f"\n📊 Classification: {classification.get('intent')} → {classification.get('spec_category')}")
        
        # Step 2: Build ES Query
        if classification.get('intent') == 'search':
            es_query = self.query_builder.build_search_query(classification, query)
        else:
            es_query = {"query": {"match_all": {}}, "size": 0}
        
        print(f"🔍 ES Query: {json.dumps(es_query, indent=2)}")
        
        # Step 3: Get Documents from ES
        docs = []
        if self.es_client:
            docs = self.es_client.search_documents(es_query)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": []
            }
        
        print(f"📄 Found {len(docs)} documents")
        
        # Step 4: Build context for LLM
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs[:5], 1):
            context_parts.append(f"[Document {i}] From: {doc.get('file_name', 'Unknown')}\n{doc.get('extracted_text', '')[:1000]}")
            sources.append({
                "filename": doc.get('file_name', 'Unknown'),
                "employee": doc.get('emp_name', 'N/A'),
                "year": doc.get('year'),
                "document_type": doc.get('document_type')
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Step 5: Generate answer with LLM
        system_prompt = """You are a helpful assistant that answers questions based on provided document excerpts.

Instructions:
1. Answer using ONLY the information from the provided documents.
2. If the answer is not in the documents, say "I couldn't find that information."
3. Be concise, clear, and direct.
4. Always cite which document(s) you got the information from.
5. If there are dates, numbers, or specific details, include them exactly as they appear."""

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
                return {
                    "answer": "Sorry, I couldn't generate an answer at this time.",
                    "sources": sources
                }
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            print(f"\n🤖 LLM Answer:\n{answer}\n")
            
            return {
                "answer": answer,
                "sources": sources,
                "classification": classification,
                "es_query": es_query
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