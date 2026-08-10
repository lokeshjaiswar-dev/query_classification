"""
Elasticsearch Client for searching
"""
from typing import Dict, List, Optional
from elasticsearch import Elasticsearch
from config import ES_HOST, ES_PORT, ES_INDEX_MASTER, ES_INDEX_DOCUMENTS


class ESClient:
    """Elasticsearch client for searching"""
    
    def __init__(self, host=ES_HOST, port=ES_PORT):
        self.es = Elasticsearch(
            [{'host': host, 'port': port, 'scheme': 'http'}],
            request_timeout=60,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.master_index = ES_INDEX_MASTER
        self.documents_index = ES_INDEX_DOCUMENTS
        
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {host}:{port}")
        print(f"[es_client] ✅ Connected to Elasticsearch")
    
    def search_documents(self, es_query: Dict) -> List[Dict]:
        """Execute Elasticsearch query and return results"""
        try:
            results = self.es.search(index=self.documents_index, body=es_query)
            return [hit['_source'] for hit in results['hits']['hits']]
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search_employees(self, emp_name: str = None) -> List[Dict]:
        """Search for employees"""
        query = {
            "query": {
                "match": {"emp_name": emp_name}
            } if emp_name else {"match_all": {}}
        }
        results = self.es.search(index=self.master_index, body=query)
        return [hit['_source'] for hit in results['hits']['hits']]
    
    def get_employee_documents(self, emp_name: str) -> List[Dict]:
        """Get all documents for an employee"""
        query = {
            "query": {
                "term": {"emp_name.keyword": emp_name}
            },
            "size": 100
        }
        results = self.es.search(index=self.documents_index, body=query)
        return [hit['_source'] for hit in results['hits']['hits']]
    
    def get_increment_letters(self, emp_name: str = None, year: int = None) -> List[Dict]:
        """Get increment letters with optional filters"""
        must_conditions = [{"term": {"document_type": "increment_letter"}}]
        
        if emp_name:
            must_conditions.append({"term": {"emp_name.keyword": emp_name}})
        
        if year:
            must_conditions.append({"term": {"year": year}})
        
        query = {
            "query": {
                "bool": {
                    "must": must_conditions
                }
            },
            "sort": [{"year": "asc"}],
            "size": 100
        }
        
        results = self.es.search(index=self.documents_index, body=query)
        return [hit['_source'] for hit in results['hits']['hits']]
    
    def get_first_increment(self, emp_name: str) -> Optional[Dict]:
        """Get first increment letter for an employee"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"emp_name.keyword": emp_name}},
                        {"term": {"document_type": "increment_letter"}}
                    ]
                }
            },
            "sort": [{"year": {"order": "asc"}}],
            "size": 1
        }
        
        results = self.es.search(index=self.documents_index, body=query)
        hits = results['hits']['hits']
        return hits[0]['_source'] if hits else None
    
    def get_latest_increment(self, emp_name: str) -> Optional[Dict]:
        """Get latest increment letter for an employee"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"emp_name.keyword": emp_name}},
                        {"term": {"document_type": "increment_letter"}}
                    ]
                }
            },
            "sort": [{"year": {"order": "desc"}}],
            "size": 1
        }
        
        results = self.es.search(index=self.documents_index, body=query)
        hits = results['hits']['hits']
        return hits[0]['_source'] if hits else None