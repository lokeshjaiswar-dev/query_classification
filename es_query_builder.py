"""
Builds Elasticsearch queries from user queries
"""
import re
from typing import Dict, Optional, List


class ESQueryBuilder:
    """Build Elasticsearch queries from natural language queries"""
    
    def __init__(self):
        # Mapping from spec_category to document_type
        self.category_mapping = {
            "Pay Slip": "pay_slip",
            "Employment Agreement / Offer Letter": "offer_letter",
            "Employee Information Form": "employee_form",
            "Performance Appraisal Form": "appraisal",
            "Contracts": "contract",
            "Policy Documents": "policy",
            "Financial Reports": "financial_report",
            "Invoices & POs": "invoice",
            "File Retrieval": None,
        }
    
    def build_search_query(self, classification: Dict, query_text: str) -> Dict:
        """
        Build Elasticsearch query from classification
        """
        must_conditions = []
        filter_conditions = []
        
        # 1. Extract employee name from query
        emp_name = self._extract_employee_name(query_text)
        if emp_name:
            must_conditions.append({
                "match": {
                    "emp_name": {
                        "query": emp_name,
                        "operator": "and"
                    }
                }
            })
        
        # 2. Extract document type from spec_category
        spec_category = classification.get('spec_category', '')
        doc_type = self.category_mapping.get(spec_category)
        if doc_type:
            filter_conditions.append({"term": {"document_type": doc_type}})
        
        # 3. Extract year from query
        year = self._extract_year(query_text)
        if year:
            filter_conditions.append({"term": {"year": year}})
        
        # 4. Extract increment number
        inc_num = self._extract_increment_number(query_text)
        if inc_num:
            filter_conditions.append({"term": {"increment_number": inc_num}})
        
        # 5. Extract specific fields (salary, amount, etc.)
        field_query = self._extract_field_query(query_text)
        if field_query:
            must_conditions.append({
                "match": {
                    "extracted_text": field_query
                }
            })
        
        # 6. Build the query
        es_query = {
            "query": {
                "bool": {
                    "must": must_conditions if must_conditions else [{"match_all": {}}],
                    "filter": filter_conditions
                }
            }
        }
        
        # 7. Add sorting
        if "first" in query_text.lower() or "earliest" in query_text.lower():
            es_query["sort"] = [{"year": {"order": "asc"}}]
            es_query["size"] = 1
        elif "latest" in query_text.lower() or "recent" in query_text.lower():
            es_query["sort"] = [{"year": {"order": "desc"}}]
            es_query["size"] = 1
        else:
            es_query["size"] = 10
        
        return es_query
    
    def _extract_employee_name(self, query: str) -> Optional[str]:
        """Extract employee name from query"""
        patterns = [
            r'(?:for|of|about|get|find|show)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:salary|increment|offer)',
            r'employee\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_year(self, query: str) -> Optional[int]:
        """Extract year from query"""
        match = re.search(r'(20\d{2})', query)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_increment_number(self, query: str) -> Optional[int]:
        """Extract increment number from query"""
        mapping = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
        }
        
        query_lower = query.lower()
        for word, num in mapping.items():
            if word in query_lower:
                return num
        
        match = re.search(r'(\d+)(?:th)?\s+increment', query_lower)
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_field_query(self, query: str) -> Optional[str]:
        """Extract what specific information is being asked"""
        fields = {
            'salary': ['salary', 'pay', 'compensation', 'amount'],
            'date': ['date', 'when', 'timeline'],
            'percentage': ['percentage', '%', 'hike'],
            'designation': ['designation', 'role', 'title']
        }
        
        query_lower = query.lower()
        for field, keywords in fields.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return field
        
        return None