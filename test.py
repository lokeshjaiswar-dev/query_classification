# ============================================
# FILE: test_api.py
# PURPOSE: Test if API key and model are working
# ============================================

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-nano-9b-v2:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def test_api():
    """
    Simple test to check if API key and model are working.
    """
    print("=" * 60)
    print("API CONNECTION TEST")
    print("=" * 60)
    
    # ─── Check 1: Is API key present? ───
    print("\n1️⃣ Checking API key...")
    if not API_KEY:
        print("   ❌ ERROR: OPENROUTER_API_KEY not found in .env file")
        print("   Please create a .env file with: OPENROUTER_API_KEY=sk-or-v1-...")
        return False
    print(f"   ✅ API key found: {API_KEY[:15]}...{API_KEY[-10:]}")
    
    # ─── Check 2: Can we make a simple API call? ───
    print("\n2️⃣ Making test API call...")
    
    # Simple test message
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Say 'Hello, API is working!' in exactly 5 words."}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        print(f"   📤 Sending request to: {ENDPOINT}")
        print(f"   📤 Using model: {MODEL}")
        
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # ─── Check 3: Check response status ───
        print(f"\n3️⃣ Checking response...")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Connection SUCCESSFUL!")
            
            # Parse and display response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            print("\n4️⃣ Response from API:")
            print(f"   📥 {content}")
            print("\n   ✅ API is working perfectly!")
            return True
            
        elif response.status_code == 401:
            print("   ❌ Authentication failed! (401 Unauthorized)")
            print("   Please check your API key.")
            print(f"   Response: {response.text}")
            return False
            
        elif response.status_code == 429:
            print("   ❌ Rate limit exceeded! (429 Too Many Requests)")
            print("   Please wait and try again later.")
            return False
            
        else:
            print(f"   ❌ API error! Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Connection timed out!")
        print("   Please check your internet connection.")
        return False
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection error!")
        print("   Please check your internet connection.")
        return False
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_complete_flow():
    """
    Test the complete flow with classification.
    """
    print("\n" + "=" * 60)
    print("COMPLETE FLOW TEST")
    print("=" * 60)
    
    # Import the necessary modules
    try:
        from classification_builder import build_grouped_examples, build_system_prompt_with_examples
        from llm_client import LLMClient
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Make sure you're running from the correct directory.")
        return False
    
    print("\n1️⃣ Building examples from CSV...")
    try:
        grouped_examples = build_grouped_examples("query_classifications.csv", max_examples_per_category=3)
        print("   ✅ Examples built successfully")
    except Exception as e:
        print(f"   ❌ Error building examples: {e}")
        return False
    
    print("\n2️⃣ Building system prompt...")
    try:
        system_prompt = build_system_prompt_with_examples(grouped_examples)
        print("   ✅ System prompt built successfully")
    except Exception as e:
        print(f"   ❌ Error building system prompt: {e}")
        return False
    
    print("\n3️⃣ Creating LLM client...")
    try:
        client = LLMClient(model=MODEL, api_key=API_KEY, endpoint=ENDPOINT)
        client.set_system_prompt(system_prompt)
        print("   ✅ LLM client created successfully")
    except Exception as e:
        print(f"   ❌ Error creating client: {e}")
        return False
    
    print("\n4️⃣ Testing classification...")
    test_query = "get me all the files and take the salary of the last doucment"
    print(f"   Query: '{test_query}'")
    
    try:
        result = client.classify_query(test_query)
        print("   ✅ Classification successful!")
        print(f"\n   Result:")
        print(f"   - Intent: {result.get('intent')}")
        print(f"   - Spec Category: {result.get('spec_category')}")
        print(f"   - Route: {result.get('route')}")
        print(f"   - ES Index: {result.get('es_index')}")
        print(f"   - Search Strategy: {result.get('search_strategy')}")
        return True
    except Exception as e:
        print(f"   ❌ Classification error: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 STARTING API TESTS...\n")
    
    # Test 1: Basic API connection
    api_working = test_api()
    
    if api_working:
        print("\n" + "=" * 60)
        print("✅ API BASIC TEST PASSED!")
        print("=" * 60)
        
        # Test 2: Complete flow
        flow_working = test_complete_flow()
        
        if flow_working:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! System is ready to use.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️ API works but complete flow has issues.")
            print("Please check the error messages above.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ API TEST FAILED! Please check:")
        print("1. Your .env file has OPENROUTER_API_KEY")
        print("2. Your internet connection")
        print("3. The API key is valid")
        print("=" * 60)