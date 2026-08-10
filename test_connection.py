from elasticsearch import Elasticsearch

# Try different connection methods
print("Testing Elasticsearch connection...")

# Method 1: Using URL
try:
    es = Elasticsearch("http://localhost:9200")
    if es.ping():
        print("✅ Method 1: Connected successfully!")
        print(f"   Info: {es.info()['version']['number']}")
    else:
        print("❌ Method 1: Failed to ping")
except Exception as e:
    print(f"❌ Method 1 Error: {e}")

# Method 2: Using list of hosts
try:
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    if es.ping():
        print("✅ Method 2: Connected successfully!")
    else:
        print("❌ Method 2: Failed to ping")
except Exception as e:
    print(f"❌ Method 2 Error: {e}")
 
# Method 3: Using different host
try:
    es = Elasticsearch("http://127.0.0.1:9200")
    if es.ping():
        print("✅ Method 3: Connected successfully!")
    else:
        print("❌ Method 3: Failed to ping")
except Exception as e:
    print(f"❌ Method 3 Error: {e}")