import json

person = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "addr":{
        "street": "123 Main St",
        "zip": "10001"
    }
}
print("Running")
# print(json.dumps(person, indent=4))
# print(json.loads(person))

print(person.keys())
print(person.values())
print(person.items())

for (key, value) in person.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for (sub_key, sub_value) in value.items():
            print(f"  {sub_key}: {sub_value}")
    print(f"{key}: {value}")
