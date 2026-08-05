import request

data = request.get("https://jsonplaceholder.typicode.com/todos")

print(data)
print(data.json())