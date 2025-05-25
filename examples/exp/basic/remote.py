# client.py
from easyremote import Client

# Connect to the gateway server
client = Client("localhost:8080")

# Call remote functions
result1 = client.execute("add_numbers", 10, 20)
print(f"Calculation result: {result1}")  # Output: 30

result2 = client.execute("ai_inference", "Hello World")
print(f"AI result: {result2}")  # Output: AI processing result: Hello World
