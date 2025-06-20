# client.py
from easyremote import Client

# Connect to the gateway server
client = Client() # default gateway is easynet.run:8617

# Call remote functions
result1 = client.execute("add_numbers", 10, 20)
print(f"Calculation result: {result1}")  # Output: 30

# print(client.get_remote_functions())  # List available remote functions

result2 = client.execute("ai_inference", "Hello World")
print(f"AI result: {result2}")  # Output: AI processing result: Hello World
