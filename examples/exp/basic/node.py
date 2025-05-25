# compute_node.py
from easyremote import ComputeNode

# Connect to the gateway server
node = ComputeNode("localhost:8080")

# Register a simple function
@node.register
def add_numbers(a, b):
    return a + b

# Register an AI inference function
@node.register
def ai_inference(text):
    # Here you can call your local AI model
    return f"AI processing result: {text}"

# Start providing services
node.serve()
