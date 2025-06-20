from easyremote.cli.accelerator import remote

@remote()
def add_numbers(a, b):
    pass

@remote()
def ai_inference(text):
    pass
    
result1 = add_numbers(10, 20)
result2 = ai_inference("Hello World")
print(f"Calculation result: {result1}")  # Output: 30
print(f"AI result: {result2}")  # Output: AI processing result: Hello World