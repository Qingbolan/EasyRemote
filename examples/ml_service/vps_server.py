from fastapi import FastAPI
from easyremote import Server, remote
from fastapi.responses import StreamingResponse  # Add this import
from typing import AsyncGenerator

app = FastAPI()
server = Server(port=8080)

# Example implementations for debugging purposes
@remote
def analyze_sentiment(text: str) -> dict:
    # A dummy implementation for testing
    return {"text": text, "sentiment": "neutral"}

@remote
def generate_text(prompt: str, max_length: int = 100) -> AsyncGenerator[str, None]:
    async def generator():
        for i in range(5):  # Dummy logic to simulate event generation
            yield f"{prompt} -- chunk {i}"
    
    return generator()

@app.on_event("startup")
async def startup():
    server.start_background()

@app.post("/sentiment")
async def sentiment_endpoint(text: str):
    result = await analyze_sentiment(text)
    return result

@app.post("/generate")
async def generate_endpoint(prompt: str):
    async def stream_generator():
        async for chunk in generate_text(prompt):
            yield f"data: {chunk}\n\n"
            
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)