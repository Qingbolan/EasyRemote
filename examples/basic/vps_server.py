from fastapi import FastAPI
from easyremote import Server, remote
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()
server = Server(port=8080)

@remote(node_id="basic-compute")
def add(a: int, b: int) -> int:
    pass

@remote(node_id="basic-compute")
def process_data(data: dict) -> dict:
    pass

@remote(node_id="basic-compute", stream=True, async_func=True)
async def stream_process(data: list):
    """
    异步生成器函数，逐步处理数据列表中的每个项，并返回处理后的数据块。
    使用asyncio.sleep()代替time.sleep()以避免阻塞。
    """
    print(f"Processing data in stream asynchronously: {data}")
    for item in data:
        processed = {"processed": item * 2}
        yield json.dumps(processed)
        await asyncio.sleep(0.5)  # 使用异步睡眠

@app.on_event("startup")
async def startup():
    server.start_background()

@app.post("/add")
async def add_endpoint(a: int, b: int):
    print("get /add")
    result = add(a, b)
    return {"result": result}

@app.post("/process")
async def process_endpoint(data: dict):
    print("get /process")
    result = process_data(data)
    return result

@app.get("/generate")
async def generate_endpoint(start: int, count: int):
    async def event_generator():
        try:
            for data in stream_process(range(start, start + count)):
                # 确保每个事件都符合SSE格式
                yield f"data: {data}\n\n"
        except Exception as e:
            # 错误处理
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"  # 添加CORS支持
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        loop="asyncio",  # 明确指定使用asyncio事件循环
        timeout_keep_alive=65  # 增加keep-alive超时时间
    )