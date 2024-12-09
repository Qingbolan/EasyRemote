from fastapi import FastAPI
from easyremote import Server, remote
from fastapi.responses import StreamingResponse
import json
import asyncio
import threading

app = FastAPI()
server = Server(port=8080)

@remote(node_id="basic-compute")
def add(a: int, b: int) -> int:
    pass

@remote(node_id="basic-compute")
def process_data(data: dict) -> dict:
    pass

@remote(node_id="basic-compute")
async def stream_process(data: list):
    pass

def run_server_in_thread():
    """在单独的线程中运行easyremote服务器"""
    server.start()

@app.on_event("startup")
async def startup():
    """使用线程来启动服务器，避免事件循环冲突"""
    server_thread = threading.Thread(target=run_server_in_thread)
    server_thread.daemon = True  # 设置为守护线程，这样主程序退出时它会自动终止
    server_thread.start()

@app.post("/add")
async def add_endpoint(a: int, b: int):
    try:
        result = add(a, b)  # 不需要await，因为是同步函数
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/process")
async def process_endpoint(data: dict):
    try:
        result = process_data(data)  # 不需要await，因为是同步函数
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/generate")
async def generate_endpoint(start: int, count: int):
    async def event_generator():
        try:
            async for data in stream_process(range(start, start + count)):
                if isinstance(data, str):
                    yield f"data: {data}"
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/event-stream"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        loop="asyncio",
        timeout_keep_alive=65,
        log_level="debug"
    )