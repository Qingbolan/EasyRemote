# examples/basic/vps_server.py
from fastapi import FastAPI
from easyremote import Server, remote
from fastapi.responses import StreamingResponse

app = FastAPI()
server = Server(port=8080)

# 对远程函数使用与node端一致的设定
@remote(node_id="basic-compute")
def add(a: int, b: int) -> int:
    pass

@remote(node_id="basic-compute")
def process_data(data: dict) -> dict:
    pass

# 使用 stream=True, async_func=True 来匹配 compute_node 中的异步流生成器
@remote(node_id="basic-compute", stream=True, async_func=True)
async def generate_numbers(start: int, count: int):
    pass

@app.on_event("startup")
async def startup():
    server.start_background()

@app.post("/add")
async def add_endpoint(a: int, b: int):
    print("get /add")
    # add 是同步函数调用结果，不需要 await
    result = add(a, b)
    return {"result": result}

@app.post("/process")
async def process_endpoint(data: dict):
    print("get /process")
    # process_data 同步函数调用结果，不需要 await
    result = process_data(data)
    return result

@app.get("/generate")
async def generate_endpoint(start: int, count: int):
    # generate_numbers 是异步生成器函数，此处直接将其返回给 StreamingResponse
    # StreamingResponse 可以直接使用异步生成器
    return StreamingResponse(generate_numbers(start, count), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
