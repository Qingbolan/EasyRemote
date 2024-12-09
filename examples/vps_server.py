# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from easyremote import Server, remote
from fastapi.responses import Response
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期事件处理程序，用于启动和关闭 easyremote 服务器。"""
    # 启动 easyremote 服务器
    server = Server(port=8080)
    server.start_background()
    # print("EasyRemote 服务器已在后台启动。")
    try:
        yield
    finally:
        # 在应用关闭时，执行必要的清理操作
        server.stop()  # 假设 easyremote 提供 stop 方法
        # print("EasyRemote 服务器已停止。")

app = FastAPI(lifespan=lifespan)

# 注册远程函数，无需实现，计算节点会处理
@remote(node_id="basic-compute")
def add(a: int, b: int) -> int:
    pass

@remote(node_id="basic-compute")
def process_data(data: dict) -> dict:
    pass

@remote(node_id="basic-compute")
def process_photo(photo_bytes: bytes) -> bytes:
    pass

@app.post("/add")
async def add_endpoint(a: int, b: int):
    """处理 /add 请求，返回两个数的和。"""
    # print("收到 /add 请求")
    try:
        result = add(a, b)
        return {"result": result}
    except Exception as e:
        # print(f"/add 处理时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_endpoint(data: dict):
    """处理 /process 请求，返回处理后的数据。"""
    # print("收到 /process 请求")
    try:
        result = process_data(data)
        return result
    except Exception as e:
        # print(f"/process 处理时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_photo")
async def process_photo_endpoint(file: UploadFile = File(...)):
    """处理 /process_photo 请求，上传并处理照片。"""
    # print("收到 /process_photo 请求")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="无效的图片格式。仅支持 JPEG 和 PNG。")
    
    try:
        photo_bytes = await file.read()
        processed_bytes = process_photo(photo_bytes)
        return Response(content=processed_bytes, media_type=file.content_type)
    except Exception as e:
        # print(f"/process_photo 处理时出错: {e}")
        raise HTTPException(status_code=500, detail="照片处理失败。")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        loop="asyncio",  # 明确指定使用 asyncio 事件循环
        timeout_keep_alive=65  # 增加 keep-alive 超时时间
    )
