from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from easyremote import Server, remote
import io

app = FastAPI()
server = Server(port=8080)

@remote
def process_image(image_data: bytes) -> bytes:
    pass

@remote
def process_video(video_data: bytes):
    pass

@app.on_event("startup")
async def startup():
    server.start_background()

@app.on_event("shutdown")
async def shutdown():
    server.stop()

@app.post("/process/image")
async def process_image_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    result = await process_image(image_data)
    return Response(content=result, media_type="image/png")

@app.post("/process/video")
async def process_video_endpoint(file: UploadFile = File(...)):
    video_data = await file.read()
    
    async def stream_generator():
        async for frame in process_video(video_data):
            yield frame
            
    return StreamingResponse(
        stream_generator(),
        media_type="image/jpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# examples/image_processing/client.py
import requests
import cv2
import numpy as np
from pathlib import Path

class ImageClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def process_image(self, image_path: str) -> np.ndarray:
        """处理单张图片"""
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/process/image",
                files=files
            )
            
        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")
            
        # 转换回numpy数组
        nparr = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    def process_video(self, video_path: str, output_path: str):
        """处理视频"""
        with open(video_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/process/video",
                files=files,
                stream=True
            )
            
        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")
            
        # 准备视频写入器
        frame = None
        writer = None
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                # 解码帧
                nparr = np.frombuffer(chunk, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30,  # FPS
                            (width, height)
                        )
                        
                    writer.write(frame)
                    
        if writer:
            writer.release()

# 使用示例
if __name__ == "__main__":
    client = ImageClient()
    
    # 处理图片
    result = client.process_image("input.jpg")
    cv2.imwrite("output.jpg", result)
    
    # 处理视频
    client.process_video("input.mp4", "output.mp4")