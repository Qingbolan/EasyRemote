import cv2
import numpy as np
from easyremote import ComputeNode

node = ComputeNode(
    vps_address="127.0.0.1:8080",
    node_id="image-processor"
)

@node.register
def process_image(image_data: bytes) -> bytes:
    """处理图像：灰度化、高斯模糊等"""
    # 将bytes转换为numpy数组
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 处理图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 转换回bytes
    _, img_encoded = cv2.imencode('.png', blurred)
    return img_encoded.tobytes()

@node.register(stream=True)
def process_video(video_data: bytes):
    """处理视频流"""
    # 将bytes转换为临时文件
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_data)
        
    cap = cv2.VideoCapture(temp_path)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # 转换为bytes
            _, frame_encoded = cv2.imencode('.jpg', blurred)
            yield frame_encoded.tobytes()
            
    finally:
        cap.release()
        import os
        os.remove(temp_path)

if __name__ == "__main__":
    node.serve()
