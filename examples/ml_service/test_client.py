# test_client.py
import requests
import sounddevice as sd
import wavio
import cv2
from PIL import Image
import io
import numpy as np
import time

def test_translate():
    response = requests.post(
        "http://127.0.0.1:8000/translate",
        json={"text": "Hello, world!", "target_language": "de"}
    )
    if response.status_code == 200:
        print("Translation result:", response.json())
    else:
        print("Translation failed:", response.json())

def record_audio(duration=5, fs=16000):
    """
    录制音频并返回 WAV 格式的字节数据。
    
    Args:
        duration (int): 录音时长（秒）。
        fs (int): 采样率。
    
    Returns:
        bytes: WAV 文件的字节数据。
    """
    print(f"开始录制音频，时长 {duration} 秒...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # 等待录音完成
    print("录音完成。")
    
    # 将录音数据保存到字节流
    with io.BytesIO() as wav_io:
        wavio.write(wav_io, recording, fs, sampwidth=2)
        wav_bytes = wav_io.getvalue()
    return wav_bytes

def test_speech_to_text():
    """
    录制音频并测试 /speech_to_text 端点。
    """
    audio_bytes = record_audio(duration=5, fs=16000)
    files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
    
    print("上传音频并请求语音转文本...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/speech_to_text",
            files=files
        )
        if response.status_code == 200:
            print("Speech-to-text result:", response.json())
        else:
            print("Speech-to-text failed:", response.json())
    except Exception as e:
        print("请求过程中出错:", e)

def capture_image():
    """
    捕捉一张图像并返回 PNG 格式的字节数据。
    
    Returns:
        bytes: PNG 文件的字节数据。
    """
    print("启动摄像头，准备捕捉图像...")
    cap = cv2.VideoCapture(0)  # 0 是默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头。")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据。")
        cap.release()
        return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 将 OpenCV 的 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # 将图像保存到字节流
    with io.BytesIO() as img_io:
        pil_image.save(img_io, format='PNG')
        img_bytes = img_io.getvalue()
    print("图像捕捉完成。")
    return img_bytes

def test_style_transfer():
    """
    捕捉图像并测试 /style_transfer 端点。
    """
    image_bytes = capture_image()
    if image_bytes is None:
        print("图像捕捉失败，跳过样式迁移测试。")
        return
    
    files = {"file": ("captured_image.png", image_bytes, "image/png")}
    data = {"style": "abstract"}  # 您可以更改为其他风格，如 "sepia"
    
    print("上传图像并请求样式迁移...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/style_transfer",
            files=files,
            data=data
        )
        if response.status_code == 200:
            # 保存处理后的图像
            with open("styled_image.png", "wb") as f:
                f.write(response.content)
            print("样式迁移完成。处理后的图像已保存为 styled_image.png。")
        else:
            print("样式迁移失败:", response.json())
    except Exception as e:
        print("请求过程中出错:", e)

def test_predict():
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=[10, 20, 30, 40, 50]
    )
    if response.status_code == 200:
        print("Prediction result:", response.json())
    else:
        print("Prediction failed:", response.json())

if __name__ == "__main__":
    print("测试文本翻译功能...")
    test_translate()
    time.sleep(1)  # 等待1秒

    print("\n测试语音转文本功能...")
    test_speech_to_text()
    time.sleep(1)  # 等待1秒

    print("\n测试图像风格迁移功能...")
    test_style_transfer()
    time.sleep(1)  # 等待1秒

    print("\n测试机器学习预测功能...")
    test_predict()
