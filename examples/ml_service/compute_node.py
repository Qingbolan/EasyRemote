# compute_node.py
from easyremote import ComputeNode
from PIL import Image
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import pipeline
import librosa
import whisper
import torch

# 初始化计算节点，连接到新的 EasyRemote 端口 8081
node = ComputeNode(
    vps_address="127.0.0.1:8081",
    node_id="advanced-compute"
)

# 预加载翻译模型和 Whisper 模型
print("预加载翻译模型和 Whisper 模型...")
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
whisper_model = whisper.load_model("base").to("cpu").float()  # 显式设置为 FP32
print("模型预加载完成。")

# 1. 文本翻译
@node.register
def translate_text(text: str) -> str:
    """使用 Hugging Face 预加载的翻译模型进行文本翻译"""
    try:
        translation = translator(text)[0]["translation_text"]
        return translation
    except Exception as e:
        raise ValueError(f"翻译失败: {e}")

# 2. 语音转文本
@node.register
def speech_to_text(audio_bytes: bytes) -> str:
    """使用预加载的 Whisper 模型将语音转换为文本"""
    try:
        # 将字节数据加载为音频数组
        with io.BytesIO(audio_bytes) as audio_io:
            y, sr = librosa.load(audio_io, sr=16000)
        # Whisper 需要时间序列数据
        result = whisper_model.transcribe(y)
        return result["text"]
    except Exception as e:
        raise ValueError(f"语音转文本失败: {e}")

# 3. 图像风格迁移
@node.register
def image_style_transfer(image_bytes: bytes, style: str) -> bytes:
    """应用简单的图像风格迁移"""
    try:
        # 加载图像
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((256, 256))  # 调整图像大小

        # 使用 NumPy 修改颜色（模拟风格迁移）
        img_array = np.array(image)
        if style == "abstract":
            img_array = 255 - img_array  # 反色
        elif style == "sepia":
            # 应用 sepia 滤镜
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            img_array = img_array @ sepia_filter.T
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            raise ValueError("未知的风格！支持的风格有 'abstract' 和 'sepia'。")

        # 保存处理后的图像
        processed_image = Image.fromarray(img_array)
        output = io.BytesIO()
        processed_image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        raise ValueError(f"图像风格迁移失败: {e}")

# 4. 机器学习预测
@node.register
def predict_future(data: list) -> dict:
    """使用简单的线性回归模型预测未来值"""
    try:
        model = LinearRegression()
        X = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data)
        model.fit(X, y)

        # 预测下一个时间步
        next_value = model.predict([[len(data)]])[0]
        return {"next_value": next_value}
    except Exception as e:
        raise ValueError(f"预测失败: {e}")

if __name__ == "__main__":
    node.serve()
