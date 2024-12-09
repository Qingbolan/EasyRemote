# compute_node.py
from easyremote import ComputeNode
from PIL import Image
import io

# 初始化 ComputeNode
node = ComputeNode(
    vps_address="127.0.0.1:8080",
    node_id="basic-compute"
)

@node.register
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    # print("执行 /add")
    return a + b

@node.register
def process_data(data: dict) -> dict:
    """处理数据，将每个值乘以2。"""
    # print("执行 /process")
    return {k: v * 2 for k, v in data.items()}

@node.register
def process_photo(photo_bytes: bytes) -> bytes:
    """
    处理照片，将其转换为灰度图像。

    Args:
        photo_bytes (bytes): 原始照片的字节数据。

    Returns:
        bytes: 处理后的灰度照片的字节数据。
    """
    # print("执行 /process_photo")
    try:
        # 从字节数据中打开图像
        image = Image.open(io.BytesIO(photo_bytes))
        # 转换为灰度图像
        grayscale = image.convert("L")
        # 将处理后的图像保存到字节流
        output = io.BytesIO()
        grayscale.save(output, format='PNG')  # 保持原格式或选择其他格式
        return output.getvalue()
    except Exception as e:
        # print(f"处理照片时出错: {e}")
        raise e

if __name__ == "__main__":
    node.serve()
