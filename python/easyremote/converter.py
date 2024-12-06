# python/easyremote/converter.py
from typing import Any, Dict, Tuple
from PIL import Image
import numpy as np
import io
import pickle
from .logging_config import logger

class DataConverter:
    def encode(self, obj: Any) -> Tuple[bytes, Dict]:
        """将Python对象编码为二进制数据和元数据"""
        try:
            if isinstance(obj, bytes):
                return obj, {}
            elif isinstance(obj, Image.Image):
                buf = io.BytesIO()
                obj.save(buf, format='PNG')
                return buf.getvalue(), {}
            elif isinstance(obj, np.ndarray):
                return obj.tobytes(), {}
            else:
                # 对于其他对象，使用pickle序列化
                return pickle.dumps(obj), {}
        except Exception as e:
            logger.error(f"Encoding error: {e}", exc_info=True)
            raise

    def decode(self, data: bytes, metadata: Dict) -> Any:
        """从二进制数据还原Python对象"""
        try:
            # 尝试使用pickle反序列化
            try:
                return pickle.loads(data)
            except Exception:
                # 如果失败，尝试作为图像加载
                try:
                    img = Image.open(io.BytesIO(data))
                    return img
                except Exception:
                    # 最后尝试作为numpy数组
                    return np.frombuffer(data, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Decoding error: {e}", exc_info=True)
            raise