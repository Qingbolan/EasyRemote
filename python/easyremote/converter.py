from enum import Enum
from typing import Any, Dict, Tuple, Union
from PIL import Image
import numpy as np
import io
import json
from .logging_config import logger
import base64
import gzip

class DataType(str, Enum):
    NULL = "null"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    BYTES = "bytes"
    LIST = "list"
    DICT = "dict"
    SET = "set"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    NDARRAY = "ndarray"

class DataConverter:
    def __init__(self):
        self._type_handlers = {
            type(None): self._handle_none,
            int: self._handle_primitive,
            float: self._handle_primitive,
            bool: self._handle_primitive,
            str: self._handle_primitive,
            bytes: self._handle_bytes,
            bytearray: self._handle_bytes,
            list: self._handle_list,
            tuple: self._handle_list,
            dict: self._handle_dict,
            set: self._handle_list,
            np.ndarray: self._handle_ndarray,
            Image.Image: self._handle_image,
        }

    
    def encode(self, obj: Any) -> Tuple[bytes, Dict]:
        """将Python对象编码为二进制数据和元数据"""
        # 遍历所有已注册的类型处理器
        for type_class, handler in self._type_handlers.items():
            # 使用isinstance来检查类型继承关系
            if isinstance(obj, type_class):
                data, meta = handler(obj)
                # 确保type字段是字符串
                if isinstance(meta.get('type'), DataType):
                    meta['type'] = meta['type'].value
                return data, meta
                
        raise TypeError(f"Unsupported type: {type(obj)}")

    def decode(self, data: str, metadata: Dict) -> Any:
        """从字符串数据和元数据还原Python对象"""
        logger.debug("Starting decode method")
        type_name = metadata["type"]
        logger.debug(f"Type name: {type_name}")

        if type_name == DataType.NULL:
            return None

        if type_name in (DataType.INT, DataType.FLOAT, DataType.BOOL, DataType.STR):
            logger.debug("Decoding basic type")
            # 处理收到的 Base64 编码的数据
            compressed_data = base64.b64decode(data)
            logger.debug("Base64 decoding successful")
            # 解压缩数据
            decompressed_data = gzip.decompress(compressed_data)
            logger.debug("Gzip decompression successful")
            # 将字节数据转换为字符串
            string_value = decompressed_data.decode('utf-8')
            logger.debug(f"Decoded string value: {string_value}")

            if type_name == DataType.STR:
                return string_value
            elif type_name == DataType.INT:
                return int(string_value)
            elif type_name == DataType.FLOAT:
                return float(string_value)
            elif type_name == DataType.BOOL:
                return string_value.lower() == 'true'
            else:
                raise ValueError(f"Unsupported basic type: {type_name}")
            
        if type_name == DataType.BYTES:
            return data
            
        if type_name == DataType.LIST:
            obj = json.loads(data.decode())
            return tuple(obj) if metadata.get("tuple") else obj
            
        if type_name == DataType.SET:
            return set(json.loads(data.decode()))
            
        if type_name == DataType.DICT:
            decoded_dict = json.loads(data.decode())
            return self._restore_dict_values(decoded_dict)
            
        if type_name == DataType.IMAGE:
            return self._decode_image(data, metadata)
            
        if type_name in (DataType.NDARRAY, DataType.VIDEO, DataType.AUDIO):
            return self._decode_ndarray(data, metadata)
            
        raise ValueError(f"Unknown type: {type_name}")

    def _handle_none(self, _: None) -> Tuple[bytes, Dict]:
        return b"", {"type": DataType.NULL}

    def _handle_primitive(self, obj: Union[int, float, bool, str]) -> Tuple[bytes, Dict]:
        return str(obj).encode(), {
            "type": type(obj).__name__
        }

    def _handle_bytes(self, obj: Union[bytes, bytearray]) -> Tuple[bytes, Dict]:
        return bytes(obj), {
            "type": DataType.BYTES
        }

    def _handle_list(self, obj: Union[list, tuple, set]) -> Tuple[bytes, Dict]:
        data = json.dumps(list(obj)).encode()
        metadata = {
            "type": DataType.LIST if isinstance(obj, (list, tuple)) else DataType.SET
        }
        if isinstance(obj, tuple):
            metadata["tuple"] = True
        return data, metadata

    def _process_dict_value(self, value: Any) -> Any:
        """处理字典中的值"""
        if isinstance(value, bytes):
            return {
                "_type": "bytes",
                "data": value.hex()
            }
        elif isinstance(value, dict):
            return {key: self._process_dict_value(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [self._process_dict_value(item) if isinstance(item, dict) 
                   else item for item in value]
        else:
            return value

    def _restore_dict_values(self, obj: Any) -> Any:
        """还原字典中的特殊类型值"""
        if isinstance(obj, dict):
            # 检查是否是特殊类型的字典
            if "_type" in obj and obj["_type"] == "bytes":
                return bytes.fromhex(obj["data"])
            
            # 递归处理普通字典
            return {key: self._restore_dict_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_dict_values(item) for item in obj]
        else:
            return obj

    def _handle_ndarray(self, obj: np.ndarray) -> Tuple[bytes, Dict]:
        metadata = {
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        }
        
        # 判断数组类型
        if len(obj.shape) == 4:  # [frames, height, width, channels]
            metadata["type"] = DataType.VIDEO
            metadata["frames"] = obj.shape[0]
        elif len(obj.shape) == 2 and obj.shape[1] in (1, 2):  # [samples, channels]
            metadata["type"] = DataType.AUDIO
            metadata["channels"] = obj.shape[1]
        else:
            metadata["type"] = DataType.NDARRAY
            
        return obj.tobytes(), metadata

    def _decode_image(self, data: bytes, metadata: Dict) -> Image.Image:
        """从二进制数据还原图片"""
        img = Image.open(io.BytesIO(data))
        
        # 还原格式信息
        img.format = metadata["format"]
        
        # 还原图片信息
        if "info" in metadata:
            img.info.update(metadata["info"])
            
        return img

    def _decode_ndarray(self, data: bytes, metadata: Dict) -> np.ndarray:
        shape = tuple(metadata["shape"])
        dtype = np.dtype(metadata["dtype"])
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _handle_image(self, obj: Image.Image) -> Tuple[bytes, Dict]:
        """处理图像对象"""
        try:
            logger.debug(f"Handling image of size {obj.size}, mode {obj.mode}")
            buf = io.BytesIO()
            format = obj.format or 'PNG'
            obj.save(buf, format=format)
            
            metadata = {
                "type": "image",  # 直接使用字符串类型
                "format": format,
                "mode": obj.mode,
                "size": [obj.size[0], obj.size[1]]  # 转换为列表
            }
            
            logger.debug(f"Created image metadata: {metadata}")
            return buf.getvalue(), metadata
            
        except Exception as e:
            logger.error(f"Error in _handle_image: {str(e)}", exc_info=True)
            raise    

    def _handle_dict(self, obj: dict) -> Tuple[bytes, Dict]:
        """处理字典类型,支持嵌套的特殊类型"""
        # 创建一个新字典来存储处理后的值
        processed_dict = {}
        
        # 遍历字典中的每个键值对
        for key, value in obj.items():
            # 如果值是bytes类型
            if isinstance(value, bytes):
                processed_dict[key] = {
                    "_type": "bytes",
                    "data": value.hex()  # 将bytes转换为十六进制字符串
                }
            # 如果值是字典类型
            elif isinstance(value, dict):
                processed_dict[key] = self._process_dict_value(value)
            # 如果值是列表类型
            elif isinstance(value, list):
                processed_dict[key] = [self._process_dict_value(item) if isinstance(item, dict) 
                                    else item for item in value]
            # 其他基本类型直接保存
            else:
                processed_dict[key] = value

        # 创建符合Go端期望的metadata结构
        metadata = {
            "type": "dict",
            "extra": {}
        }

        return json.dumps(processed_dict).encode(), metadata