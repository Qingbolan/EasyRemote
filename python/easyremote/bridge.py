# bridge.py
import os
import platform
import ctypes
from pathlib import Path
import json
import gzip
from typing import Dict, Any
from .logging_config import logger

class Bridge:
    def __init__(self):
        logger.debug("Initializing Bridge")
        try:
            self.lib = self._load_library()
            self._setup_functions()
            logger.debug("Bridge initialization successful")
        except Exception as e:
            logger.error(f"Bridge initialization failed: {str(e)}", exc_info=True)
            raise

    def _load_library(self) -> ctypes.CDLL:
        # 确定库文件名
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        logger.debug(f"System: {system}")
        logger.debug(f"Machine: {machine}")
        
        if system == 'windows':
            lib_name = f'easyremote_{system}_{machine}.dll'
        elif system == 'darwin':
            lib_name = f'easyremote_{system}_{machine}.dylib'
        else:
            lib_name = f'easyremote_{system}_{machine}.so'
        
        logger.debug(f"Library name: {lib_name}")
        
        # 查找库文件
        pkg_dir = Path(__file__).parent
        lib_path = pkg_dir / 'core' / lib_name
        
        logger.debug(f"Full library path: {lib_path.absolute()}")
        logger.debug(f"Library exists: {lib_path.exists()}")
        
        if not lib_path.exists():
            logger.error(f"Library not found: {lib_path}")
            raise RuntimeError(f"Library not found: {lib_path}")
        
        try:
            lib = ctypes.WinDLL(str(lib_path)) if system == 'windows' else ctypes.CDLL(str(lib_path))
            
            # Log available functions
            logger.debug("Available functions in library:")
            for name in dir(lib):
                if not name.startswith('_'):
                    logger.debug(f"- {name}")
                    
            return lib
        except Exception as e:
            logger.error(f"Failed to load library: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load library: {e}")

    def _setup_functions(self):
        logger.debug("Setting up function signatures")
        try:
            # 设置 ProcessData 函数签名
            self.lib.ProcessData.argtypes = [
                ctypes.c_char_p,  # data
                ctypes.c_int,     # length
                ctypes.c_char_p,  # metadata
            ]
            self.lib.ProcessData.restype = ctypes.c_char_p

            # 设置 FreeString 函数签名
            self.lib.FreeString.argtypes = [ctypes.c_char_p]
            self.lib.FreeString.restype = None
            
            logger.debug("Function signatures setup completed")
        except Exception as e:
            logger.error("Failed to setup function signatures", exc_info=True)
            raise

    def _compress_data(self, data: bytes) -> bytes:
        """使用gzip压缩数据"""
        logger.debug(f"Compressing data of size: {len(data)}")
        try:
            compressed = gzip.compress(data)
            logger.debug(f"Compressed size: {len(compressed)}")
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}", exc_info=True)
            raise
    
    def process_data(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Starting process_data")
        logger.debug(f"Original data size: {len(data)}")
        logger.debug(f"Original metadata: {metadata}")
        
        result_ptr = None
        try:
            # 构建符合 Go 端要求的完整数据包
            packet = {
                "data": data,
                "metadata": {
                    "type": metadata["type"] if isinstance(metadata["type"], str) else metadata["type"].value,
                    "format": metadata.get("format", ""),
                    "mode": metadata.get("mode", ""),
                    "size": metadata.get("size", []),
                    "shape": metadata.get("shape", []),
                    "dtype": metadata.get("dtype", ""),
                    "channels": metadata.get("channels", 0),
                    "extra": metadata.get("extra", {})
                }
            }
            
            logger.debug(f"Prepared packet metadata: {packet['metadata']}")
            
            # 压缩数据
            compressed_data = self._compress_data(data)
            logger.debug(f"Compressed data size: {len(compressed_data)}")
            
            # 准备参数
            logger.debug("Preparing parameters")
            data_bytes = ctypes.create_string_buffer(compressed_data)
            data_len = ctypes.c_int(len(compressed_data))
            
            # 准备metadata JSON字符串
            metadata_json = json.dumps(packet["metadata"])
            logger.debug(f"Metadata JSON: {metadata_json}")
            metadata_bytes = ctypes.create_string_buffer(metadata_json.encode('utf-8'))
            
            logger.debug("Calling ProcessData function")
            # 调用Go函数
            result_ptr = self.lib.ProcessData(
                data_bytes,
                data_len,
                metadata_bytes
            )
            
            if not result_ptr:
                logger.error("ProcessData returned NULL")
                raise RuntimeError("ProcessData returned NULL")

            # 解析结果
            logger.debug("Parsing result")
            result_str = ctypes.string_at(result_ptr).decode('utf-8')
            logger.debug(f"Raw result from ProcessData: {result_str}")
            
            try:
                result = json.loads(result_str)
                logger.debug(f"Parsed result: {result}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse result JSON: {str(e)}")
                raise RuntimeError(f"Invalid JSON response: {result_str}")

            # 检查错误
            if "error" in result:
                logger.error(f"ProcessData returned error: {result['error']}")
                raise RuntimeError(result["error"])

            logger.debug("Process completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in process_data: {str(e)}", exc_info=True)
            raise
        finally:
            # 清理内存
            if result_ptr:
                logger.debug("Cleaning up memory")
                try:
                    self.lib.FreeString(result_ptr)
                except Exception as e:
                    logger.error(f"Error in memory cleanup: {str(e)}", exc_info=True)
                    
                    