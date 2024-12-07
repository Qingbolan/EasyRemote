import os
import platform
import ctypes
from pathlib import Path
import gzip
from typing import Any
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
            return lib
        except Exception as e:
            logger.error(f"Failed to load library: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load library: {e}")
    
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
        
    def _decompress_data(self, data: bytes) -> bytes:
        """使用gzip解压数据"""
        logger.debug(f"Decompressing data of size: {len(data)}")
        try:
            decompressed = gzip.decompress(data)
            logger.debug(f"Decompressed size: {len(decompressed)}")
            return decompressed
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}", exc_info=True)
            raise

    def _setup_functions(self):
        logger.debug("Setting up function signatures")
        try:
            # 设置 ProcessData 函数签名
            self.lib.ProcessData.argtypes = [
                ctypes.POINTER(ctypes.c_char),  # data
                ctypes.c_size_t,                # length
                ctypes.POINTER(ctypes.c_size_t),   # outLength
            ]
            self.lib.ProcessData.restype = ctypes.POINTER(ctypes.c_char)

            # 设置 FreeResult 函数签名
            self.lib.FreeResult.argtypes = [ctypes.POINTER(ctypes.c_char)]
            self.lib.FreeResult.restype = None

            logger.debug("Function signatures setup completed")
        except Exception as e:
            logger.error("Failed to setup function signatures", exc_info=True)
            raise

    def process_data(self, data: bytes) -> bytes:
        logger.debug("Starting process_data")
        logger.debug(f"Original data size: {len(data)}")

        result_ptr = None
        try:
            # 压缩数据
            compressed_data = self._compress_data(data)
            logger.debug(f"Compressed data size: {len(compressed_data)}")

            # 准备参数
            data_bytes = (ctypes.c_char * len(compressed_data)).from_buffer_copy(compressed_data)
            data_len = ctypes.c_size_t(len(compressed_data))
            out_length = ctypes.c_size_t()

            logger.debug("Calling ProcessData function")
            # 调用Go函数
            result_ptr = self.lib.ProcessData(
                data_bytes,
                data_len,
                ctypes.byref(out_length),
            )

            if not result_ptr:
                logger.error("ProcessData returned NULL")
                raise RuntimeError("ProcessData returned NULL")

            total_length = out_length.value
            logger.debug(f"Out length value: {total_length}")

            if total_length <= 0:
                logger.error(f"Invalid total_length received: {total_length}")
                raise ValueError(f"Invalid total_length received: {total_length}")

            result_data = ctypes.string_at(result_ptr, total_length)
            logger.debug(f"Result data length: {len(result_data)}")

            # 解压缩数据
            decompressed_data = self._decompress_data(result_data)
            logger.debug(f"Decompressed result data size: {len(decompressed_data)}")

            return decompressed_data

        except Exception as e:
            logger.error(f"Error in process_data: {str(e)}", exc_info=True)
            raise
        finally:
            # 清理内存
            if result_ptr:
                logger.debug("Cleaning up memory")
                try:
                    self.lib.FreeResult(result_ptr)
                except Exception as e:
                    logger.error(f"Error in memory cleanup: {str(e)}", exc_info=True)