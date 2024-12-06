# decorator.py
import functools
from typing import Any, Callable, TypeVar
from .logging_config import logger
from .converter import DataConverter
from .bridge import Bridge

T = TypeVar('T')

class easyremote:
    """远程执行装饰器"""
    
    def __init__(self):
        logger.debug("Initializing easyremote decorator")
        self.converter = DataConverter()
        self.bridge = Bridge()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        logger.debug(f"Decorating function: {func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling wrapped function: {func.__name__}")
            logger.debug(f"Args: {args}")
            logger.debug(f"Kwargs: {kwargs}")
            
            try:
                # 1. 首先序列化输入参数
                encoded_args = []
                for arg in args:
                    data, metadata = self.converter.encode(arg)
                    encoded_args.append({
                        'data': data,
                        'metadata': metadata
                    })
                    
                # 2. 调用原始函数
                result = func(*args, **kwargs)
                logger.debug(f"Original function result: {result}")
                
                # 3. 序列化结果
                if result is not None:  # 处理可能的None返回值
                    data, metadata = self.converter.encode(result)
                    logger.debug(f"Encoded result metadata: {metadata}")
                    
                    # 4. 通过桥接处理数据
                    processed = self.bridge.process_data(data, metadata)
                    logger.debug(f"Processed result: {processed}")
                    
                    # 5. 解码并返回结果
                    final_result = self.converter.decode(
                        processed['data'],
                        processed['metadata']
                    )
                    logger.debug(f"Final decoded result: {final_result}")
                    return final_result
                return result
                
            except Exception as e:
                logger.error(f"Error in wrapper: {str(e)}", exc_info=True)
                raise
                
        return wrapper

    def _handle_generator(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        logger.debug("Handling generator function")
        gen = func(*args, **kwargs)

        def wrapped_generator():
            try:
                for item in gen:
                    logger.debug(f"Processing generator item: {item}")
                    if item is not None:
                        data, metadata = self.converter.encode(item)
                        processed = self.bridge.process_data(data, metadata)
                        result = self.converter.decode(
                            processed['data'],
                            processed['metadata']
                        )
                        logger.debug(f"Yielding processed item: {result}")
                        yield result
            except Exception as e:
                logger.error(f"Error in generator: {str(e)}", exc_info=True)
                raise
            finally:
                gen.close()
                
        return wrapped_generator()