# decorator.py

import functools
import types  # 引入 types 模块
from typing import Callable, TypeVar
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
                # 调用原始函数
                result = func(*args, **kwargs)
                logger.debug(f"Original function result: {result}")

                # 检查结果是否为生成器
                if isinstance(result, types.GeneratorType):
                    logger.debug("Result is a generator")
                    return self._handle_generator(result)
                else:
                    # 序列化结果
                    if result is not None:
                        data, _ = self.converter.encode(result)
                        logger.debug(f"Encoded result")

                        # 通过桥接处理数据
                        processed_data = self.bridge.process_data(data)
                        logger.debug(f"Processed data size: {len(processed_data)}")

                        # 解码并返回结果
                        final_result = self.converter.decode(
                            processed_data,
                            {}
                        )
                        logger.debug(f"Final decoded result: {final_result}")
                        return final_result
                    return result

            except Exception as e:
                logger.error(f"Error in wrapper: {str(e)}", exc_info=True)
                raise

        return wrapper

    def _handle_generator(self, generator):
        logger.debug("Handling generator function")

        def generator_wrapper():
            for item in generator:
                try:
                    logger.debug(f"Processing generator item: {item}")

                    # 序列化生成器产生的项
                    data, _ = self.converter.encode(item)
                    logger.debug(f"Encoded generator item")

                    # 通过桥接处理数据
                    processed_data = self.bridge.process_data(data)
                    logger.debug(f"Processed data size: {len(processed_data)}")

                    # 解码并生成结果
                    final_result = self.converter.decode(
                        processed_data,
                        {}
                    )
                    logger.debug(f"Final decoded generator item: {final_result}")

                    yield final_result

                except Exception as e:
                    logger.error(f"Error processing generator item: {str(e)}", exc_info=True)
                    raise

        return generator_wrapper()