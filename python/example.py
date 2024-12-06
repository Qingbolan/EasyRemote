from easyremote.converter import DataConverter
from easyremote.bridge import Bridge
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def World(txt):
    # print(f"{txt} World!")
    return f"{txt} World!"

def main():
    # 初始化 DataConverter 和 Bridge
    converter = DataConverter()
    bridge = Bridge()
    
    txt = "Hello"
    
    # 调用函数获取原始结果
    original_result = World(txt)
    logger.debug(f"Original function result: {original_result}")
    
    try:
        # 编码原始结果
        data, metadata = converter.encode(original_result)
        logger.debug(f"Encoded data: {data}")
        logger.debug(f"Metadata: {metadata}")
        
        # 处理数据
        processed_result = bridge.process_data(data, metadata)
        logger.debug(f"Processed result: {processed_result}")
        
        # 解码处理后的结果
        final_result = converter.decode(
            processed_result['data'],
            processed_result['metadata']
        )
        print(final_result)
        logger.info(final_result)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
if __name__ == '__main__':
    main()