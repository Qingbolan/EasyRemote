# examples/basic/compute_node.py
from easyremote import ComputeNode
import asyncio
import time

node = ComputeNode(
    vps_address="127.0.0.1:8080",
    node_id="basic-compute"
)

@node.register
def add(a: int, b: int) -> int:
    # print("/add")
    return a + b

@node.register
def process_data(data: dict) -> dict:
    # print("/process")
    return {k: v * 2 for k, v in data.items()}

import json

# 注册一个同步生成器函数（流式）
@node.register(stream=True, async_func=True)
async def stream_process(data: list):
    """
    异步生成器函数，逐步处理数据列表中的每个项，并返回处理后的数据块。
    使用asyncio.sleep()代替time.sleep()以避免阻塞。
    """
    # print(f"Processing data in stream asynchronously: {data}")
    # for item in data:
    #     processed = {"processed": item * 2}
    #     yield json.dumps(processed)
    #     await asyncio.sleep(0.5)  # 使用异步睡眠
    for item in data:
        try:
            processed = {"processed": item * 2}
            yield json.dumps(processed) + "\n"
            await asyncio.sleep(0.5)
        except Exception as e:
            error_msg = {"error": str(e)}
            yield json.dumps(error_msg) + "\n"

if __name__ == "__main__":
    node.serve()
