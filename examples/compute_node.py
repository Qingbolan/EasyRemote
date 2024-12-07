# compute_node_example.py

import asyncio
import time
from typing import Generator, AsyncGenerator

from easyremote import ComputeNode

# 初始化 ComputeNode，连接到 VPS 服务器
# 在测试中，我们假设服务器运行在 127.0.0.1:8080
node = ComputeNode(vps_address="127.0.0.1:8080")

# 注册一个同步函数
@node.register(stream=False, async_func=False)
def process_data(data: dict) -> dict:
    """
    简单的同步函数，通过将 'value' 字段的值加倍来处理数据。
    """
    print(f"Processing data synchronously: {data}")
    result = {"result": data["value"] * 2}
    return result

# 注册一个异步函数
@node.register(stream=False, async_func=True)
async def async_process_data(data: dict) -> dict:
    """
    异步函数，通过将 'value' 字段的值乘以三来处理数据，模拟异步操作。
    """
    print(f"Processing data asynchronously: {data}")
    await asyncio.sleep(1)  # 模拟异步处理延迟
    result = {"result": data["value"] * 3}
    return result

# 注册一个同步生成器函数（流式）
@node.register(stream=True, async_func=False)
def stream_process(data: list) -> Generator[dict, None, None]:
    """
    同步生成器函数，逐步处理数据列表中的每个项，并返回处理后的数据块。
    """
    print(f"Processing data in stream synchronously: {data}")
    for item in data:
        processed = {"processed": item * 2}
        yield processed
        time.sleep(0.5)  # 模拟处理延迟

# 注册一个异步生成器函数（流式异步）
@node.register(stream=True, async_func=True)
async def async_stream_process(data: list) -> AsyncGenerator[dict, None]:
    """
    异步生成器函数，逐步异步处理数据列表中的每个项，并返回处理后的数据块。
    """
    print(f"Processing data in stream asynchronously: {data}")
    for item in data:
        await asyncio.sleep(0.5)  # 模拟异步处理延迟
        processed = {"processed": item * 3}
        yield processed

if __name__ == "__main__":
    # 启动计算节点。blocking=True 会阻塞主线程，保持节点运行。
    node.serve(blocking=True)
