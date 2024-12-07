# examples/basic/compute_node.py
from easyremote import ComputeNode
import asyncio

node = ComputeNode(
    vps_address="127.0.0.1:8080",
    node_id="basic-compute"
)

@node.register
def add(a: int, b: int) -> int:
    print("/add")
    return a + b

@node.register
def process_data(data: dict) -> dict:
    print("/process")
    return {k: v * 2 for k, v in data.items()}

@node.register(stream=True, async_func=True)
async def generate_numbers(start: int, count: int):
    for i in range(start, start + count):
        await asyncio.sleep(0.1)  # 异步等待
        yield i

if __name__ == "__main__":
    node.serve()
