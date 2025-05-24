# EasyRemote 示例代码说明

## 📁 示例组织结构

我们的示例代码按复杂度分为三个层次：

```
examples/
├── basic/               # 基础示例
│   ├── vps_server.py   # 基本网关服务器
│   ├── compute_node.py # 基本计算节点
│   └── test_client.py  # 基本客户端测试
├── ml_service/         # 机器学习服务示例  
├── concurrent_streaming/ # 并发流处理示例
└── advanced/           # 高级示例
    ├── distributed_ai_agents.py    # 分布式AI代理
    ├── multi_node_load_balancing.py # 多节点负载均衡
    ├── edge_computing_network.py   # 边缘计算网络
    └── streaming_pipeline.py       # 流处理管道
```

## 🌟 基础示例

### 1. 基本网关服务器 (`basic/vps_server.py`)

这是最简单的网关服务器实现，适合部署在VPS上：

```python
from easyremote import Server
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建并启动服务器
if __name__ == "__main__":
    server = Server(
        host="0.0.0.0",    # 监听所有接口
        port=8080          # 标准端口
    )
    
    print("🚀 启动EasyRemote网关服务器...")
    print("📡 监听地址: 0.0.0.0:8080")
    server.start()
```

**使用场景：**
- VPS部署
- Docker容器部署
- 生产环境网关

### 2. 基本计算节点 (`basic/compute_node.py`)

展示如何创建和注册计算节点：

```python
from easyremote import ComputeNode
import time

# 连接到网关服务器
node = ComputeNode("your-server-ip:8080")

@node.register
def simple_calculation(x, y):
    """简单的数学计算"""
    return x + y * 2

@node.register
def data_processing(data_list):
    """处理数据列表"""
    return [item.upper() for item in data_list]

if __name__ == "__main__":
    print("💻 启动计算节点...")
    node.serve()
```

**使用场景：**
- 个人电脑贡献算力
- 服务器资源共享
- 开发测试环境

### 3. 基本客户端测试 (`basic/test_client.py`)

演示如何调用远程函数：

```python
from easyremote import Client

def test_basic_functions():
    # 连接到服务器
    client = Client("your-server-ip:8080")
    
    # 测试简单计算
    result1 = client.execute("simple_calculation", 10, 5)
    print(f"计算结果: {result1}")
    
    # 测试数据处理
    result2 = client.execute("data_processing", ["hello", "world"])
    print(f"处理结果: {result2}")

if __name__ == "__main__":
    test_basic_functions()
```

## 🤖 机器学习服务示例

### AI推理服务 (`ml_service/`)

展示如何部署AI模型服务：

```python
# ml_service/ai_node.py
from easyremote import ComputeNode
import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageClassifier:
    def __init__(self):
        # 加载预训练模型
        self.model = torch.load("your_model.pth")
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    
    def predict(self, image_data):
        # 图像预处理
        image = Image.open(image_data)
        tensor = self.transform(image).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(tensor)
            prediction = torch.argmax(output, dim=1)
        
        return prediction.item()

# 创建分类器实例
classifier = ImageClassifier()
node = ComputeNode("gateway:8080")

@node.register
def classify_image(image_bytes):
    """图像分类服务"""
    return classifier.predict(image_bytes)

@node.register
def batch_classify(image_list):
    """批量图像分类"""
    results = []
    for image_data in image_list:
        result = classifier.predict(image_data)
        results.append(result)
    return results

if __name__ == "__main__":
    print("🤖 启动AI推理节点...")
    node.serve()
```

**使用场景：**
- AI模型部署
- 图像/文本处理服务
- 私有AI推理

## ⚡ 高级示例

### 1. 分布式AI代理 (`advanced/distributed_ai_agents.py`)

展示如何构建协作的AI代理网络：

```python
from easyremote import ComputeNode, Client
import asyncio
import random

class AIAgent:
    def __init__(self, agent_id, server_address):
        self.agent_id = agent_id
        self.node = ComputeNode(server_address)
        self.client = Client(server_address)
        
    def register_capabilities(self):
        @self.node.register
        def process_task(task_data):
            """处理分配的任务"""
            # 模拟AI处理逻辑
            result = f"Agent-{self.agent_id} processed: {task_data}"
            return result
        
        @self.node.register
        def collaborate(task, other_agents):
            """与其他代理协作"""
            results = []
            for agent in other_agents:
                # 委托部分任务给其他代理
                sub_result = self.client.execute(
                    f"agent_{agent}_process", 
                    task
                )
                results.append(sub_result)
            return results

# 创建多个AI代理
agents = []
for i in range(3):
    agent = AIAgent(f"agent_{i}", "gateway:8080")
    agent.register_capabilities()
    agents.append(agent)

# 启动所有代理
async def start_agents():
    tasks = []
    for agent in agents:
        task = asyncio.create_task(agent.node.serve_async())
        tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(start_agents())
```

**使用场景：**
- 分布式AI系统
- 多代理协作
- 任务分解和协调

### 2. 多节点负载均衡 (`advanced/multi_node_load_balancing.py`)

演示如何实现智能负载均衡：

```python
from easyremote import ComputeNode
import psutil
import time

class ResourceAwareNode:
    def __init__(self, server_address, node_type="general"):
        self.node = ComputeNode(server_address)
        self.node_type = node_type
        
    def get_system_info(self):
        """获取系统资源信息"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "node_type": self.node_type
        }
    
    def register_functions(self):
        @self.node.register
        def cpu_intensive_task(data):
            """CPU密集型任务"""
            # 模拟CPU密集型计算
            result = sum(i**2 for i in range(len(data)))
            return result
        
        @self.node.register
        def memory_intensive_task(large_data):
            """内存密集型任务"""
            # 模拟内存密集型处理
            processed = [x * 2 for x in large_data]
            return len(processed)
        
        @self.node.register
        def get_node_status():
            """报告节点状态"""
            return self.get_system_info()

# 创建不同类型的节点
cpu_node = ResourceAwareNode("gateway:8080", "cpu_optimized")
memory_node = ResourceAwareNode("gateway:8080", "memory_optimized")
general_node = ResourceAwareNode("gateway:8080", "general")

# 注册功能
for node in [cpu_node, memory_node, general_node]:
    node.register_functions()

# 并发启动节点
import threading

def start_node(node):
    node.node.serve()

if __name__ == "__main__":
    threads = []
    for node in [cpu_node, memory_node, general_node]:
        thread = threading.Thread(target=start_node, args=(node,))
        thread.start()
        threads.append(thread)
    
    # 等待所有节点启动
    for thread in threads:
        thread.join()
```

**使用场景：**
- 异构硬件管理
- 智能任务调度
- 资源优化利用

### 3. 边缘计算网络 (`advanced/edge_computing_network.py`)

展示边缘计算场景的实现：

```python
from easyremote import ComputeNode, Client
import json
import time
from datetime import datetime

class EdgeDevice:
    def __init__(self, device_id, location, server_address):
        self.device_id = device_id
        self.location = location
        self.node = ComputeNode(server_address)
        self.client = Client(server_address)
        
    def register_edge_functions(self):
        @self.node.register
        def process_sensor_data(sensor_readings):
            """处理传感器数据"""
            timestamp = datetime.now().isoformat()
            processed_data = {
                "device_id": self.device_id,
                "location": self.location,
                "timestamp": timestamp,
                "readings": sensor_readings,
                "avg_value": sum(sensor_readings) / len(sensor_readings)
            }
            return processed_data
        
        @self.node.register
        def local_analytics(data_batch):
            """本地数据分析"""
            analytics = {
                "total_samples": len(data_batch),
                "max_value": max(data_batch),
                "min_value": min(data_batch),
                "trend": "increasing" if data_batch[-1] > data_batch[0] else "decreasing"
            }
            return analytics
        
        @self.node.register
        def edge_coordination(task):
            """边缘设备协调"""
            # 寻找附近的边缘设备
            nearby_devices = self.find_nearby_devices()
            
            # 分发任务给附近设备
            results = []
            for device in nearby_devices:
                try:
                    result = self.client.execute(f"process_task_{device}", task)
                    results.append(result)
                except Exception as e:
                    print(f"设备 {device} 不可用: {e}")
            
            return results
    
    def find_nearby_devices(self):
        """查找附近的边缘设备"""
        # 简化的设备发现逻辑
        all_devices = ["edge_001", "edge_002", "edge_003"]
        return [d for d in all_devices if d != self.device_id]

# 创建边缘设备网络
edge_devices = [
    EdgeDevice("edge_001", "Beijing", "gateway:8080"),
    EdgeDevice("edge_002", "Shanghai", "gateway:8080"),
    EdgeDevice("edge_003", "Shenzhen", "gateway:8080")
]

# 注册所有边缘设备
for device in edge_devices:
    device.register_edge_functions()

# 启动边缘计算网络
async def start_edge_network():
    tasks = []
    for device in edge_devices:
        task = asyncio.create_task(device.node.serve_async())
        tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import asyncio
    print("🌐 启动边缘计算网络...")
    asyncio.run(start_edge_network())
```

**使用场景：**
- IoT数据处理
- 边缘AI推理
- 分布式传感器网络

## 🔄 并发流处理示例

### 实时数据流处理 (`concurrent_streaming/`)

展示如何处理实时数据流：

```python
# concurrent_streaming/stream_processor.py
from easyremote import ComputeNode
import asyncio
import queue
import threading

class StreamProcessor:
    def __init__(self, server_address):
        self.node = ComputeNode(server_address)
        self.data_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
    def register_stream_functions(self):
        @self.node.register
        def process_stream_chunk(data_chunk):
            """处理数据流片段"""
            processed = []
            for item in data_chunk:
                # 实时数据处理逻辑
                result = item * 2 + 1
                processed.append(result)
            return processed
        
        @self.node.register
        def aggregate_results(result_chunks):
            """聚合处理结果"""
            all_results = []
            for chunk in result_chunks:
                all_results.extend(chunk)
            
            return {
                "total_items": len(all_results),
                "sum": sum(all_results),
                "average": sum(all_results) / len(all_results)
            }
    
    async def stream_worker(self):
        """流处理工作者"""
        while True:
            try:
                # 获取数据块
                if not self.data_queue.empty():
                    data_chunk = self.data_queue.get()
                    
                    # 处理数据
                    result = await self.process_chunk_async(data_chunk)
                    self.results_queue.put(result)
                
                await asyncio.sleep(0.1)  # 避免过度占用CPU
                
            except Exception as e:
                print(f"流处理错误: {e}")
    
    async def process_chunk_async(self, chunk):
        """异步处理数据块"""
        # 这里可以调用其他节点或进行复杂处理
        return [x * 2 for x in chunk]

# 使用示例
processor = StreamProcessor("gateway:8080")
processor.register_stream_functions()

async def main():
    # 启动流处理器
    processor_task = asyncio.create_task(processor.node.serve_async())
    worker_task = asyncio.create_task(processor.stream_worker())
    
    # 模拟数据流输入
    for i in range(10):
        data_chunk = list(range(i*10, (i+1)*10))
        processor.data_queue.put(data_chunk)
        await asyncio.sleep(1)
    
    await asyncio.gather(processor_task, worker_task)

if __name__ == "__main__":
    asyncio.run(main())
```

**使用场景：**
- 实时数据分析
- 流媒体处理
- 在线监控系统

## 🚀 运行示例

### 1. 基础示例运行步骤

```bash
# 终端1: 启动网关服务器
cd examples/basic
python vps_server.py

# 终端2: 启动计算节点
python compute_node.py

# 终端3: 运行客户端测试
python test_client.py
```

### 2. 高级示例运行步骤

```bash
# 启动分布式AI代理
cd examples/advanced
python distributed_ai_agents.py

# 启动多节点负载均衡
python multi_node_load_balancing.py

# 启动边缘计算网络
python edge_computing_network.py
```

## 💡 示例修改指南

### 自定义网关地址

在所有示例中，将 `"gateway:8080"` 替换为您的实际网关地址：

```python
# 替换前
node = ComputeNode("gateway:8080")

# 替换后
node = ComputeNode("your-actual-server:8080")
```

### 添加自定义函数

```python
@node.register
def your_custom_function(param1, param2):
    """您的自定义函数"""
    # 实现您的逻辑
    result = process_your_data(param1, param2)
    return result
```

### 错误处理

```python
@node.register
def robust_function(data):
    """带错误处理的函数"""
    try:
        result = risky_operation(data)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## 🔗 相关资源

- 📖 [快速开始指南](quick-start.md)
- 📚 [API参考文档](api-reference.md)
- 🎓 [基础教程](../tutorials/basic-usage.md)
- 🚀 [高级场景教程](../tutorials/advanced-scenarios.md)
- 🏗️ [架构文档](../architecture/overview.md)

## 💬 获取帮助

如果您在运行示例时遇到问题：

- 🐛 [报告问题](https://github.com/Qingbolan/EasyCompute/issues)
- 💬 [社区讨论](https://github.com/Qingbolan/EasyCompute/discussions)
- 📧 [邮件支持](mailto:silan.hu@u.nus.edu) 