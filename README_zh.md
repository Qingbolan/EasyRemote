# EasyRemote

<div align="center">

![EasyRemote Logo](docs/easyremote-logo.png)

[![PyPI version](https://badge.fury.io/py/easyremote.svg)](https://badge.fury.io/py/easyremote)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/easyremote)]()
[![Downloads](https://pepy.tech/badge/easyremote)]()

*一个用于无忧远程计算资源共享的轻量级框架*

[English](README.md) | 中文

</div>

## 为什么选择 EasyRemote？

**您是否厌倦了：**

* **为 AI 开发支付昂贵的云 GPU 费用？**
* **在部署演示时遇到复杂的问题？**
* **寻找团队内部共享计算资源的方法？**

**EasyRemote 只需几行代码就能将本地计算资源（AI 模型、数据处理功能）暴露为远程服务。您只需要一个便宜的 VPS！**

```
 # 就是这么简单：
 from easyremote import remote
 
 @remote
 def run_model(input_data):
     return your_ai_model(input_data)  # 在本地 GPU 上运行
```

## 特性

* **🚀 超级简单**：使用单个装饰器即可将任何函数转换为远程服务
* **💰 经济实惠**：通过便宜的 VPS 使用您的本地 GPU
* **🔒 私密安全**：所有计算都在您的本地机器上进行
* **🌐 灵活部署**：完美适用于演示、原型和团队协作

## 快速开始

### 1. 安装

```
 pip install easyremote
```

### 2. 设置 VPS（网关）

```
 from easyremote import Server
 
 server = Server(port=8080)
 server.start()
```

### 3. 配置本地节点

```
 from easyremote import ComputeNode
 
 # 连接到您的 VPS
 node = ComputeNode("your-vps-ip:8080")
 
 # 定义您的远程函数
 @node.register
 def process_data(data):
     return heavy_computation(data)  # 在本地运行
 
 # 开始服务
 node.serve()
```

### 4. 调用远程函数

```
 # 在任何能访问互联网的地方
 from easyremote import Client
 
 client = Client("vps-ip:8080")
 result = client.call("process_data", data=my_data)
```

## 高级用法

### 异步支持

```
 @node.register(async_func=True)
 async def async_process(data):
     result = await complex_async_operation(data)
     return result
```

### 流式结果

```
 @node.register(stream=True)
 def stream_results(data):
     for chunk in process_large_dataset(data):
         yield chunk
```

## 实际应用示例

**查看我们的示例目录，包含：**

* **AI 模型服务**
* **数据管道处理**
* **团队资源共享**
* **以及更多！**

## 架构

```
 Client -> VPS（网关）-> 本地计算节点
                    -> 本地计算节点
                    -> 本地计算节点
```

## 性能

* **高效的二进制协议**
* **支持大数据传输**
* **自动连接管理**

## 路线图

* **多节点集群支持**
* **增强的安全功能**
* **基于网页的管理界面**
* **更多语言 SDK**
* **Docker 支持**

## 贡献

**我们欢迎贡献！请查看我们的**[贡献指南](CONTRIBUTING.md)

## 许可证

**MIT 许可证**

## 联系与支持

* **作者：胡思蓝**
* **邮箱：**[silan.hu@u.nus.edu](mailto:silan.hu@u.nus.edu)
* **GitHub：**[Qingbolan](https://github.com/Qingbolan)

## 致谢

**特别感谢所有帮助改进 EasyRemote 的贡献者！**

---

*如果您觉得 EasyRemote 有用，请考虑给我一个星标 ⭐*
