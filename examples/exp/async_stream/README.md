# EasyRemote Async & Streaming Test Environment

这个测试环境专门用于测试 EasyRemote 的异步和流式功能。

## 📁 文件结构

```
async_stream/
├── server.py          # 网关服务器
├── compute_node.py    # 计算节点（包含各种测试函数）
├── client.py          # 测试客户端
└── README.md          # 说明文档
```

## 🚀 快速开始

### 方法一：一键运行完整测试
```bash
cd examples/exp/async_stream
python run_tests.py
```

### 方法二：分步运行
#### 1. 启动服务器
```bash
python server.py
```

#### 2. 启动计算节点
在新的终端窗口中：
```bash
python compute_node.py
```

#### 3. 运行测试客户端
在第三个终端窗口中：
```bash
python client.py
```

### 方法三：快速验证修复
```bash
# 确保服务器和计算节点已启动，然后运行：
python verify_fix.py
```

### 方法四：详细测试
```bash
# 运行完整的测试套件：
python test_fix.py
```

## 📋 测试功能

### 同步函数 (Synchronous Functions)
- `sync_add(a, b)` - 简单的加法运算
- `sync_process_data(data)` - 数据处理

### 异步函数 (Asynchronous Functions)
- `async_computation(data, delay)` - 异步计算，支持延迟参数
- `async_ai_simulation(text, model_delay)` - 模拟AI处理

### 流式函数 (Streaming Functions)
- `sync_number_stream(start, count, interval)` - 同步数字流
- `async_data_stream(config)` - 异步传感器数据流
- `async_ml_inference_stream(images, batch_size)` - 异步ML推理流
- `async_complex_pipeline(data_source, config)` - 复杂异步管道

## 🔧 配置参数

### 异步数据流配置
```python
config = {
    'sensors': ['temperature', 'humidity', 'pressure'],
    'sample_rate': 2,  # 每秒采样数
    'duration': 10     # 持续时间（秒）
}
```

### ML推理流配置
```python
images = ['image_001.jpg', 'image_002.jpg', ...]
batch_size = 2  # 批处理大小
```

### 复杂管道配置
```python
config = {
    'stages': 3,           # 处理阶段数
    'items_per_stage': 5,  # 每阶段项目数
    'stage_delay': 1.0     # 每阶段延迟（秒）
}
```

## 📊 测试场景

### 1. 基础功能测试
- 同步函数调用
- 异步函数调用
- 性能对比

### 2. 流式数据处理
- 实时传感器数据模拟
- 批量图像处理
- 多阶段数据管道

### 3. 并发处理
- 多个异步任务并发执行
- 流式数据的实时处理
- 资源管理和清理

## 🎯 使用示例

### 手动测试单个函数
```python
from easyremote import Client

client = Client("localhost:8080")

# 测试异步计算
result = client.execute("async_computation", [1, 2, 3, 4, 5], 2.0)
print(result)

# 测试数据流
config = {'sensors': ['temperature'], 'sample_rate': 1, 'duration': 5}
stream_result = client.execute("async_data_stream", config)
print(stream_result)
```

### 自定义测试
你可以修改 `client.py` 中的测试参数来进行自定义测试：

```python
# 修改异步计算参数
result = client.execute("async_computation", [10, 20, 30], 3.0)

# 修改流式配置
config = {
    'sensors': ['temperature', 'humidity', 'pressure'],
    'sample_rate': 5,  # 更高的采样率
    'duration': 15     # 更长的持续时间
}
```

## 🔍 监控和调试

### 日志输出
所有组件都配置了详细的日志输出，可以观察：
- 函数调用时间
- 数据处理进度
- 错误和异常信息

### 性能监控
- 异步函数的执行时间
- 流式数据的吞吐量
- 内存和CPU使用情况

## ⚠️ 注意事项

1. **端口冲突**: 确保端口 8080 没有被其他程序占用
2. **依赖安装**: 确保已安装 EasyRemote 及其依赖
3. **启动顺序**: 先启动服务器，再启动计算节点，最后运行客户端
4. **资源清理**: 测试完成后记得停止所有进程

## 🛠️ 故障排除

### 常见问题
1. **连接失败**: 检查服务器是否正常启动
2. **函数未找到**: 确保计算节点已连接并注册函数
3. **流式数据异常**: 检查网络连接和超时设置
4. **序列化错误**: 确保流式函数使用了正确的装饰器参数

### 流式函数实现方案

#### 🚨 当前实现的问题
经过深入分析，发现当前的"收集式流式处理"存在以下问题：

1. **不是真正的流式处理**：所有数据被收集后一次性返回，失去实时性
2. **内存效率低**：需要存储所有中间结果
3. **无法处理无限流**：不适合持续的数据流场景
4. **用户体验差**：客户端无法获得实时进度更新

#### 📊 实现方案对比

| 方案 | 实时性 | 内存效率 | 无限流支持 | 实现复杂度 |
|------|--------|----------|------------|------------|
| 收集式流式 | ❌ 无 | ❌ 低 | ❌ 不支持 | ✅ 简单 |
| 真正流式 | ✅ 实时 | ✅ 高 | ✅ 支持 | ⚠️ 中等 |

#### 🔧 当前的权宜方案（收集式）
```python
# ✅ 当前可用的实现方式（自动检测函数类型）
@node.register()
def sync_stream_function(count: int) -> str:
    """同步流式函数 - 返回收集的流数据"""
    results = []
    for i in range(count):
        data = {"item": i, "timestamp": datetime.now().isoformat()}
        results.append(json.dumps(data))
        time.sleep(0.1)  # 模拟处理时间
    return "\n".join(results) + "\n"

@node.register()
async def async_stream_function(config: dict) -> str:
    """异步流式函数 - 返回收集的流数据"""
    results = []
    for i in range(config.get('count', 5)):
        data = {"item": i, "timestamp": datetime.now().isoformat()}
        results.append(json.dumps(data))
        await asyncio.sleep(0.1)  # 异步等待
    return "\n".join(results) + "\n"
```

#### 🎯 理想的真正流式实现
```python
# 🚀 真正的流式处理（需要 EasyRemote 引擎改进）
@node.register(stream=True)
def true_sync_stream(count: int) -> Generator[Dict[str, Any], None, None]:
    """真正的同步流式函数"""
    for i in range(count):
        data = {"item": i, "timestamp": datetime.now().isoformat()}
        yield data  # 实时产生数据
        time.sleep(0.1)

@node.register(stream=True)
async def true_async_stream(config: dict) -> AsyncGenerator[Dict[str, Any], None]:
    """真正的异步流式函数"""
    for i in range(config.get('count', 5)):
        data = {"item": i, "timestamp": datetime.now().isoformat()}
        yield data  # 实时产生数据
        await asyncio.sleep(0.1)
```

#### ❌ 避免使用的方式
```python
# ❌ 会导致序列化错误
@node.register()
def generator_function():
    yield data  # 生成器对象无法正确序列化

# ❌ 会导致执行错误
@node.register(function_type=FunctionType.GENERATOR)
def explicit_generator():
    yield data
```

#### 📋 改进建议
查看 `streaming_analysis.md` 获取详细的问题分析和改进方案。

### 流式数据格式
返回的字符串格式为每行一个 JSON 对象：
```
{"item": 1, "timestamp": "2025-05-25T17:40:00"}
{"item": 2, "timestamp": "2025-05-25T17:40:01"}
{"item": 3, "timestamp": "2025-05-25T17:40:02"}
```

### 调试技巧
- 查看日志输出了解详细信息
- 使用较小的测试参数进行初步验证
- 逐个测试功能模块
- 使用 `test_fix.py` 快速验证流式功能

## 📈 扩展功能

你可以基于这个测试环境扩展更多功能：
- 添加新的异步函数
- 实现更复杂的流式处理逻辑
- 集成真实的AI模型
- 添加性能基准测试

## 🎉 开始测试

现在你可以开始测试 EasyRemote 的异步和流式功能了！按照上面的步骤启动各个组件，然后观察测试结果。 