# 并发流式任务案例

本案例演示如何使用 EasyRemote 同时运行两个流式任务：
1. **实时数据处理流** - 处理连续的传感器数据流并返回分析结果
2. **机器学习推理流** - 对图像序列进行连续的AI推理

## 🎯 案例特色

✅ **真实并发处理** - 两个流式任务同时运行，展示真正的并发能力  
✅ **实时性能监控** - 监控每个流的吞吐量、延迟和处理进度  
✅ **丰富的可视化** - 使用Rich库提供美观的实时控制台界面  
✅ **容错机制** - 单个流出错不影响其他流的正常运行  
✅ **资源隔离** - 不同计算节点处理不同类型的任务  

## 📁 案例结构

```
concurrent_streaming/
├── README.md           # 本文件
├── requirements.txt    # 依赖项列表
├── run_demo.py         # 快速启动脚本 ⭐
├── stream_server.py    # VPS流式服务器
├── data_node.py        # 数据处理计算节点
├── ml_node.py          # 机器学习计算节点
└── client_demo.py      # 客户端演示
```

## 🚀 快速开始

### 方法 1: 快速演示（推荐）
```bash
cd examples/concurrent_streaming
python run_demo.py
```

### 方法 2: 完整部署
如果你想体验完整的分布式流式处理：

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 启动VPS服务器
```bash
python stream_server.py
```

#### 3. 启动数据处理节点（新终端）
```bash
python data_node.py
```

#### 4. 启动机器学习节点（新终端）
```bash
python ml_node.py
```

#### 5. 运行客户端演示（新终端）
```bash
python client_demo.py
```

## 📊 演示效果

运行演示时，你将看到：

```
🎯 EasyRemote Concurrent Streaming Demo
==================================================
🔗 Testing node connections...
✅ Data node connected: data-node
✅ ML node connected: ml-node
📊 Available models: mobilenet_v2, resnet50, efficientnet_b0

⚡ Configuring streaming tasks...
📊 Data stream: ['temperature', 'humidity', 'pressure'] @ 2 Hz
🤖 ML stream: mobilenet_v2 (24 images)

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Stream          ┃ Status      ┃ Progress   ┃ Latest Result    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Data Processing │ 🟢 Active   │ 15 samples │ Temp: 25.2°C    │
│ ML Inference    │ 🟢 Active   │ 3 batches  │ Progress: 50.0%  │
└─────────────────┴─────────────┴────────────┴──────────────────┘
```

## 🔧 流式任务详情

### 数据处理流 (`data_node.py`)
- **功能**: 实时处理多传感器数据流
- **输入**: 传感器配置（类型、采样率、持续时间）
- **输出**: 
  - 实时传感器读数
  - 统计分析（均值、方差、趋势）
  - 异常检测结果
- **特点**: 
  - 低延迟（< 10ms）
  - 高吞吐量（100+ samples/s）
  - 滑动窗口分析
  - 实时异常检测

### 机器学习推理流 (`ml_node.py`)
- **功能**: 批量图像分类推理
- **输入**: 模型配置（模型名称、批次大小、图像数量）
- **输出**: 
  - 分类结果和置信度
  - 批处理性能指标
  - 推理进度追踪
- **特点**: 
  - 支持多种模型（MobileNet, ResNet, EfficientNet）
  - 批处理优化
  - GPU加速支持（如可用）
  - 详细性能分析

## 🎛️ 配置选项

### 数据流配置
```python
data_config = {
    'sensors': ['temperature', 'humidity', 'pressure'],
    'sample_rate': 10,  # 采样率 (Hz)
    'duration': 60      # 持续时间 (秒)
}
```

### ML推理配置
```python
ml_config = {
    'model_name': 'mobilenet_v2',  # 模型选择
    'batch_size': 8,               # 批次大小
    'num_images': 100,             # 图像总数
    'delay': 0.1                   # 批次间延迟
}
```

## 📈 性能监控

演示包含实时性能监控：
- **吞吐量**: 每秒处理的样本/批次数
- **延迟**: 单次处理时间
- **进度**: 任务完成百分比
- **资源使用**: CPU、内存占用（如启用）
- **错误率**: 失败任务统计

## 🛠️ 扩展和定制

### 添加新的传感器类型
在 `data_node.py` 中的 `SensorDataProcessor` 类中添加：
```python
def generate_sensor_data(self, sensor_type: str) -> float:
    base_values = {
        'temperature': 25.0,
        'humidity': 60.0,
        'pressure': 1013.25,
        'your_sensor': 100.0  # 添加新传感器
    }
    # ... 其余代码
```

### 添加新的ML模型
在 `ml_node.py` 中的 `MockImageClassifier` 类中添加：
```python
self.model_info = {
    'mobilenet_v2': {...},
    'your_model': {
        'classes': 1000,
        'input_size': (224, 224),
        'inference_time': 0.1
    }
}
```

## 🐛 故障排除

### 常见问题

1. **ImportError: No module named 'rich'**
   ```bash
   pip install rich numpy
   ```

2. **连接失败**
   - 检查VPS服务器是否启动
   - 确认端口8080未被占用
   - 检查防火墙设置

3. **流式任务无响应**
   - 检查计算节点是否正常连接
   - 查看服务器日志输出
   - 重启相关组件

## 🔗 相关资源

- [EasyRemote 官方文档](https://github.com/Qingbolan/EasyRemote)
- [流式处理最佳实践](../basic/)
- [性能优化指南](../../docs/performance.md) 