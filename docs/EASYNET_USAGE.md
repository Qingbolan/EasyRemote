# EasyNet - 自动Python脚本加速工具

EasyNet是一个类似于`torchrun`的命令行工具，可以自动加速Python脚本中的计算密集型函数，无需修改现有代码。

## 🚀 快速开始

### 安装

```bash
pip install easyremote
```

### 基本使用

```bash
# 普通运行
python your_script.py

# 使用EasyNet自动加速
easynet your_script.py

# 带性能分析
easynet --profile your_script.py

# 指定网关地址
easynet --gateway remote-server:8080 your_script.py

# 详细输出
easynet --verbose your_script.py arg1 arg2
```

## 📋 命令行选项

| 选项 | 简写 | 描述 |
|------|------|------|
| `--gateway` | `-g` | EasyRemote网关地址 (默认: easynet.run:8617) |
| `--auto-discover` | `-a` | 自动发现可用网关 |
| `--profile` | `-p` | 启用性能分析和报告 |
| `--verbose` | `-v` | 启用详细输出 |
| `--version` | | 显示版本信息 |

## 🎯 使用方式

### 1. 装饰器方式（推荐）

在你的Python代码中使用EasyNet装饰器：

```python
from easyremote.cli.accelerator import remote, accelerate, smart_accelerate

# 远程执行
@remote()
def compute_heavy_task(data):
    # 计算密集型任务，强制远程执行
    return process_data(data)

# 智能加速
@accelerate()
def ml_training(dataset, epochs=100):
    # 机器学习训练，智能决策本地或远程
    return train_model(dataset, epochs)

# 智能加速（自动决策）
@smart_accelerate()
def adaptive_function(data):
    # 根据数据大小和复杂度自动选择本地或远程执行
    return process_data(data)
```

### 2. 类级别加速

```python
from easyremote.cli.accelerator import auto_accelerate

@auto_accelerate()
class DataPipeline:
    def preprocess(self, data):
        return clean_data(data)
    
    def train(self, data):
        return train_model(data)
    
    def predict(self, model, data):
        return model.predict(data)
```

### 3. 透明加速（实验性）

```bash
# 直接运行现有脚本，EasyNet会自动识别和加速合适的函数
easynet existing_script.py
```

## 🧠 智能加速策略

EasyNet使用多种启发式方法来决定是否加速函数：

### 自动检测条件

1. **数据大小**：大型数组或列表（>1000元素）
2. **函数复杂度**：包含循环、数学运算的多行函数
3. **计算关键词**：numpy, pandas, torch, sklearn等
4. **历史性能**：基于过往执行时间的学习

### 加速决策流程

```
函数调用 → 分析参数大小 → 检查函数复杂度 → 查看历史性能 → 决定执行位置
    ↓              ↓              ↓              ↓
  小数据         简单函数        无历史数据      本地执行
    ↓              ↓              ↓              ↓
  大数据         复杂函数        远程更快        远程执行
```

## 📊 性能监控

### 启用性能分析

```bash
easynet --profile your_script.py
```

### 性能报告示例

```
================================================================================
🚀 EasyNet Acceleration Report
================================================================================
📊 Summary:
  Total accelerated functions: 5
  Total function calls: 23
  Remote executions: 15
  Local executions: 8
  Acceleration ratio: 65.2%
  Total remote time: 12.450s
  Total local time: 18.230s
  Time saved: 5.780s

📋 Function Details:
  __main__.matrix_multiplication:
    Calls: 3 (Remote: 3, Local: 0)
    Remote %: 100.0%
    Avg times: Remote 2.150s, Local 0.000s
    Speedup: N/A
    Last: remote in 2.145s

  __main__.heavy_data_processing:
    Calls: 5 (Remote: 4, Local: 1)
    Remote %: 80.0%
    Avg times: Remote 1.200s, Local 2.100s
    Speedup: 1.75x
    Last: remote in 1.180s
================================================================================
```

## 🔧 高级配置

### 环境变量

```bash
export EASYNET_GATEWAY=easynet.run:8617
export EASYNET_PROFILE=true
export EASYNET_VERBOSE=true
```

### 配置文件

创建 `.easynet.toml` 文件：

```toml
[easynet]
gateway = "easynet.run:8617"
auto_discover = false
profile = true
verbose = false

[acceleration]
min_data_size = 1000
min_function_lines = 5
force_remote_keywords = ["torch", "tensorflow", "sklearn"]
```

## 🎯 最佳实践

### 1. 函数设计

```python
# ✅ 好的设计 - 适合远程执行
@remote()
def process_large_dataset(data, config):
    """处理大型数据集"""
    results = []
    for item in data:
        processed = complex_computation(item, config)
        results.append(processed)
    return results

# ❌ 不适合远程执行 - 太简单
def simple_add(a, b):
    return a + b
```

### 2. 数据传输优化

```python
# ✅ 传输优化的结果
@remote()
def compute_summary_stats(large_data):
    # 在远程计算汇总统计，只返回小结果
    return {
        'mean': np.mean(large_data),
        'std': np.std(large_data),
        'count': len(large_data)
    }

# ❌ 避免传输大量数据
@remote()
def process_and_return_all(large_data):
    # 避免返回与输入同样大的数据
    return [x * 2 for x in large_data]
```

### 3. 错误处理

```python
@remote(fallback_local=True)
def robust_computation(data):
    """如果远程执行失败，自动回退到本地执行"""
    return expensive_computation(data)
```

## 🔍 故障排除

### 常见问题

1. **连接失败**
   ```bash
   # 检查网关是否运行
   telnet localhost 8080
   
   # 使用详细模式查看错误
   easynet --verbose your_script.py
   ```

2. **函数未加速**
   ```python
   # 添加profile=True查看决策过程
   @remote(profile=True)
   def your_function(data):
       return process(data)
   ```

3. **性能不如预期**
   ```bash
   # 使用性能分析找出瓶颈
   easynet --profile --verbose your_script.py
   ```

### 调试模式

```bash
# 启用详细日志
export EASYNET_LOG_LEVEL=debug
easynet --verbose your_script.py
```

## 🌟 示例项目

查看完整示例：

```bash
# 运行演示脚本
python examples/easynet_demo.py

# 使用EasyNet加速
easynet examples/easynet_demo.py

# 带性能分析
easynet --profile examples/easynet_demo.py
```

## 🚀 与其他工具对比

| 特性 | EasyNet | torchrun | Ray |
|------|---------|----------|-----|
| 自动加速 | ✅ | ❌ | ❌ |
| 零代码修改 | ✅ | ❌ | ❌ |
| 智能决策 | ✅ | ❌ | ❌ |
| 性能分析 | ✅ | ❌ | ✅ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 📚 更多资源

- [EasyRemote文档](../README.md)
- [API参考](./API_REFERENCE.md)
- [示例代码](../examples/)
- [性能优化指南](./PERFORMANCE_GUIDE.md) 