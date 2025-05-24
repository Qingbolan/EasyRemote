#!/usr/bin/env python3
"""
数据处理计算节点
处理实时传感器数据流并返回统计分析结果
"""
import asyncio
import time
import random
import numpy as np
import logging
from typing import Dict, List, Generator
from datetime import datetime
from easyremote import ComputeNode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SensorDataProcessor:
    """传感器数据处理器"""
    
    def __init__(self):
        self.data_buffer = {
            'temperature': [],
            'humidity': [],
            'pressure': []
        }
        self.window_size = 10
        
    def generate_sensor_data(self, sensor_type: str) -> float:
        """生成模拟传感器数据"""
        base_values = {
            'temperature': 25.0,  # °C
            'humidity': 60.0,     # %
            'pressure': 1013.25   # hPa
        }
        
        # 添加一些随机变化和趋势
        base = base_values.get(sensor_type, 0.0)
        noise = random.gauss(0, 0.5)
        trend = 0.1 * np.sin(time.time() / 100)  # 缓慢的周期性变化
        
        return round(base + noise + trend, 2)
    
    def update_buffer(self, sensor_type: str, value: float):
        """更新数据缓冲区"""
        if sensor_type in self.data_buffer:
            self.data_buffer[sensor_type].append(value)
            if len(self.data_buffer[sensor_type]) > self.window_size:
                self.data_buffer[sensor_type].pop(0)
    
    def calculate_statistics(self, sensor_type: str) -> Dict:
        """计算统计信息"""
        data = self.data_buffer.get(sensor_type, [])
        if not data:
            return {}
            
        return {
            'mean': round(np.mean(data), 2),
            'std': round(np.std(data), 2),
            'min': round(np.min(data), 2),
            'max': round(np.max(data), 2),
            'trend': self._calculate_trend(data),
            'samples': len(data)
        }
    
    def _calculate_trend(self, data: List[float]) -> str:
        """计算趋势方向"""
        if len(data) < 3:
            return 'insufficient_data'
            
        recent = data[-3:]
        if recent[-1] > recent[0]:
            return 'increasing'
        elif recent[-1] < recent[0]:
            return 'decreasing'
        else:
            return 'stable'

# 创建计算节点
node = ComputeNode("localhost:8080", node_id="data-node")
processor = SensorDataProcessor()

@node.register(stream=True, name="process_sensor_data_stream")
def process_sensor_data_stream(sensor_config: dict) -> Generator[Dict, None, None]:
    """
    处理传感器数据流
    
    Args:
        sensor_config: 传感器配置
            - sensors: 传感器类型列表
            - sample_rate: 采样率 (samples/second)
            - duration: 持续时间 (seconds)
    
    Yields:
        Dict: 实时分析结果
    """
    sensors = sensor_config.get('sensors', ['temperature'])
    sample_rate = sensor_config.get('sample_rate', 1)
    duration = sensor_config.get('duration', 30)
    
    interval = 1.0 / sample_rate
    start_time = time.time()
    sample_count = 0
    
    logger.info(f"📊 Starting sensor data stream: {sensors}")
    logger.info(f"📈 Sample rate: {sample_rate} Hz, Duration: {duration}s")
    
    try:
        while time.time() - start_time < duration:
            current_time = datetime.now()
            readings = {}
            statistics = {}
            
            # 生成并处理每个传感器的数据
            for sensor_type in sensors:
                # 生成传感器读数
                value = processor.generate_sensor_data(sensor_type)
                readings[sensor_type] = value
                
                # 更新缓冲区
                processor.update_buffer(sensor_type, value)
                
                # 计算统计信息
                statistics[sensor_type] = processor.calculate_statistics(sensor_type)
            
            sample_count += 1
            
            # 构造返回结果
            result = {
                'timestamp': current_time.isoformat(),
                'sample_id': sample_count,
                'elapsed_time': round(time.time() - start_time, 2),
                'readings': readings,
                'statistics': statistics,
                'health_status': 'healthy',
                'node_id': 'data-node'
            }
            
            # 检测异常值
            anomalies = []
            for sensor_type, stats in statistics.items():
                if stats and stats.get('std', 0) > 2.0:  # 标准差过大
                    anomalies.append(f"{sensor_type}_high_variance")
            
            if anomalies:
                result['anomalies'] = anomalies
                result['health_status'] = 'warning'
            
            yield result
            
            # 等待下一个采样间隔
            time.sleep(interval)
            
    except Exception as e:
        logger.error(f"❌ Error in sensor data stream: {e}")
        yield {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'health_status': 'error',
            'node_id': 'data-node'
        }
    
    logger.info(f"✅ Sensor data stream completed: {sample_count} samples")

@node.register(name="get_sensor_status")
def get_sensor_status() -> Dict:
    """获取传感器状态"""
    return {
        'node_id': 'data-node',
        'status': 'online',
        'available_sensors': ['temperature', 'humidity', 'pressure'],
        'buffer_size': processor.window_size,
        'current_data': {
            sensor: len(data) for sensor, data in processor.data_buffer.items()
        },
        'capabilities': {
            'max_sample_rate': 100,  # Hz
            'max_duration': 3600,    # seconds
            'real_time_analysis': True,
            'anomaly_detection': True
        }
    }

@node.register(name="reset_data_buffer")
def reset_data_buffer() -> Dict:
    """重置数据缓冲区"""
    processor.data_buffer = {
        'temperature': [],
        'humidity': [],
        'pressure': []
    }
    return {
        'node_id': 'data-node',
        'status': 'buffer_reset',
        'timestamp': datetime.now().isoformat()
    }

async def main():
    """主函数"""
    logger.info("🚀 Starting Data Processing Node...")
    logger.info("📊 Registered functions:")
    logger.info("  - process_sensor_data_stream (streaming)")
    logger.info("  - get_sensor_status")
    logger.info("  - reset_data_buffer")
    
    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        node.stop()
        logger.info("✅ Data node stopped gracefully")

if __name__ == "__main__":
    asyncio.run(main()) 