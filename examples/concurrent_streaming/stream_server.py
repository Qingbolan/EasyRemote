#!/usr/bin/env python3
"""
并发流式任务服务器
支持同时运行多个流式任务的VPS服务器
"""
import asyncio
import logging
from easyremote import Server, remote
from easyremote.core.utils.performance import get_performance_monitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingServer:
    """流式任务服务器"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = Server(port=port, max_queue_size=1000)
        self.performance_monitor = get_performance_monitor()
        
    async def start(self):
        """启动服务器和性能监控"""
        logger.info("🚀 Starting EasyRemote Streaming Server...")
        
        # 启动性能监控
        await self.performance_monitor.start()
        logger.info("📊 Performance monitor started")
        
        # 启动服务器
        logger.info(f"🌐 Server starting on port {self.port}")
        logger.info("Waiting for compute nodes to connect...")
        logger.info("Available streaming functions:")
        logger.info("  - data_node: process_sensor_data_stream (real-time data analysis)")
        logger.info("  - ml_node: classify_image_stream (AI inference)")
        
        # 阻塞运行服务器
        self.server.start()
        
    async def stop(self):
        """停止服务器"""
        logger.info("🛑 Stopping server...")
        await self.performance_monitor.stop()
        await self.server.stop()

# 注册远程流式函数（由计算节点实现）
@remote(node_id="data-node")
def process_sensor_data_stream(sensor_config: dict):
    """
    处理传感器数据流
    Args:
        sensor_config: 传感器配置 {'sensors': [...], 'sample_rate': 100, 'duration': 60}
    Returns:
        Generator yielding analysis results
    """
    pass

@remote(node_id="ml-node") 
def classify_image_stream(model_config: dict):
    """
    图像分类流式推理
    Args:
        model_config: 模型配置 {'model_name': 'resnet50', 'batch_size': 8, 'num_images': 100}
    Returns:
        Generator yielding classification results
    """
    pass

@remote(node_id="data-node")
def get_sensor_status() -> dict:
    """获取传感器状态（非流式）"""
    pass

@remote(node_id="ml-node")
def get_model_info() -> dict:
    """获取模型信息（非流式）"""
    pass

def execute_concurrent_streams():
    """
    演示并发执行两个流式任务
    这个函数可以被客户端调用来启动并发流
    """
    try:
        server = Server.current()
        
        # 配置数据处理流
        sensor_config = {
            'sensors': ['temperature', 'humidity', 'pressure'],
            'sample_rate': 10,  # 10 samples/second
            'duration': 30      # 30 seconds
        }
        
        # 配置ML推理流  
        model_config = {
            'model_name': 'mobilenet_v2',
            'batch_size': 4,
            'num_images': 50
        }
        
        logger.info("🔄 Starting concurrent streaming tasks...")
        
        # 启动数据处理流
        data_stream = process_sensor_data_stream(sensor_config)
        logger.info("📊 Data processing stream started")
        
        # 启动ML推理流
        ml_stream = classify_image_stream(model_config)
        logger.info("🤖 ML inference stream started")
        
        return {
            'data_stream': data_stream,
            'ml_stream': ml_stream,
            'status': 'concurrent_streams_started'
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting concurrent streams: {e}")
        return {'error': str(e)}

async def main():
    """主函数"""
    server = StreamingServer(port=8080)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        await server.stop()
        logger.info("✅ Server stopped gracefully")

if __name__ == "__main__":
    asyncio.run(main()) 