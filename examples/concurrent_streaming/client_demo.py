#!/usr/bin/env python3
"""
并发流式任务客户端演示
同时运行数据处理流和机器学习推理流
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any
import logging
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Rich控制台
console = Console()

class ConcurrentStreamDemo:
    """并发流式任务演示"""
    
    def __init__(self, vps_address: str = "localhost:8080"):
        self.vps_address = vps_address
        self.data_stream_results = []
        self.ml_stream_results = []
        self.data_stream_active = False
        self.ml_stream_active = False
        self.start_time = None
        
    def test_node_connections(self) -> bool:
        """测试节点连接"""
        console.print("🔗 Testing node connections...", style="blue")
        
        # 演示模式：模拟连接测试
        console.print("⚠️  Running in demo mode (nodes not available)", style="yellow")
        return self._demo_mode_connections()
    
    def _demo_mode_connections(self) -> bool:
        """演示模式的连接测试"""
        console.print("✅ Data node connected (demo): data-node", style="green")
        console.print("✅ ML node connected (demo): ml-node", style="green")
        console.print("📊 Available models: mobilenet_v2, resnet50, efficientnet_b0", style="cyan")
        return True
    
    async def run_data_stream(self, config: Dict[str, Any]):
        """运行数据处理流"""
        console.print("📊 Starting data processing stream...", style="blue")
        self.data_stream_active = True
        
        try:
            # 演示模式：直接生成模拟数据
            await self._demo_data_stream(config)
            
        except Exception as e:
            console.print(f"❌ Data stream error: {e}", style="red")
        finally:
            self.data_stream_active = False
            console.print("✅ Data stream completed", style="green")
    
    async def run_ml_stream(self, config: Dict[str, Any]):
        """运行ML推理流"""
        console.print("🤖 Starting ML inference stream...", style="blue")
        self.ml_stream_active = True
        
        try:
            # 演示模式：直接生成模拟数据
            await self._demo_ml_stream(config)
            
        except Exception as e:
            console.print(f"❌ ML stream error: {e}", style="red")
        finally:
            self.ml_stream_active = False
            console.print("✅ ML stream completed", style="green")
    
    async def _demo_data_stream(self, config: Dict[str, Any]):
        """演示模式的数据流"""
        import random
        
        duration = config.get('duration', 20)
        sample_rate = config.get('sample_rate', 2)
        sensors = config.get('sensors', ['temperature', 'humidity'])
        
        samples = int(duration * sample_rate)
        
        for i in range(samples):
            # 生成模拟传感器数据
            readings = {}
            statistics = {}
            
            for sensor in sensors:
                base_values = {'temperature': 25.0, 'humidity': 60.0, 'pressure': 1013.25}
                value = base_values.get(sensor, 0) + random.gauss(0, 1)
                readings[sensor] = round(value, 2)
                
                statistics[sensor] = {
                    'mean': round(value + random.gauss(0, 0.1), 2),
                    'std': round(abs(random.gauss(0.5, 0.2)), 2),
                    'trend': random.choice(['increasing', 'decreasing', 'stable'])
                }
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'sample_id': i + 1,
                'elapsed_time': round(i / sample_rate, 2),
                'readings': readings,
                'statistics': statistics,
                'health_status': 'healthy',
                'node_id': 'data-node'
            }
            
            self.data_stream_results.append(result)
            await asyncio.sleep(1.0 / sample_rate)
    
    async def _demo_ml_stream(self, config: Dict[str, Any]):
        """演示模式的ML推理流"""
        import random
        
        model_name = config.get('model_name', 'mobilenet_v2')
        batch_size = config.get('batch_size', 4)
        num_images = config.get('num_images', 20)
        
        classes = ['dog', 'cat', 'bird', 'car', 'airplane', 'ship', 'truck', 'horse']
        
        num_batches = (num_images + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)
            current_batch_size = end_idx - start_idx
            
            # 模拟批处理时间
            batch_time = current_batch_size * 0.1  # 每张图片0.1秒
            
            batch_results = []
            for img_idx in range(start_idx, end_idx):
                classification = {
                    'predictions': [
                        {
                            'class': random.choice(classes),
                            'confidence': round(random.uniform(0.7, 0.95), 4),
                            'class_id': random.randint(0, 999)
                        }
                    ],
                    'model_used': model_name,
                    'inference_time': 0.05
                }
                
                batch_results.append({
                    'image_id': f"img_{img_idx:04d}",
                    'classification': classification,
                    'timestamp': datetime.now().isoformat()
                })
            
            result = {
                'batch_id': f"batch_{batch_idx + 1:03d}",
                'batch_size': current_batch_size,
                'batch_time': round(batch_time, 4),
                'throughput': round(current_batch_size / batch_time, 2),
                'total_processed': end_idx,
                'progress': round((end_idx / num_images) * 100, 1),
                'results': batch_results,
                'model_info': {'name': model_name},
                'node_id': 'ml-node',
                'status': 'processing' if end_idx < num_images else 'completed'
            }
            
            self.ml_stream_results.append(result)
            await asyncio.sleep(batch_time)
    
    def create_status_table(self) -> Table:
        """创建状态表格"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Stream", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Latest Result", style="white")
        
        # 数据流状态
        data_status = "🟢 Active" if self.data_stream_active else "🔴 Inactive"
        data_progress = f"{len(self.data_stream_results)} samples"
        data_latest = ""
        if self.data_stream_results:
            latest = self.data_stream_results[-1]
            temp = latest.get('readings', {}).get('temperature', 'N/A')
            data_latest = f"Temp: {temp}°C"
        
        table.add_row("Data Processing", data_status, data_progress, data_latest)
        
        # ML流状态
        ml_status = "🟢 Active" if self.ml_stream_active else "🔴 Inactive"
        ml_progress = f"{len(self.ml_stream_results)} batches"
        ml_latest = ""
        if self.ml_stream_results:
            latest = self.ml_stream_results[-1]
            progress = latest.get('progress', 0)
            ml_latest = f"Progress: {progress}%"
        
        table.add_row("ML Inference", ml_status, ml_progress, ml_latest)
        
        return table
    
    def create_summary_panel(self) -> Panel:
        """创建摘要面板"""
        if not self.start_time:
            return Panel("Starting...", title="Summary")
        
        elapsed = time.time() - self.start_time
        data_rate = len(self.data_stream_results) / elapsed if elapsed > 0 else 0
        ml_rate = len(self.ml_stream_results) / elapsed if elapsed > 0 else 0
        
        summary_text = f"""
📊 Data Stream: {len(self.data_stream_results)} samples ({data_rate:.1f}/s)
🤖 ML Stream: {len(self.ml_stream_results)} batches ({ml_rate:.1f}/s)
⏱️  Elapsed Time: {elapsed:.1f}s
🔄 Both streams active: {self.data_stream_active and self.ml_stream_active}
        """.strip()
        
        return Panel(summary_text, title="Real-time Summary", border_style="blue")
    
    async def run_concurrent_demo(self):
        """运行并发演示"""
        console.print("\n🚀 Starting Concurrent Streaming Demo", style="bold blue")
        console.print("=" * 50)
        
        # 测试连接
        if not self.test_node_connections():
            console.print("❌ Connection test failed. Exiting.", style="red")
            return
        
        console.print("\n⚡ Configuring streaming tasks...", style="blue")
        
        # 配置数据处理流
        data_config = {
            'sensors': ['temperature', 'humidity', 'pressure'],
            'sample_rate': 2,  # 2 samples/second
            'duration': 20     # 20 seconds
        }
        
        # 配置ML推理流
        ml_config = {
            'model_name': 'mobilenet_v2',
            'batch_size': 4,
            'num_images': 24,
            'delay': 0.5
        }
        
        console.print(f"📊 Data stream: {data_config['sensors']} @ {data_config['sample_rate']} Hz", style="cyan")
        console.print(f"🤖 ML stream: {ml_config['model_name']} ({ml_config['num_images']} images)", style="cyan")
        
        self.start_time = time.time()
        
        # 创建实时显示
        with Live(self.create_status_table(), refresh_per_second=2) as live:
            # 启动并发任务
            tasks = [
                asyncio.create_task(self.run_data_stream(data_config)),
                asyncio.create_task(self.run_ml_stream(ml_config))
            ]
            
            # 监控任务进度
            while any(not task.done() for task in tasks):
                # 更新显示
                live.update(self.create_status_table())
                await asyncio.sleep(0.5)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
        
        # 显示最终结果
        console.print("\n" + "=" * 50)
        console.print("🎉 Demo Completed!", style="bold green")
        console.print(self.create_summary_panel())
        
        # 显示详细统计
        self._show_detailed_stats()
    
    def _show_detailed_stats(self):
        """显示详细统计信息"""
        console.print("\n📈 Detailed Statistics", style="bold blue")
        
        # 数据流统计
        if self.data_stream_results:
            console.print("\n📊 Data Processing Stream:", style="cyan")
            total_samples = len(self.data_stream_results)
            if total_samples > 0:
                latest_result = self.data_stream_results[-1]
                total_time = latest_result.get('elapsed_time', 0)
                console.print(f"  • Total samples: {total_samples}")
                console.print(f"  • Duration: {total_time:.1f}s")
                console.print(f"  • Average rate: {total_samples/total_time:.2f} samples/s" if total_time > 0 else "  • Rate: N/A")
                
                # 显示最新读数
                if 'readings' in latest_result:
                    console.print("  • Latest readings:")
                    for sensor, value in latest_result['readings'].items():
                        console.print(f"    - {sensor}: {value}")
        
        # ML流统计
        if self.ml_stream_results:
            console.print("\n🤖 ML Inference Stream:", style="cyan")
            total_batches = len(self.ml_stream_results)
            total_images = 0
            total_inference_time = 0
            
            for result in self.ml_stream_results:
                if result.get('status') != 'error':
                    total_images += result.get('batch_size', 0)
                    total_inference_time += result.get('batch_time', 0)
            
            console.print(f"  • Total batches: {total_batches}")
            console.print(f"  • Total images: {total_images}")
            console.print(f"  • Total inference time: {total_inference_time:.2f}s")
            
            if total_inference_time > 0:
                console.print(f"  • Average throughput: {total_images/total_inference_time:.2f} images/s")
            
            # 显示最新分类结果
            if self.ml_stream_results and 'results' in self.ml_stream_results[-1]:
                latest_batch = self.ml_stream_results[-1]['results']
                if latest_batch:
                    latest_classification = latest_batch[-1]['classification']
                    top_pred = latest_classification['predictions'][0]
                    console.print(f"  • Latest classification: {top_pred['class']} ({top_pred['confidence']:.3f})")

async def main():
    """主函数"""
    console.print("🎯 EasyRemote Concurrent Streaming Demo", style="bold magenta")
    console.print("Demonstrating parallel data processing and ML inference streams\n")
    
    demo = ConcurrentStreamDemo()
    
    try:
        await demo.run_concurrent_demo()
    except KeyboardInterrupt:
        console.print("\n⚠️  Demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}", style="red")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 