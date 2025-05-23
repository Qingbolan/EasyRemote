#!/usr/bin/env python3
"""
机器学习计算节点
实现图像分类的流式推理任务
"""
import asyncio
import time
import random
import logging
import base64
import io
from typing import Dict, List, Generator, Any
from datetime import datetime
from easyremote import ComputeNode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockImageClassifier:
    """模拟图像分类器"""
    
    def __init__(self):
        self.model_info = {
            'mobilenet_v2': {
                'classes': 1000,
                'input_size': (224, 224),
                'inference_time': 0.05  # seconds
            },
            'resnet50': {
                'classes': 1000,
                'input_size': (224, 224),
                'inference_time': 0.08
            },
            'efficientnet_b0': {
                'classes': 1000,
                'input_size': (224, 224),
                'inference_time': 0.06
            }
        }
        
        # ImageNet 类别示例（简化版）
        self.imagenet_classes = [
            'dog', 'cat', 'bird', 'car', 'airplane',
            'ship', 'truck', 'horse', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'keyboard', 'mouse',
            'remote', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors'
        ]
        
        self.inference_count = 0
        
    def generate_mock_image(self) -> Dict:
        """生成模拟图像数据"""
        # 模拟生成图像元数据
        width, height = random.choice([(224, 224), (256, 256), (299, 299)])
        
        return {
            'width': width,
            'height': height,
            'channels': 3,
            'format': 'RGB',
            'size_bytes': width * height * 3,
            # 在实际应用中，这里会是真实的图像数据
            'data': f"mock_image_{self.inference_count}.jpg"
        }
    
    def classify_image(self, image_data: Dict, model_name: str) -> Dict:
        """对图像进行分类"""
        if model_name not in self.model_info:
            model_name = 'mobilenet_v2'
            
        model_config = self.model_info[model_name]
        
        # 模拟推理时间
        time.sleep(model_config['inference_time'])
        
        # 生成随机预测结果
        num_predictions = min(5, len(self.imagenet_classes))
        predicted_classes = random.sample(self.imagenet_classes, num_predictions)
        
        # 生成置信度分数（确保总和接近1.0）
        raw_scores = [random.random() for _ in range(num_predictions)]
        total = sum(raw_scores)
        confidences = [score/total for score in raw_scores]
        confidences.sort(reverse=True)
        
        predictions = []
        for i, class_name in enumerate(predicted_classes[:num_predictions]):
            predictions.append({
                'class': class_name,
                'confidence': round(confidences[i], 4),
                'class_id': random.randint(0, 999)
            })
        
        self.inference_count += 1
        
        return {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence'],
            'model_used': model_name,
            'inference_time': model_config['inference_time'],
            'preprocessing_time': random.uniform(0.001, 0.005)
        }

# 创建计算节点
node = ComputeNode("localhost:8080", node_id="ml-node")
classifier = MockImageClassifier()

@node.register(stream=True, name="classify_image_stream")
def classify_image_stream(model_config: dict) -> Generator[Dict, None, None]:
    """
    图像分类流式推理
    
    Args:
        model_config: 模型配置
            - model_name: 模型名称
            - batch_size: 批处理大小
            - num_images: 处理图像数量
            - delay: 图像间延迟 (optional)
    
    Yields:
        Dict: 分类结果
    """
    model_name = model_config.get('model_name', 'mobilenet_v2')
    batch_size = model_config.get('batch_size', 1)
    num_images = model_config.get('num_images', 10)
    delay = model_config.get('delay', 0.1)  # 图像间延迟
    
    logger.info(f"🤖 Starting image classification stream")
    logger.info(f"📝 Model: {model_name}, Batch size: {batch_size}, Images: {num_images}")
    
    start_time = time.time()
    processed_images = 0
    batch_images = []
    
    try:
        for image_idx in range(num_images):
            # 生成模拟图像
            image_data = classifier.generate_mock_image()
            batch_images.append({
                'image_id': f"img_{image_idx:04d}",
                'data': image_data
            })
            
            # 当批次满了或者是最后一批时，处理批次
            if len(batch_images) >= batch_size or image_idx == num_images - 1:
                batch_start_time = time.time()
                batch_results = []
                
                for img_item in batch_images:
                    # 对每张图像进行分类
                    classification_result = classifier.classify_image(
                        img_item['data'], model_name
                    )
                    
                    batch_results.append({
                        'image_id': img_item['image_id'],
                        'image_info': img_item['data'],
                        'classification': classification_result,
                        'timestamp': datetime.now().isoformat()
                    })
                
                batch_time = time.time() - batch_start_time
                processed_images += len(batch_images)
                
                # 构造批次结果
                result = {
                    'batch_id': f"batch_{(image_idx // batch_size) + 1:03d}",
                    'batch_size': len(batch_images),
                    'batch_time': round(batch_time, 4),
                    'throughput': round(len(batch_images) / batch_time, 2),  # images/second
                    'total_processed': processed_images,
                    'progress': round((processed_images / num_images) * 100, 1),
                    'results': batch_results,
                    'model_info': {
                        'name': model_name,
                        'total_inferences': classifier.inference_count
                    },
                    'node_id': 'ml-node',
                    'status': 'processing'
                }
                
                # 计算性能指标
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    result['overall_throughput'] = round(processed_images / elapsed_time, 2)
                    result['estimated_completion'] = round(
                        (num_images - processed_images) / (processed_images / elapsed_time), 1
                    ) if processed_images > 0 else 0
                
                yield result
                
                # 清空批次缓冲区
                batch_images = []
                
                # 添加延迟（如果需要）
                if delay > 0:
                    time.sleep(delay)
        
        # 发送完成信号
        final_result = {
            'batch_id': 'final',
            'status': 'completed',
            'total_processed': processed_images,
            'total_time': round(time.time() - start_time, 2),
            'overall_throughput': round(processed_images / (time.time() - start_time), 2),
            'model_info': {
                'name': model_name,
                'total_inferences': classifier.inference_count
            },
            'node_id': 'ml-node'
        }
        
        yield final_result
        
    except Exception as e:
        logger.error(f"❌ Error in image classification stream: {e}")
        yield {
            'batch_id': 'error',
            'status': 'error',
            'error': str(e),
            'total_processed': processed_images,
            'node_id': 'ml-node',
            'timestamp': datetime.now().isoformat()
        }
    
    logger.info(f"✅ Image classification stream completed: {processed_images} images")

@node.register(name="get_model_info")
def get_model_info() -> Dict:
    """获取模型信息"""
    return {
        'node_id': 'ml-node',
        'status': 'online',
        'available_models': list(classifier.model_info.keys()),
        'model_details': classifier.model_info,
        'supported_formats': ['RGB', 'BGR', 'RGBA'],
        'capabilities': {
            'max_batch_size': 32,
            'gpu_acceleration': False,  # 模拟环境
            'concurrent_streams': 4,
            'supported_image_sizes': ['224x224', '256x256', '299x299']
        },
        'statistics': {
            'total_inferences': classifier.inference_count,
            'uptime': time.time(),  # 简化的运行时间
        }
    }

@node.register(name="benchmark_model")
def benchmark_model(model_name: str, num_samples: int = 10) -> Dict:
    """对模型进行基准测试"""
    if model_name not in classifier.model_info:
        return {'error': f'Model {model_name} not found'}
    
    logger.info(f"🧪 Benchmarking model: {model_name}")
    
    start_time = time.time()
    inference_times = []
    
    for i in range(num_samples):
        image_data = classifier.generate_mock_image()
        inference_start = time.time()
        classifier.classify_image(image_data, model_name)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
    
    total_time = time.time() - start_time
    
    return {
        'model_name': model_name,
        'num_samples': num_samples,
        'total_time': round(total_time, 4),
        'avg_inference_time': round(sum(inference_times) / len(inference_times), 4),
        'min_inference_time': round(min(inference_times), 4),
        'max_inference_time': round(max(inference_times), 4),
        'throughput': round(num_samples / total_time, 2),
        'node_id': 'ml-node'
    }

async def main():
    """主函数"""
    logger.info("🚀 Starting ML Inference Node...")
    logger.info("🤖 Registered functions:")
    logger.info("  - classify_image_stream (streaming)")
    logger.info("  - get_model_info")
    logger.info("  - benchmark_model")
    logger.info(f"📊 Available models: {list(classifier.model_info.keys())}")
    
    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        node.stop()
        logger.info("✅ ML node stopped gracefully")

if __name__ == "__main__":
    asyncio.run(main()) 