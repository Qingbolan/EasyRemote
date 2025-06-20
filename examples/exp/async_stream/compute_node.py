# async_stream_compute_node.py
"""
Async and Streaming Compute Node
异步和流式计算节点 - 提供各种测试函数
"""
import asyncio
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any
from easyremote import ComputeNode
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create compute node
node = ComputeNode("localhost:8080", node_id="async-stream-node")

# ========== Synchronous Functions ==========

@node.register
def sync_add(a: int, b: int) -> int:
    """Synchronous addition function"""
    logger.info(f"Sync add: {a} + {b}")
    return a + b

@node.register
def sync_process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous data processing"""
    logger.info(f"Sync processing data: {data}")
    return {
        "original": data,
        "processed": {k: v * 2 if isinstance(v, (int, float)) else v for k, v in data.items()},
        "timestamp": datetime.now().isoformat()
    }

# ========== Asynchronous Functions ==========

@node.register()
async def async_computation(data: List[int], delay: float = 1.0) -> Dict[str, Any]:
    """Asynchronous computation with configurable delay"""
    logger.info(f"Starting async computation for {len(data)} items with {delay}s delay")
    
    await asyncio.sleep(delay)  # Simulate async work
    
    result = {
        "input_data": data,
        "sum": sum(data),
        "average": sum(data) / len(data) if data else 0,
        "max": max(data) if data else None,
        "min": min(data) if data else None,
        "processing_time": delay,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Async computation completed: sum={result['sum']}")
    return result

@node.register()
async def async_ai_simulation(text: str, model_delay: float = 2.0) -> Dict[str, Any]:
    """Simulate async AI processing"""
    logger.info(f"Starting AI simulation for text: '{text[:50]}...'")
    
    # Simulate AI model loading and processing
    await asyncio.sleep(model_delay)
    
    # Mock AI results
    sentiment_score = random.uniform(-1, 1)
    confidence = random.uniform(0.7, 0.95)
    
    result = {
        "input_text": text,
        "sentiment": "positive" if sentiment_score > 0 else "negative",
        "sentiment_score": round(sentiment_score, 3),
        "confidence": round(confidence, 3),
        "word_count": len(text.split()),
        "processing_time": model_delay,
        "model": "mock_ai_v1.0",
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"AI simulation completed: sentiment={result['sentiment']}")
    return result

# ========== Streaming Functions ==========

@node.register()
def sync_number_stream(start: int, count: int, interval: float = 0.5) -> str:
    """Synchronous streaming number generator - returns collected stream data"""
    logger.info(f"Starting sync number stream: {start} to {start + count - 1}")
    
    results = []
    for i in range(count):
        number = start + i
        result = {
            "number": number,
            "square": number ** 2,
            "cube": number ** 3,
            "timestamp": datetime.now().isoformat(),
            "progress": f"{i + 1}/{count}"
        }
        
        results.append(json.dumps(result))
        time.sleep(interval)  # Synchronous sleep
    
    logger.info("Sync number stream completed")
    return "\n".join(results) + "\n"

@node.register()
async def async_data_stream(config: Dict[str, Any]) -> str:
    """Asynchronous streaming data generator - returns collected stream data"""
    sensors = config.get('sensors', ['temperature', 'humidity'])
    sample_rate = config.get('sample_rate', 1)  # samples per second
    duration = config.get('duration', 10)  # seconds
    
    logger.info(f"Starting async data stream: {sensors} @ {sample_rate}Hz for {duration}s")
    
    interval = 1.0 / sample_rate
    samples = int(duration * sample_rate)
    
    results = []
    for i in range(samples):
        # Generate mock sensor data
        readings = {}
        for sensor in sensors:
            base_values = {'temperature': 25.0, 'humidity': 60.0, 'pressure': 1013.25}
            base = base_values.get(sensor, 0.0)
            noise = random.gauss(0, 1)
            readings[sensor] = round(base + noise, 2)
        
        result = {
            "sample_id": i + 1,
            "readings": readings,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": round(i * interval, 2),
            "progress": f"{i + 1}/{samples}"
        }
        
        results.append(json.dumps(result))
        await asyncio.sleep(interval)  # Asynchronous sleep
    
    logger.info("Async data stream completed")
    return "\n".join(results) + "\n"

@node.register()
async def async_ml_inference_stream(images: List[str], batch_size: int = 2) -> str:
    """Asynchronous ML inference streaming - returns collected stream data"""
    logger.info(f"Starting ML inference stream: {len(images)} images, batch_size={batch_size}")
    
    classes = ['cat', 'dog', 'bird', 'car', 'airplane', 'ship']
    
    results = []
    # Process in batches
    for batch_idx in range(0, len(images), batch_size):
        batch = images[batch_idx:batch_idx + batch_size]
        
        # Simulate batch processing time
        processing_time = len(batch) * 0.3
        await asyncio.sleep(processing_time)
        
        batch_results = []
        for img_idx, image_name in enumerate(batch):
            # Mock inference result
            prediction = {
                "class": random.choice(classes),
                "confidence": round(random.uniform(0.7, 0.95), 3),
                "inference_time": 0.1
            }
            
            batch_results.append({
                "image": image_name,
                "prediction": prediction
            })
        
        result = {
            "batch_id": batch_idx // batch_size + 1,
            "batch_size": len(batch),
            "results": batch_results,
            "processing_time": processing_time,
            "total_processed": min(batch_idx + batch_size, len(images)),
            "total_images": len(images),
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(json.dumps(result))
    
    logger.info("ML inference stream completed")
    return "\n".join(results) + "\n"

# ========== Complex Async Generator ==========

@node.register()
async def async_complex_pipeline(data_source: str, config: Dict[str, Any]) -> str:
    """Complex async pipeline with multiple processing stages - returns collected stream data"""
    stages = config.get('stages', 3)
    items_per_stage = config.get('items_per_stage', 5)
    stage_delay = config.get('stage_delay', 1.0)
    
    logger.info(f"Starting complex pipeline: {stages} stages, {items_per_stage} items/stage")
    
    results = []
    for stage in range(1, stages + 1):
        logger.info(f"Processing stage {stage}/{stages}")
        
        for item in range(1, items_per_stage + 1):
            # Simulate complex processing
            await asyncio.sleep(stage_delay / items_per_stage)
            
            result = {
                "pipeline": data_source,
                "stage": stage,
                "item": item,
                "data": {
                    "processed_value": stage * item * random.randint(10, 100),
                    "quality_score": round(random.uniform(0.8, 1.0), 3),
                    "metadata": {
                        "stage_name": f"processing_stage_{stage}",
                        "complexity": "high" if stage > 2 else "medium"
                    }
                },
                "progress": {
                    "stage_progress": f"{item}/{items_per_stage}",
                    "total_progress": f"{(stage-1)*items_per_stage + item}/{stages*items_per_stage}"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(json.dumps(result))
    
    # Final summary
    summary = {
        "pipeline": data_source,
        "status": "completed",
        "total_stages": stages,
        "total_items": stages * items_per_stage,
        "completion_time": datetime.now().isoformat()
    }
    
    results.append(json.dumps(summary))
    logger.info("Complex pipeline completed")
    return "\n".join(results) + "\n"

def main():
    """Start the compute node"""
    logger.info("🚀 Starting Async & Streaming Compute Node (Fixed Version)")
    logger.info("📋 Registered functions:")
    logger.info("  Sync Functions:")
    logger.info("    - sync_add(a, b)")
    logger.info("    - sync_process_data(data)")
    logger.info("  Async Functions:")
    logger.info("    - async_computation(data, delay)")
    logger.info("    - async_ai_simulation(text, model_delay)")
    logger.info("  Stream-like Functions (return collected data):")
    logger.info("    - sync_number_stream(start, count, interval) -> str")
    logger.info("    - async_data_stream(config) -> str")
    logger.info("    - async_ml_inference_stream(images, batch_size) -> str")
    logger.info("    - async_complex_pipeline(data_source, config) -> str")
    logger.info("💡 Note: Stream functions return newline-separated JSON data")
    
    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        node.stop()
        logger.info("✅ Compute node stopped gracefully")

if __name__ == "__main__":
    main() 