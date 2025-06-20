# improved_compute_node.py
"""
Improved Async and Streaming Compute Node
改进的异步和流式计算节点 - 使用真正的流式处理
"""
import asyncio
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Generator, AsyncGenerator
from easyremote import ComputeNode
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create compute node
node = ComputeNode("localhost:8080", node_id="improved-async-stream-node")

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

# ========== True Streaming Functions ==========

@node.register(stream=True)
def sync_number_stream(start: int, count: int, interval: float = 0.5) -> Generator[Dict[str, Any], None, None]:
    """True synchronous streaming number generator"""
    logger.info(f"Starting sync number stream: {start} to {start + count - 1}")
    
    for i in range(count):
        number = start + i
        result = {
            "number": number,
            "square": number ** 2,
            "cube": number ** 3,
            "timestamp": datetime.now().isoformat(),
            "progress": f"{i + 1}/{count}",
            "stream_type": "sync_generator"
        }
        
        logger.info(f"Yielding number: {number}")
        yield result
        time.sleep(interval)  # Synchronous sleep
    
    logger.info("Sync number stream completed")

@node.register(stream=True)
async def async_data_stream(config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """True asynchronous streaming data generator"""
    sensors = config.get('sensors', ['temperature', 'humidity'])
    sample_rate = config.get('sample_rate', 0.2)  # samples per second
    duration = config.get('duration', 10)  # seconds
    
    logger.info(f"Starting async data stream: {sensors} @ {sample_rate}Hz for {duration}s")
    
    interval = 1.0 / sample_rate
    samples = int(duration * sample_rate)
    
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
            "progress": f"{i + 1}/{samples}",
            "stream_type": "async_generator"
        }
        
        logger.info(f"Yielding sample {i + 1}/{samples}")
        yield result
        await asyncio.sleep(interval)  # Asynchronous sleep
    
    logger.info("Async data stream completed")

@node.register(stream=True)
async def async_ml_inference_stream(images: List[str], batch_size: int = 2) -> AsyncGenerator[Dict[str, Any], None]:
    """True asynchronous ML inference streaming"""
    logger.info(f"Starting ML inference stream: {len(images)} images, batch_size={batch_size}")
    
    classes = ['cat', 'dog', 'bird', 'car', 'airplane', 'ship']
    
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
            "timestamp": datetime.now().isoformat(),
            "stream_type": "async_generator"
        }
        
        logger.info(f"Yielding batch {result['batch_id']}")
        yield result
    
    logger.info("ML inference stream completed")

@node.register(stream=True)
async def async_complex_pipeline(data_source: str, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """True complex async pipeline with real-time streaming"""
    stages = config.get('stages', 3)
    items_per_stage = config.get('items_per_stage', 5)
    stage_delay = config.get('stage_delay', 1.0)
    
    logger.info(f"Starting complex pipeline: {stages} stages, {items_per_stage} items/stage")
    
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
                "timestamp": datetime.now().isoformat(),
                "stream_type": "async_generator"
            }
            
            logger.info(f"Yielding stage {stage}, item {item}")
            yield result
    
    # Final summary
    summary = {
        "pipeline": data_source,
        "status": "completed",
        "total_stages": stages,
        "total_items": stages * items_per_stage,
        "completion_time": datetime.now().isoformat(),
        "stream_type": "async_generator"
    }
    
    logger.info("Complex pipeline completed")
    yield summary

# ========== Real-time Event Stream ==========

@node.register(stream=True)
async def real_time_event_stream(event_config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """Real-time event streaming with different event types"""
    event_types = event_config.get('event_types', ['user_action', 'system_alert', 'data_update'])
    event_rate = event_config.get('event_rate', 2)  # events per second
    duration = event_config.get('duration', 15)  # seconds
    
    logger.info(f"Starting real-time event stream: {event_types} @ {event_rate} events/sec for {duration}s")
    
    interval = 1.0 / event_rate
    total_events = int(duration * event_rate)
    
    for event_id in range(1, total_events + 1):
        event_type = random.choice(event_types)
        
        # Generate event-specific data
        event_data = {
            "user_action": {
                "action": random.choice(['click', 'scroll', 'type', 'navigate']),
                "user_id": f"user_{random.randint(1, 100)}",
                "page": f"/page/{random.randint(1, 10)}"
            },
            "system_alert": {
                "severity": random.choice(['info', 'warning', 'error']),
                "component": random.choice(['database', 'api', 'cache', 'queue']),
                "message": f"System event {event_id}"
            },
            "data_update": {
                "table": random.choice(['users', 'orders', 'products']),
                "operation": random.choice(['insert', 'update', 'delete']),
                "record_count": random.randint(1, 50)
            }
        }
        
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data[event_type],
            "metadata": {
                "source": "improved-async-stream-node",
                "priority": random.choice(['low', 'medium', 'high']),
                "correlation_id": f"corr_{random.randint(1000, 9999)}"
            },
            "stream_type": "real_time_events"
        }
        
        logger.info(f"Yielding event {event_id}: {event_type}")
        yield event
        await asyncio.sleep(interval)
    
    logger.info("Real-time event stream completed")

def main():
    """Start the improved compute node"""
    logger.info("🚀 Starting Improved Async & Streaming Compute Node")
    logger.info("📋 Registered functions:")
    logger.info("  Sync Functions:")
    logger.info("    - sync_add(a, b)")
    logger.info("    - sync_process_data(data)")
    logger.info("  Async Functions:")
    logger.info("    - async_computation(data, delay)")
    logger.info("    - async_ai_simulation(text, model_delay)")
    logger.info("  True Streaming Functions:")
    logger.info("    - sync_number_stream(start, count, interval) -> Generator")
    logger.info("    - async_data_stream(config) -> AsyncGenerator")
    logger.info("    - async_ml_inference_stream(images, batch_size) -> AsyncGenerator")
    logger.info("    - async_complex_pipeline(data_source, config) -> AsyncGenerator")
    logger.info("    - real_time_event_stream(event_config) -> AsyncGenerator")
    logger.info("💡 Note: These are TRUE streaming functions using generators")
    
    try:
        node.serve(blocking=True)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        node.stop()
        logger.info("✅ Improved compute node stopped gracefully")

if __name__ == "__main__":
    main() 