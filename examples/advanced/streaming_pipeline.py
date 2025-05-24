# Advanced Demo: Streaming Data Processing Pipeline
# This demonstrates real-time streaming data processing across distributed nodes

from easyremote import ComputeNode, remote
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any

# ================================
# Streaming Data Processing Nodes
# ================================

# Data Ingestion Node
ingestion_node = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="data-ingestion-01"
)

@ingestion_node.register(streaming=True, buffer_size=1000)
async def ingest_sensor_data(data_source: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Ingest real-time sensor data from various sources"""
    print(f"📡 Starting data ingestion from {data_source}")
    
    # Simulate real-time sensor data
    sensor_id = 1
    while True:
        # Generate mock sensor data
        sensor_data = {
            "sensor_id": f"{data_source}_{sensor_id}",
            "timestamp": time.time(),
            "temperature": 20 + 10 * (0.5 - hash(str(time.time())) % 100 / 100),
            "humidity": 50 + 20 * (0.5 - hash(str(time.time() + 1)) % 100 / 100),
            "pressure": 1013 + 50 * (0.5 - hash(str(time.time() + 2)) % 100 / 100),
            "location": {"lat": 37.7749, "lng": -122.4194}
        }
        
        print(f"📊 Ingested: Sensor {sensor_id} - Temp: {sensor_data['temperature']:.1f}°C")
        yield sensor_data
        
        sensor_id += 1
        await asyncio.sleep(1)  # 1 second intervals

# Data Processing Node 1 - Data Cleaning
processing_node_1 = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="data-processor-cleaning"
)

@processing_node_1.register(streaming=True, max_concurrent=5)
async def clean_sensor_data(raw_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Clean and validate incoming sensor data"""
    print("🧹 Starting data cleaning pipeline")
    
    async for data in raw_data_stream:
        # Data validation and cleaning
        cleaned_data = data.copy()
        
        # Remove outliers
        if abs(cleaned_data["temperature"]) > 100:
            print(f"⚠️  Outlier detected: Temperature {cleaned_data['temperature']}°C")
            continue
            
        # Normalize data
        cleaned_data["temperature_normalized"] = (cleaned_data["temperature"] - 20) / 30
        cleaned_data["humidity_normalized"] = cleaned_data["humidity"] / 100
        
        # Add data quality score
        cleaned_data["quality_score"] = 0.9 if all([
            -50 < cleaned_data["temperature"] < 50,
            0 < cleaned_data["humidity"] < 100,
            900 < cleaned_data["pressure"] < 1100
        ]) else 0.6
        
        cleaned_data["processing_stage"] = "cleaned"
        print(f"✨ Cleaned: {cleaned_data['sensor_id']} (Quality: {cleaned_data['quality_score']})")
        
        yield cleaned_data
        await asyncio.sleep(0.1)  # Processing delay

# Data Processing Node 2 - Feature Engineering
processing_node_2 = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="data-processor-features"
)

@processing_node_2.register(streaming=True, max_concurrent=3)
async def extract_features(cleaned_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Extract advanced features from sensor data"""
    print("🔧 Starting feature extraction pipeline")
    
    data_buffer = []
    
    async for data in cleaned_data_stream:
        data_buffer.append(data)
        
        # Process when we have enough data for time-series features
        if len(data_buffer) >= 5:
            current_data = data_buffer[-1]
            
            # Time-series features
            temps = [d["temperature"] for d in data_buffer[-5:]]
            features = {
                "temperature_trend": (temps[-1] - temps[0]) / len(temps),
                "temperature_volatility": sum(abs(temps[i] - temps[i-1]) for i in range(1, len(temps))) / (len(temps) - 1),
                "temperature_moving_avg": sum(temps) / len(temps)
            }
            
            # Combine with original data
            enriched_data = {**current_data, **features}
            enriched_data["processing_stage"] = "feature_extracted"
            
            print(f"🎯 Features extracted: {enriched_data['sensor_id']} - Trend: {features['temperature_trend']:.3f}")
            
            yield enriched_data
            
            # Keep buffer size manageable
            if len(data_buffer) > 10:
                data_buffer = data_buffer[-5:]
        
        await asyncio.sleep(0.2)

# Analytics Node - Real-time Analysis
analytics_node = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="real-time-analytics"
)

@analytics_node.register(streaming=True, gpu_accelerated=True)
async def real_time_analytics(feature_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Perform real-time analytics on processed data"""
    print("📈 Starting real-time analytics")
    
    anomaly_threshold = 0.7
    alert_buffer = []
    
    async for data in feature_data_stream:
        # Anomaly detection using multiple features
        anomaly_score = 0.0
        
        # Check temperature volatility
        if data["temperature_volatility"] > 2.0:
            anomaly_score += 0.3
        
        # Check trend changes
        if abs(data["temperature_trend"]) > 1.0:
            anomaly_score += 0.4
        
        # Check data quality
        if data["quality_score"] < 0.8:
            anomaly_score += 0.3
        
        # Generate analytics result
        analytics_result = {
            **data,
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > anomaly_threshold,
            "risk_level": "high" if anomaly_score > 0.8 else "medium" if anomaly_score > 0.5 else "low",
            "processing_stage": "analyzed"
        }
        
        # Generate alerts for anomalies
        if analytics_result["is_anomaly"]:
            alert = {
                "alert_id": f"ALERT_{int(time.time())}",
                "sensor_id": data["sensor_id"],
                "anomaly_score": anomaly_score,
                "alert_time": time.time(),
                "message": f"Anomaly detected: Score {anomaly_score:.2f}"
            }
            alert_buffer.append(alert)
            print(f"🚨 ALERT: {alert['message']} for {data['sensor_id']}")
        
        analytics_result["recent_alerts"] = alert_buffer[-5:]  # Keep last 5 alerts
        
        print(f"📊 Analytics: {data['sensor_id']} - Risk: {analytics_result['risk_level']}")
        yield analytics_result
        
        await asyncio.sleep(0.1)

# ================================
# Client Side - Streaming Pipeline Orchestration
# ================================

@remote(node_id="data-ingestion-01", streaming=True)
async def ingest_sensor_data(data_source: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Remote streaming data ingestion"""
    async for data in []:  # Implementation is remote
        yield data

@remote(node_id="data-processor-cleaning", streaming=True)
async def clean_sensor_data(raw_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Remote streaming data cleaning"""
    async for data in []:  # Implementation is remote
        yield data

@remote(node_id="data-processor-features", streaming=True)
async def extract_features(cleaned_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Remote streaming feature extraction"""
    async for data in []:  # Implementation is remote
        yield data

@remote(node_id="real-time-analytics", streaming=True)
async def real_time_analytics(feature_data_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
    """Remote streaming analytics"""
    async for data in []:  # Implementation is remote
        yield data

class StreamingPipelineOrchestrator:
    """Orchestrator for the entire streaming pipeline"""
    
    def __init__(self):
        self.active_pipelines = {}
        self.metrics = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "processing_rate": 0.0
        }
    
    async def start_pipeline(self, data_source: str, pipeline_id: str = None):
        """Start a complete streaming data pipeline"""
        if pipeline_id is None:
            pipeline_id = f"pipeline_{data_source}_{int(time.time())}"
        
        print(f"🚀 Starting streaming pipeline: {pipeline_id}")
        print(f"📍 Data source: {data_source}")
        
        # Create the pipeline chain
        raw_stream = ingest_sensor_data(data_source)
        cleaned_stream = clean_sensor_data(raw_stream)
        feature_stream = extract_features(cleaned_stream)
        analytics_stream = real_time_analytics(feature_stream)
        
        # Store pipeline reference
        self.active_pipelines[pipeline_id] = {
            "data_source": data_source,
            "start_time": time.time(),
            "status": "running"
        }
        
        # Process the final analytics stream
        async for result in analytics_stream:
            await self.handle_analytics_result(result, pipeline_id)
    
    async def handle_analytics_result(self, result: Dict[str, Any], pipeline_id: str):
        """Handle final analytics results"""
        # Update metrics
        self.metrics["total_processed"] += 1
        
        if result["is_anomaly"]:
            self.metrics["anomalies_detected"] += 1
            await self.handle_anomaly_alert(result, pipeline_id)
        
        # Calculate processing rate
        pipeline_info = self.active_pipelines[pipeline_id]
        elapsed_time = time.time() - pipeline_info["start_time"]
        self.metrics["processing_rate"] = self.metrics["total_processed"] / elapsed_time
        
        # Log result
        print(f"✅ Pipeline {pipeline_id}: Processed {result['sensor_id']} "
              f"(Risk: {result['risk_level']}, Rate: {self.metrics['processing_rate']:.2f}/s)")
        
        # Store result for monitoring
        if "results" not in pipeline_info:
            pipeline_info["results"] = []
        pipeline_info["results"].append(result)
        
        # Keep only recent results
        if len(pipeline_info["results"]) > 100:
            pipeline_info["results"] = pipeline_info["results"][-50:]
    
    async def handle_anomaly_alert(self, result: Dict[str, Any], pipeline_id: str):
        """Handle anomaly alerts"""
        alert_message = (f"🚨 ANOMALY ALERT in {pipeline_id}\n"
                        f"Sensor: {result['sensor_id']}\n"
                        f"Score: {result['anomaly_score']:.3f}\n"
                        f"Risk Level: {result['risk_level']}")
        
        print(alert_message)
        
        # In real implementation, send to alerting system
        # await send_to_slack(alert_message)
        # await send_to_pagerduty(result)
    
    async def run_multiple_pipelines(self, data_sources: list):
        """Run multiple streaming pipelines concurrently"""
        print(f"🌊 Starting {len(data_sources)} concurrent streaming pipelines")
        
        tasks = []
        for source in data_sources:
            task = asyncio.create_task(self.start_pipeline(source))
            tasks.append(task)
        
        # Run for a limited time (in production, would run indefinitely)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=30.0)
        except asyncio.TimeoutError:
            print("⏰ Demo timeout reached, stopping pipelines...")
        
        print(f"📊 Final metrics: {self.metrics}")
    
    def get_pipeline_status(self):
        """Get status of all active pipelines"""
        return {
            "active_pipelines": len(self.active_pipelines),
            "metrics": self.metrics,
            "pipeline_details": self.active_pipelines
        }

async def run_streaming_demo():
    """Run the complete streaming pipeline demo"""
    print("🌊 EasyRemote Streaming Pipeline Demo")
    print("=" * 50)
    
    orchestrator = StreamingPipelineOrchestrator()
    
    # Demo 1: Single pipeline
    print("\n📊 Demo 1: Single Streaming Pipeline")
    print("-" * 30)
    
    try:
        await asyncio.wait_for(
            orchestrator.start_pipeline("temperature_sensors_zone_a"),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        print("⏰ Single pipeline demo completed")
    
    # Demo 2: Multiple concurrent pipelines
    print("\n🌊 Demo 2: Multiple Concurrent Pipelines")
    print("-" * 30)
    
    data_sources = [
        "temperature_sensors_zone_a",
        "humidity_sensors_zone_b", 
        "pressure_sensors_zone_c",
        "environmental_station_01"
    ]
    
    await orchestrator.run_multiple_pipelines(data_sources)
    
    # Display final status
    print("\n📈 Pipeline Status Summary")
    print("-" * 30)
    status = orchestrator.get_pipeline_status()
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    print("Starting streaming pipeline nodes...")
    
    # In real scenario, these would run on separate machines
    # ingestion_node.serve()
    # processing_node_1.serve()
    # processing_node_2.serve()
    # analytics_node.serve()
    
    # Run the streaming demo
    asyncio.run(run_streaming_demo()) 