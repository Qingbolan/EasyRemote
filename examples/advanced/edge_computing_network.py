# Advanced Demo: Edge Computing Network
# This demonstrates a distributed edge computing network with IoT sensors and real-time processing

from easyremote import ComputeNode, remote
import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional

# ================================
# Edge Computing Nodes
# ================================

# IoT Sensor Edge Node
iot_sensor_node = ComputeNode(
    vps_address="edge-gateway.example.com:8080",
    node_id="iot-sensor-edge-01",
    capabilities={
        "location": {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco"},
        "sensors": ["temperature", "humidity", "air_quality", "noise"],
        "edge_type": "iot_sensor",
        "power_source": "battery"
    }
)

@iot_sensor_node.register(interval=5)  # Report every 5 seconds
async def collect_environmental_data():
    """Collect environmental sensor data from IoT devices"""
    # Simulate sensor readings
    sensor_data = {
        "timestamp": time.time(),
        "location": {"lat": 37.7749, "lng": -122.4194},
        "sensors": {
            "temperature": 20 + random.uniform(-5, 15),  # 15-35°C
            "humidity": 45 + random.uniform(-15, 25),    # 30-70%
            "air_quality": random.randint(30, 150),      # AQI
            "noise_level": 35 + random.uniform(0, 40)    # 35-75 dB
        },
        "device_info": {
            "battery_level": 85 + random.uniform(-20, 15),
            "signal_strength": -60 + random.uniform(-20, 15),
            "node_id": "iot-sensor-edge-01"
        }
    }
    
    print(f"📊 IoT Sensor: T={sensor_data['sensors']['temperature']:.1f}°C, "
          f"AQI={sensor_data['sensors']['air_quality']}")
    return sensor_data

@iot_sensor_node.register
async def detect_anomalies(current_data: Dict[str, Any], historical_data: List[Dict[str, Any]]):
    """Local anomaly detection on edge device"""
    anomalies = []
    
    # Temperature anomaly detection
    temp = current_data["sensors"]["temperature"]
    if temp > 35 or temp < 5:
        anomalies.append({
            "type": "temperature",
            "severity": "high" if temp > 40 or temp < 0 else "medium",
            "value": temp,
            "threshold": "5-35°C"
        })
    
    # Air quality anomaly detection
    aqi = current_data["sensors"]["air_quality"]
    if aqi > 100:
        anomalies.append({
            "type": "air_quality",
            "severity": "high" if aqi > 150 else "medium",
            "value": aqi,
            "threshold": "< 100 AQI"
        })
    
    # Noise anomaly detection
    noise = current_data["sensors"]["noise_level"]
    if noise > 70:
        anomalies.append({
            "type": "noise",
            "severity": "medium",
            "value": noise,
            "threshold": "< 70 dB"
        })
    
    if anomalies:
        print(f"🚨 Anomalies detected: {len(anomalies)} issues")
    
    return {
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "detection_time": time.time(),
        "node_id": current_data["device_info"]["node_id"]
    }

# Mobile Edge Node (Vehicle)
mobile_edge_node = ComputeNode(
    vps_address="edge-gateway.example.com:8080",
    node_id="mobile-edge-vehicle-01",
    capabilities={
        "mobility": True,
        "gps": True,
        "camera": True,
        "lidar": True,
        "edge_type": "vehicle",
        "processing_power": "high"
    }
)

@mobile_edge_node.register
async def process_traffic_data(location: Dict[str, float], camera_data: bytes = None):
    """Process traffic data from mobile edge device (vehicle)"""
    # Simulate traffic analysis
    traffic_analysis = {
        "location": location,
        "timestamp": time.time(),
        "traffic_density": random.uniform(0.1, 1.0),
        "average_speed": random.uniform(20, 80),  # km/h
        "vehicle_count": random.randint(5, 50),
        "road_conditions": random.choice(["excellent", "good", "fair", "poor"]),
        "weather_impact": random.choice(["none", "light", "moderate", "severe"]),
        "processing_node": "mobile-edge-vehicle-01"
    }
    
    # Detect traffic incidents
    incidents = []
    if traffic_analysis["average_speed"] < 30 and traffic_analysis["traffic_density"] > 0.7:
        incidents.append({
            "type": "traffic_jam",
            "severity": "medium",
            "estimated_delay": random.randint(5, 20)
        })
    
    if traffic_analysis["road_conditions"] == "poor":
        incidents.append({
            "type": "road_hazard",
            "severity": "high",
            "description": "Poor road conditions detected"
        })
    
    traffic_analysis["incidents"] = incidents
    
    print(f"🚗 Traffic: {traffic_analysis['vehicle_count']} vehicles, "
          f"avg speed {traffic_analysis['average_speed']:.1f} km/h")
    
    return traffic_analysis

@mobile_edge_node.register
async def coordinate_with_nearby_vehicles(current_location: Dict[str, float], 
                                        message: str, 
                                        broadcast_radius: float = 1.0):
    """Vehicle-to-vehicle communication for coordination"""
    # Simulate V2V communication
    nearby_vehicles = [
        {"vehicle_id": f"vehicle_{i}", "distance": random.uniform(0.1, broadcast_radius)}
        for i in range(random.randint(2, 8))
    ]
    
    coordination_data = {
        "origin_vehicle": "mobile-edge-vehicle-01",
        "location": current_location,
        "message": message,
        "broadcast_radius": broadcast_radius,
        "nearby_vehicles": nearby_vehicles,
        "coordination_time": time.time(),
        "message_type": "traffic_update"
    }
    
    print(f"📡 V2V: Broadcasted to {len(nearby_vehicles)} nearby vehicles")
    return coordination_data

# Smart Infrastructure Edge Node
infrastructure_node = ComputeNode(
    vps_address="edge-gateway.example.com:8080",
    node_id="smart-infrastructure-01",
    capabilities={
        "location": {"lat": 37.7849, "lng": -122.4094, "type": "traffic_light"},
        "infrastructure_type": "traffic_control",
        "edge_type": "infrastructure",
        "processing_power": "medium"
    }
)

@infrastructure_node.register
async def optimize_traffic_lights(traffic_data: List[Dict[str, Any]], 
                                 current_timing: Dict[str, int]):
    """Optimize traffic light timing based on real-time traffic data"""
    # Analyze traffic patterns
    total_vehicles = sum(data.get("vehicle_count", 0) for data in traffic_data)
    avg_density = sum(data.get("traffic_density", 0) for data in traffic_data) / len(traffic_data)
    
    # Calculate optimal timing
    if avg_density > 0.8:  # High traffic
        optimized_timing = {
            "north_south_green": 45,  # Longer green for main direction
            "east_west_green": 30,
            "yellow": 3,
            "all_red": 2
        }
        optimization_reason = "Heavy traffic detected"
    elif avg_density < 0.3:  # Light traffic
        optimized_timing = {
            "north_south_green": 25,
            "east_west_green": 25,  # Equal timing for light traffic
            "yellow": 3,
            "all_red": 2
        }
        optimization_reason = "Light traffic, balanced timing"
    else:  # Normal traffic
        optimized_timing = {
            "north_south_green": 35,
            "east_west_green": 35,
            "yellow": 3,
            "all_red": 2
        }
        optimization_reason = "Normal traffic conditions"
    
    optimization_result = {
        "current_timing": current_timing,
        "optimized_timing": optimized_timing,
        "traffic_analysis": {
            "total_vehicles": total_vehicles,
            "average_density": avg_density,
            "data_points": len(traffic_data)
        },
        "optimization_reason": optimization_reason,
        "estimated_improvement": random.uniform(10, 30),  # % improvement
        "node_id": "smart-infrastructure-01",
        "optimization_time": time.time()
    }
    
    print(f"🚦 Traffic Light: Optimized timing for {avg_density:.2f} density")
    return optimization_result

# Edge Analytics Node
edge_analytics_node = ComputeNode(
    vps_address="edge-gateway.example.com:8080",
    node_id="edge-analytics-01",
    capabilities={
        "processing_power": "very_high",
        "edge_type": "analytics",
        "ai_acceleration": True,
        "storage": "1TB"
    }
)

@edge_analytics_node.register(gpu_required=True)
async def aggregate_edge_data(sensor_data: List[Dict[str, Any]], 
                             traffic_data: List[Dict[str, Any]],
                             time_window: int = 300):
    """Aggregate and analyze data from multiple edge sources"""
    print(f"🧠 Analytics: Processing {len(sensor_data)} sensor + {len(traffic_data)} traffic records")
    
    # Environmental analysis
    if sensor_data:
        env_stats = {
            "avg_temperature": sum(d["sensors"]["temperature"] for d in sensor_data) / len(sensor_data),
            "avg_humidity": sum(d["sensors"]["humidity"] for d in sensor_data) / len(sensor_data),
            "avg_air_quality": sum(d["sensors"]["air_quality"] for d in sensor_data) / len(sensor_data),
            "avg_noise": sum(d["sensors"]["noise_level"] for d in sensor_data) / len(sensor_data)
        }
        
        # Environmental health score
        env_health_score = min(100, (
            (100 - max(0, env_stats["avg_air_quality"] - 50)) * 0.4 +
            (100 - max(0, env_stats["avg_noise"] - 50)) * 0.3 +
            (100 if 18 <= env_stats["avg_temperature"] <= 26 else 70) * 0.3
        ))
    else:
        env_stats = {}
        env_health_score = 0
    
    # Traffic analysis
    if traffic_data:
        traffic_stats = {
            "avg_speed": sum(d["average_speed"] for d in traffic_data) / len(traffic_data),
            "avg_density": sum(d["traffic_density"] for d in traffic_data) / len(traffic_data),
            "total_vehicles": sum(d["vehicle_count"] for d in traffic_data),
            "incident_count": sum(len(d.get("incidents", [])) for d in traffic_data)
        }
        
        # Traffic efficiency score
        traffic_efficiency = min(100, (
            (traffic_stats["avg_speed"] / 60) * 50 +  # Speed factor
            (1 - traffic_stats["avg_density"]) * 30 +  # Density factor
            (1 - min(1, traffic_stats["incident_count"] / 10)) * 20  # Incident factor
        ) * 100)
    else:
        traffic_stats = {}
        traffic_efficiency = 0
    
    # Generate insights
    insights = []
    
    if env_health_score < 60:
        insights.append("Environmental conditions need attention - high pollution or noise levels")
    
    if traffic_efficiency < 50:
        insights.append("Traffic congestion detected - consider alternative routes")
    
    if sensor_data and any(d.get("anomalies", []) for d in sensor_data):
        insights.append("Multiple sensor anomalies detected - investigate potential issues")
    
    aggregated_result = {
        "analysis_window": time_window,
        "data_summary": {
            "sensor_records": len(sensor_data),
            "traffic_records": len(traffic_data),
            "analysis_time": time.time()
        },
        "environmental_analysis": {
            "statistics": env_stats,
            "health_score": env_health_score
        },
        "traffic_analysis": {
            "statistics": traffic_stats,
            "efficiency_score": traffic_efficiency
        },
        "insights": insights,
        "overall_city_score": (env_health_score + traffic_efficiency) / 2,
        "processing_node": "edge-analytics-01"
    }
    
    print(f"📈 Analytics: City score {aggregated_result['overall_city_score']:.1f}/100")
    return aggregated_result

# ================================
# Client Side - Edge Network Orchestration
# ================================

@remote(node_id="iot-sensor-edge-01")
async def collect_environmental_data():
    pass

@remote(node_id="iot-sensor-edge-01")
async def detect_anomalies(current_data: Dict[str, Any], historical_data: List[Dict[str, Any]]):
    pass

@remote(node_id="mobile-edge-vehicle-01")
async def process_traffic_data(location: Dict[str, float], camera_data: bytes = None):
    pass

@remote(node_id="mobile-edge-vehicle-01")
async def coordinate_with_nearby_vehicles(current_location: Dict[str, float], 
                                        message: str, 
                                        broadcast_radius: float = 1.0):
    pass

@remote(node_id="smart-infrastructure-01")
async def optimize_traffic_lights(traffic_data: List[Dict[str, Any]], 
                                 current_timing: Dict[str, int]):
    pass

@remote(node_id="edge-analytics-01")
async def aggregate_edge_data(sensor_data: List[Dict[str, Any]], 
                             traffic_data: List[Dict[str, Any]],
                             time_window: int = 300):
    pass

class EdgeComputingOrchestrator:
    """Orchestrator for the entire edge computing network"""
    
    def __init__(self):
        self.sensor_data_buffer = []
        self.traffic_data_buffer = []
        self.historical_data = []
        self.alerts = []
        
        # Current traffic light timing
        self.current_traffic_timing = {
            "north_south_green": 30,
            "east_west_green": 30,
            "yellow": 3,
            "all_red": 2
        }
    
    async def run_continuous_monitoring(self, duration: int = 60):
        """Run continuous edge computing monitoring"""
        print(f"🌐 Starting edge computing network monitoring for {duration}s")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Collect sensor data
            sensor_data = await collect_environmental_data()
            self.sensor_data_buffer.append(sensor_data)
            
            # Detect anomalies
            anomaly_result = await detect_anomalies(sensor_data, self.historical_data[-10:])
            
            if anomaly_result["anomalies"]:
                self.alerts.extend(anomaly_result["anomalies"])
                await self.handle_alerts(anomaly_result["anomalies"])
            
            # Simulate vehicle movement and traffic data collection
            vehicle_locations = [
                {"lat": 37.7749 + random.uniform(-0.01, 0.01), 
                 "lng": -122.4194 + random.uniform(-0.01, 0.01)}
                for _ in range(random.randint(2, 5))
            ]
            
            # Collect traffic data from multiple vehicles
            traffic_tasks = [
                process_traffic_data(location) for location in vehicle_locations
            ]
            traffic_results = await asyncio.gather(*traffic_tasks)
            self.traffic_data_buffer.extend(traffic_results)
            
            # Vehicle coordination (simulate emergency vehicle)
            if random.random() < 0.1:  # 10% chance of emergency
                coordination_result = await coordinate_with_nearby_vehicles(
                    vehicle_locations[0],
                    "Emergency vehicle approaching - please give way",
                    broadcast_radius=2.0
                )
                print(f"🚨 Emergency coordination: {coordination_result['message']}")
            
            # Optimize traffic lights every 30 seconds
            if len(self.traffic_data_buffer) >= 5 and (time.time() - start_time) % 30 < 5:
                optimization_result = await optimize_traffic_lights(
                    self.traffic_data_buffer[-10:], 
                    self.current_traffic_timing
                )
                
                # Update timing if significant improvement
                if optimization_result["estimated_improvement"] > 15:
                    self.current_traffic_timing = optimization_result["optimized_timing"]
                    print(f"🚦 Updated traffic timing: {optimization_result['optimization_reason']}")
            
            # Run analytics every 60 seconds
            if len(self.sensor_data_buffer) >= 3 and (time.time() - start_time) % 60 < 5:
                analytics_result = await aggregate_edge_data(
                    self.sensor_data_buffer[-20:],
                    self.traffic_data_buffer[-20:]
                )
                
                print(f"📊 City Analytics Update:")
                print(f"   • Environmental Health: {analytics_result['environmental_analysis']['health_score']:.1f}/100")
                print(f"   • Traffic Efficiency: {analytics_result['traffic_analysis']['efficiency_score']:.1f}/100")
                print(f"   • Overall City Score: {analytics_result['overall_city_score']:.1f}/100")
                
                if analytics_result["insights"]:
                    print(f"   • Insights: {len(analytics_result['insights'])} recommendations")
            
            # Update historical data
            self.historical_data.append(sensor_data)
            if len(self.historical_data) > 100:
                self.historical_data = self.historical_data[-50:]
            
            # Clean buffers
            if len(self.sensor_data_buffer) > 50:
                self.sensor_data_buffer = self.sensor_data_buffer[-25:]
            
            if len(self.traffic_data_buffer) > 50:
                self.traffic_data_buffer = self.traffic_data_buffer[-25:]
            
            await asyncio.sleep(5)  # 5-second monitoring interval
    
    async def handle_alerts(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies and alerts"""
        for anomaly in anomalies:
            alert_message = f"⚠️ {anomaly['type'].upper()} ALERT: " \
                          f"{anomaly['value']} (threshold: {anomaly['threshold']})"
            print(alert_message)
            
            # In real implementation, send to emergency services, city management, etc.
            if anomaly["severity"] == "high":
                print(f"🚨 HIGH PRIORITY: Immediate attention required for {anomaly['type']}")
    
    async def get_network_status(self):
        """Get current status of the edge computing network"""
        return {
            "active_sensors": 1,  # iot-sensor-edge-01
            "mobile_nodes": 1,    # mobile-edge-vehicle-01  
            "infrastructure_nodes": 1,  # smart-infrastructure-01
            "analytics_nodes": 1,  # edge-analytics-01
            "total_alerts": len(self.alerts),
            "sensor_data_points": len(self.sensor_data_buffer),
            "traffic_data_points": len(self.traffic_data_buffer),
            "current_traffic_timing": self.current_traffic_timing
        }

async def run_edge_computing_demo():
    """Run comprehensive edge computing network demo"""
    print("🌐 EasyRemote Edge Computing Network Demo")
    print("=" * 50)
    
    orchestrator = EdgeComputingOrchestrator()
    
    # Demo: Continuous edge monitoring
    print("\n📡 Demo: Real-time Edge Computing Network")
    print("-" * 40)
    
    await orchestrator.run_continuous_monitoring(duration=90)  # 90 seconds
    
    # Show final network status
    print("\n📈 Final Network Status")
    print("-" * 30)
    status = await orchestrator.get_network_status()
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    print("Starting edge computing nodes...")
    
    # In real scenario, these would run on separate edge devices
    # iot_sensor_node.serve()       # Run on IoT sensor device
    # mobile_edge_node.serve()      # Run on vehicle edge computer
    # infrastructure_node.serve()   # Run on traffic light controller
    # edge_analytics_node.serve()   # Run on edge analytics server
    
    # Run the edge computing demo
    asyncio.run(run_edge_computing_demo()) 