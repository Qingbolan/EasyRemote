# Advanced Demo: Multi-Node Load Balancing
# This example demonstrates how multiple nodes can provide the same function
# and EasyRemote automatically distributes load across them

from easyremote import ComputeNode, remote
import asyncio
import time
import random

# ================================
# Multiple GPU Nodes Providing Same Function
# ================================

# GPU Node 1 - High-end workstation
gpu_node_1 = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="gpu-workstation-01",
    capabilities={
        "gpu": {"model": "RTX 4090", "memory": "24GB"},
        "max_concurrent": 3,
        "priority": "high"
    }
)

@gpu_node_1.register(
    load_balancing=True,  # Enable load balancing for this function
    max_concurrent=3,     # Maximum concurrent executions
    timeout=300          # 5 minute timeout
)
def train_ai_model(model_config, dataset_size, epochs=10):
    """Train AI model on GPU Node 1"""
    print(f"[GPU Node 1] Starting training: {model_config['name']}")
    
    # Simulate training time based on dataset size and epochs
    training_time = (dataset_size / 1000) * epochs * random.uniform(0.8, 1.2)
    time.sleep(training_time)
    
    result = {
        "model_name": model_config['name'],
        "accuracy": random.uniform(0.85, 0.98),
        "training_time": training_time,
        "node_id": "gpu-workstation-01",
        "gpu_utilization": random.uniform(0.7, 0.95)
    }
    
    print(f"[GPU Node 1] Training completed: {result['accuracy']:.3f} accuracy")
    return result

# GPU Node 2 - Gaming PC
gpu_node_2 = ComputeNode(
    vps_address="gateway.example.com:8080", 
    node_id="gaming-pc-02",
    capabilities={
        "gpu": {"model": "RTX 3080", "memory": "12GB"},
        "max_concurrent": 2,
        "priority": "medium"
    }
)

@gpu_node_2.register(
    load_balancing=True,
    max_concurrent=2,
    timeout=400  # Slower GPU, longer timeout
)
def train_ai_model(model_config, dataset_size, epochs=10):
    """Train AI model on GPU Node 2 (same function name!)"""
    print(f"[GPU Node 2] Starting training: {model_config['name']}")
    
    # Slightly slower due to less powerful GPU
    training_time = (dataset_size / 800) * epochs * random.uniform(0.9, 1.3)
    time.sleep(training_time)
    
    result = {
        "model_name": model_config['name'],
        "accuracy": random.uniform(0.82, 0.95),
        "training_time": training_time,
        "node_id": "gaming-pc-02",
        "gpu_utilization": random.uniform(0.8, 0.98)
    }
    
    print(f"[GPU Node 2] Training completed: {result['accuracy']:.3f} accuracy")
    return result

# GPU Node 3 - Cloud instance
gpu_node_3 = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="cloud-gpu-03",
    capabilities={
        "gpu": {"model": "A100", "memory": "40GB"},
        "max_concurrent": 5,
        "priority": "highest",
        "cost_per_hour": 3.20
    }
)

@gpu_node_3.register(
    load_balancing=True,
    max_concurrent=5,
    timeout=200,  # Fastest GPU
    cost_aware=True
)
def train_ai_model(model_config, dataset_size, epochs=10):
    """Train AI model on Cloud GPU Node"""
    print(f"[Cloud GPU] Starting training: {model_config['name']}")
    
    # Fastest training due to A100
    training_time = (dataset_size / 1500) * epochs * random.uniform(0.6, 0.9)
    time.sleep(training_time)
    
    result = {
        "model_name": model_config['name'],
        "accuracy": random.uniform(0.88, 0.99),
        "training_time": training_time,
        "node_id": "cloud-gpu-03",
        "gpu_utilization": random.uniform(0.6, 0.85),
        "cost": training_time * 3.20 / 3600  # Cost calculation
    }
    
    print(f"[Cloud GPU] Training completed: {result['accuracy']:.3f} accuracy")
    return result

# ================================
# Client Side - Load Balanced Function Calls
# ================================

# The @remote decorator automatically load balances across all nodes
# that provide the "train_ai_model" function
@remote(function_name="train_ai_model", load_balancing="smart")
def train_ai_model(model_config, dataset_size, epochs=10):
    """Remote function that automatically load balances"""
    pass

# Advanced remote call with load balancing options
@remote(
    function_name="train_ai_model",
    load_balancing={
        "strategy": "resource_aware",  # Choose based on current resource usage
        "prefer_local": True,          # Prefer local network nodes
        "cost_optimization": True,     # Consider cost in selection
        "max_latency": 100            # Maximum acceptable latency (ms)
    }
)
def train_ai_model_optimized(model_config, dataset_size, epochs=10):
    """Optimized remote function with advanced load balancing"""
    pass

async def run_distributed_training_demo():
    """Demonstrate distributed training across multiple GPU nodes"""
    
    print("🚀 Starting Distributed AI Training Demo")
    print("=" * 50)
    
    # Define multiple training jobs
    training_jobs = [
        {"name": "ResNet50", "size": 1000, "epochs": 5},
        {"name": "BERT-Large", "size": 2000, "epochs": 3},
        {"name": "GPT-Small", "size": 1500, "epochs": 4},
        {"name": "ViT-Base", "size": 1200, "epochs": 6},
        {"name": "T5-Small", "size": 800, "epochs": 8},
        {"name": "CNN-Custom", "size": 600, "epochs": 10}
    ]
    
    # Method 1: Automatic load balancing
    print("\n📊 Method 1: Automatic Load Balancing")
    print("-" * 30)
    
    start_time = time.time()
    
    # Submit all jobs simultaneously - EasyRemote handles distribution
    tasks = []
    for job in training_jobs:
        task = train_ai_model(
            model_config={"name": job["name"]},
            dataset_size=job["size"],
            epochs=job["epochs"]
        )
        tasks.append(task)
    
    # Wait for all training jobs to complete
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ All training jobs completed in {total_time:.2f} seconds")
    print("\nResults Summary:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['model_name']}: "
              f"{result['accuracy']:.3f} accuracy "
              f"({result['training_time']:.2f}s on {result['node_id']})")
    
    # Method 2: Cost-optimized load balancing
    print("\n💰 Method 2: Cost-Optimized Training")
    print("-" * 30)
    
    # Use cost-aware load balancing for budget-conscious training
    cost_optimized_jobs = training_jobs[:3]  # First 3 jobs only
    
    start_time = time.time()
    tasks = []
    
    for job in cost_optimized_jobs:
        task = train_ai_model_optimized(
            model_config={"name": f"{job['name']}-CostOpt"},
            dataset_size=job["size"],
            epochs=job["epochs"]
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    total_cost = sum(r.get('cost', 0) for r in results)
    
    print(f"\n✅ Cost-optimized training completed in {total_time:.2f} seconds")
    print(f"💵 Total cost: ${total_cost:.4f}")
    
    # Method 3: Real-time load monitoring
    print("\n📈 Method 3: Real-time Load Monitoring")
    print("-" * 30)
    
    # Simulate real-time load monitoring and dynamic allocation
    for i, job in enumerate(training_jobs[:3]):
        print(f"\nSubmitting job {i+1}: {job['name']}")
        
        # Get current node status before submitting
        node_status = await get_node_status()
        print(f"Available nodes: {len(node_status['available_nodes'])}")
        
        result = await train_ai_model(
            model_config={"name": job['name']},
            dataset_size=job["size"],
            epochs=job["epochs"]
        )
        
        print(f"✅ Completed on {result['node_id']}: {result['accuracy']:.3f} accuracy")

async def get_node_status():
    """Mock function to get current node status"""
    # In real implementation, this would query the gateway
    return {
        "available_nodes": ["gpu-workstation-01", "gaming-pc-02", "cloud-gpu-03"],
        "total_capacity": 10,
        "current_load": random.uniform(0.3, 0.7)
    }

if __name__ == "__main__":
    print("Starting GPU nodes...")
    
    # In real scenario, these would run on separate machines
    # gpu_node_1.serve()  # Run on workstation
    # gpu_node_2.serve()  # Run on gaming PC  
    # gpu_node_3.serve()  # Run on cloud instance
    
    # Run the demo
    asyncio.run(run_distributed_training_demo()) 