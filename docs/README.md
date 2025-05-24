# EasyRemote Documentation

Welcome to EasyRemote - A lightweight framework for hassle-free remote computing resource sharing.

## 📚 Documentation Structure

### 🎯 Design & Architecture
- [**Project Overview**](design/overview.md) - Project vision, core concepts and value propositions
- [**Architecture Design**](design/architecture.md) - System architecture and technical design
- [**Node Types & Roles**](design/roles.md) - Different node types and their responsibilities  
- [**Application Scenarios**](design/scenarios.md) - Real-world use cases and examples
- [**Load Balancing Design**](design/load_balancing.md) - Multi-node load balancing mechanisms

### 🚀 Getting Started
- [**Quick Start Guide**](tutorials/quick-start.md) - Get up and running in minutes
- [**Installation Guide**](tutorials/installation.md) - Detailed installation instructions
- [**Basic Tutorial**](tutorials/basic-tutorial.md) - Learn the fundamentals

### 📖 User Guide
- [**API Reference**](api/README.md) - Complete API documentation
- [**Configuration Guide**](guides/configuration.md) - Configuration options and best practices
- [**Deployment Guide**](guides/deployment.md) - Production deployment strategies

### 🔧 Development
- [**Development Setup**](development/setup.md) - Setting up development environment
- [**Contributing Guide**](development/contributing.md) - How to contribute to the project
- [**Roadmap**](development/roadmap.md) - Project roadmap and future plans

### 🛠️ Operations
- [**Troubleshooting**](operations/troubleshooting.md) - Common issues and solutions
- [**Monitoring**](operations/monitoring.md) - Monitoring and observability
- [**Reconnection Guide**](RECONNECTION_GUIDE.md) - Network resilience and reconnection

### 🤖 AI & Agent Integration
- [**MCP Integration**](ai/mcp-integration.md) - Model Context Protocol compatibility
- [**AI Agent Development**](ai/agent-development.md) - Building AI agents with EasyRemote
- [**Function Registry**](ai/function-registry.md) - Managing distributed function libraries

### 🎯 Advanced Examples
- [**Examples Collection**](../examples/README.md) - Comprehensive examples and demos
- [**Multi-Node Load Balancing Demo**](../examples/advanced/multi_node_load_balancing.py) - Load balancing across multiple nodes
- [**Streaming Pipeline Demo**](../examples/advanced/streaming_pipeline.py) - Real-time data processing
- [**Distributed AI Agents Demo**](../examples/advanced/distributed_ai_agents.py) - AI agent collaboration
- [**Edge Computing Network Demo**](../examples/advanced/edge_computing_network.py) - IoT and edge scenarios

## 🎯 Quick Navigation

### For New Users
1. Read the [Project Overview](design/overview.md) to understand EasyRemote
2. Follow the [Quick Start Guide](tutorials/quick-start.md) to get running
3. Explore [Application Scenarios](design/scenarios.md) for inspiration

### For Developers
1. Review the [Architecture Design](design/architecture.md)
2. Set up [Development Environment](development/setup.md) 
3. Check the [API Reference](api/README.md)

### For Operators
1. Read the [Deployment Guide](guides/deployment.md)
2. Configure [Monitoring](operations/monitoring.md)
3. Review [Troubleshooting](operations/troubleshooting.md)

## 🔑 Core Concepts

### The EasyRemote Way
```python
# On compute provider (your local machine with GPU)
from easyremote import ComputeNode

node = ComputeNode("your-vps:8080", "gpu-node-1")

@node.register
def run_ai_model(data):
    return your_gpu_model(data)  # Runs on your local GPU

node.serve()
```

```python
# On compute consumer (anywhere in the world)
from easyremote import remote

@remote(node_id="gpu-node-1")
def run_ai_model(data):
    pass  # Implementation is remote

# Use like a local function!
result = run_ai_model(my_data)
```

### Multi-Node Load Balancing
```python
# Multiple nodes can provide the same function
# Node 1 - High-end workstation
@gpu_node_1.register(load_balancing=True)
def train_model(data):
    return train_on_rtx4090(data)

# Node 2 - Gaming PC  
@gpu_node_2.register(load_balancing=True)
def train_model(data):  # Same function name!
    return train_on_rtx3080(data)

# Node 3 - Cloud instance
@gpu_node_3.register(load_balancing=True) 
def train_model(data):  # Same function name!
    return train_on_a100(data)

# Client automatically load balances across all nodes
@remote(function_name="train_model", load_balancing="smart")
def train_model(data):
    pass

# Submit multiple tasks - automatically distributed
results = await asyncio.gather(*[
    train_model(dataset) for dataset in training_datasets
])
```

## 📞 Community & Support

- **GitHub**: [EasyRemote Repository](https://github.com/Qingbolan/EasyRemote)
- **Issues**: Report bugs and request features
- **Discussions**: Community discussions and Q&A

---

*Documentation Version: 1.0.0*  
*Last Updated: December 2024* 