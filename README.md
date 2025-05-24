# EasyRemote: Building the Next-Generation Computing Internet - EasyNet

<div align="center">

![EasyRemote Logo](docs/easyremote-logo.png)

[![PyPI version](https://badge.fury.io/py/easyremote.svg)](https://badge.fury.io/py/easyremote)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/easyremote)]()

> **"Torchrun for the World"**: Enabling any terminal user to mobilize global computing resources with a single command to execute local code.

**🌐 Building the Next-Generation Computing Internet - EasyNet**

English | [中文](README_ZH.md)

</div>

---

## 🧠 From Private Functions to Global Computing Orchestration Engine

**EasyRemote is not just a Private Function-as-a-Service (Private FaaS) platform—it's our answer to the future of computing:**

> While current cloud computing models are platform-centric, requiring data and code to "go to the cloud" to exchange resources, we believe—  
> **The next-generation computing network should be terminal-centric, language-interfaced, function-granular, and trust-bounded**.

We call it: **"EasyNet"**.

### 🎯 Core Philosophy: Code as Resource, Device as Node, Execution as Collaboration

EasyRemote is the first-stage implementation of EasyNet, allowing you to:

* **🧠 Define task logic using familiar Python function structures**
* **🔒 Deploy computing nodes on any device while maintaining privacy, performance, and control**  
* **🌐 Transform local functions into globally accessible task interfaces through lightweight VPS gateways**
* **🚀 Eventually launch tasks as simply as using `torchrun`, automatically scheduling to the most suitable resources for execution**

### 💡 Our Paradigm Shift

| Traditional Cloud Computing | **EasyNet Mode** |
|------------|-------------|
| Platform-centric | **Terminal-centric** |
| Code must go to cloud | **Code stays on your device** |
| Pay for computing power | **Contribute to earn computing power** |
| Vendor lock-in | **Decentralized collaboration** |
| Cold start delays | **Always warm** |

---

## 🔭 Current Implementation: Private Function-as-a-Service

### **Quick Experience: Join EasyNet with 12 Lines of Code**

```python
# 1. Start gateway node (any VPS)
from easyremote import Server
Server(port=8080).start()

# 2. Contribute computing node (your device)
from easyremote import ComputeNode
node = ComputeNode("your-gateway:8080")

@node.register
def ai_inference(prompt):
    return your_local_model.generate(prompt)  # Runs on your GPU

node.serve()

# 3. Global computing access (anywhere)
from easyremote import Client
result = Client("your-gateway:8080").execute("ai_inference", "Hello AI")
```

**🎉 Your device has joined EasyNet!**

### **🆚 Comparison with Traditional Cloud Services**

| Feature | AWS Lambda | Google Cloud | **EasyNet Node** |
|------|------------|--------------|----------------|
| **Computing Location** | Cloud servers | Cloud servers | **Your device** |
| **Data Privacy** | Upload to cloud | Upload to cloud | **Never leaves local** |
| **Computing Cost** | $200+/million calls | $200+/million calls | **$5 gateway fee** |
| **Hardware Limitations** | Cloud specs | Cloud specs | **Your GPU/CPU** |
| **Startup Latency** | 100-1000ms | 100-1000ms | **0ms (always online)** |

---

## 📚 Complete Documentation Guide

### 🌐 Multilingual Documentation

#### 🇺🇸 English Documentation
- **[📖 English Documentation Center](docs/en/README.md)** - Complete English documentation navigation

#### 🇨🇳 Chinese Documentation
- **[📖 中文文档中心](docs/zh/README.md)** - Complete Chinese documentation navigation

### 🚀 Quick Start
- **[5-Minute Quick Start](docs/en/user-guide/quick-start.md)** - Fastest way to get started | [中文](docs/zh/user-guide/quick-start.md)
- **[Installation Guide](docs/en/user-guide/installation.md)** - Detailed installation instructions | [中文](docs/zh/user-guide/installation.md)

### 📖 User Guide
- **[API Reference](docs/en/user-guide/api-reference.md)** - Complete API documentation | [中文](docs/zh/user-guide/api-reference.md)
- **[Basic Tutorial](docs/en/tutorials/basic-usage.md)** - Detailed basic tutorial | [中文](docs/zh/tutorials/basic-usage.md)
- **[Advanced Scenarios](docs/en/tutorials/advanced-scenarios.md)** - Complex application implementation | [中文](docs/zh/tutorials/advanced-scenarios.md)

### 🏗️ Technical Deep Dive
- **[System Architecture](docs/en/architecture/overview.md)** - Overall architecture design | [中文](docs/zh/architecture/overview.md)
- **[Deployment Guide](docs/en/tutorials/deployment.md)** - Multi-environment deployment solutions | [中文](docs/zh/tutorials/deployment.md)

### 🔬 Research Materials
- **[Technical Whitepaper](docs/en/research/whitepaper.md)** - EasyNet theoretical foundation | [中文](docs/zh/research/whitepaper.md)
- **[Research Proposal](docs/en/research/research-proposal.md)** - Academic research plan | [中文](docs/zh/research/research-proposal.md)
- **[Project Pitch](docs/en/research/pitch.md)** - Business plan overview | [中文](docs/zh/research/pitch.md)

---

## 🌟 Three Major Breakthroughs of EasyNet

### **1. 🔒 Privacy-First Architecture**
```python
@node.register
def medical_diagnosis(scan_data):
    # Medical data never leaves your HIPAA-compliant device
    # But diagnostic services can be securely accessed globally
    return your_private_ai_model.diagnose(scan_data)
```

### **2. 💰 Economic Model Reconstruction**
- **Traditional Cloud Services**: Pay-per-use, costs increase exponentially with scale
- **EasyNet Model**: Contribute computing power to earn credits, use credits to call others' computing power
- **Gateway Cost**: $5/month vs traditional cloud $200+/million calls

### **3. 🚀 Consumer Devices Participating in Global AI**
```python
# Your gaming PC can provide AI inference services globally
@node.register
def image_generation(prompt):
    return your_stable_diffusion.generate(prompt)

# Your MacBook can participate in distributed training
@node.register  
def gradient_computation(batch_data):
    return your_local_model.compute_gradients(batch_data)
```

---

## 🎯 Future Vision: Torchrun for the World

### **Phase 1 (Current): Private Function Network**
- ✅ Peer-to-peer function calls
- ✅ Privacy-preserving computing
- ✅ Zero cold-start latency

### **Phase 2 (In Development): Computing Resource Pool**
- 🔄 Automatic load balancing
- 🔄 Computing contribution credit system
- 🔄 Cross-node task orchestration

### **Phase 3 (Planned): Intelligent Scheduling Network**
- 📋 Automatic task decomposition
- 📋 Optimal resource matching
- 📋 Fault tolerance and recovery mechanisms

### **Phase 4 (Vision): Global Computing Operating System**
```bash
# Future usage paradigm
$ easynet run --task "train_llm" --data "my_dataset" --nodes 1000
# Automatically schedule 1000 global nodes to collaboratively train your model
```

---

## 🔬 Technical Architecture: Decentralization + Edge Computing

### **Network Topology**
```
🌍 Global clients
    ↓
☁️ Lightweight gateway cluster (routing only, no computing)
    ↓
💻 Personal computing nodes (actual execution)
    ↓
🔗 Peer-to-peer collaboration network
```

### **Core Technology Stack**
- **Communication Protocol**: gRPC + Protocol Buffers
- **Secure Transport**: End-to-end encryption
- **Load Balancing**: Intelligent resource awareness
- **Fault Tolerance**: Automatic retry and recovery

---

## 🌊 Join the Computing Revolution

### **🔥 Why EasyNet Will Change Everything**

**Limitations of Traditional Models**:
- 💸 Cloud service costs grow exponentially with scale
- 🔒 Data must be uploaded to third-party servers
- ⚡ Cold starts and network latency limit performance
- 🏢 Locked into major cloud service providers

**EasyNet's Breakthroughs**:
- 💰 **Computing Sharing Economy**: Contribute idle resources, gain global computing power
- 🔐 **Privacy by Design**: Data never leaves your device
- 🚀 **Edge-First**: Zero latency, optimal performance
- 🌐 **Decentralized**: No single points of failure, no vendor lock-in

### **🎯 Our Mission**

> **Redefining the future of computing**: From a few cloud providers monopolizing computing power to every device being part of the computing network.

### **🚀 Join Now**

```bash
# Become an early node in EasyNet
pip install easyremote

# Contribute your computing power
python -c "
from easyremote import ComputeNode
node = ComputeNode('demo.easynet.io:8080')
@node.register
def hello_world(): return 'Hello from my device!'
node.serve()
"
```

---

## 🏗️ Developer Ecosystem

| Role | Contribution | Benefits |
|------|-------------|----------|
| **Computing Providers** | Idle GPU/CPU time | Computing credits/token rewards |
| **Application Developers** | Innovative algorithms and applications | Global computing resource access |
| **Gateway Operators** | Network infrastructure | Routing fee sharing |
| **Ecosystem Builders** | Tools and documentation | Community governance rights |

---

## 📞 Join the Community

* **🎯 Technical Discussions**: [GitHub Issues](https://github.com/Qingbolan/EasyCompute/issues)
* **💬 Community Chat**: [GitHub Discussions](https://github.com/Qingbolan/EasyCompute/discussions)
* **📧 Business Collaboration**: [silan.hu@u.nus.edu](mailto:silan.hu@u.nus.edu)
* **👨‍💻 Project Founder**: [Silan Hu](https://github.com/Qingbolan) - NUS PhD Candidate

---

<div align="center">

## 🌟 "The future of software isn't deployed on the cloud, but runs on your system + EasyNet"

**🚀 Ready to join the computing revolution?**

```bash
pip install easyremote
```

**Don't just see it as a distributed function tool — it's a prototype running on old-world tracks but heading towards a new-world destination.**

*⭐ If you believe in this new worldview, please give us a star!*

</div> 