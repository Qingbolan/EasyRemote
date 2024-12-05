# EasyCompute

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/easy-compute.svg)](https://pypi.org/project/easy-compute/)

EasyCompute is a powerful and flexible distributed computing framework that enables seamless integration between private network computing resources and public network services. It's designed to help developers easily utilize private network computing power through public network endpoints.

## 🌟 Key Features

- **Simple API**: Intuitive Python API with decorator-based handlers
- **Flexible Communication**: Support for various data transmission modes including:
  - Simple request-response
  - Stream processing
  - Bi-directional streaming
  - Multi-modal data handling
- **Robust Architecture**: Built on Kitex RPC framework with high performance and stability
- **Smart Load Balancing**: Automatic distribution of computing tasks
- **Auto Recovery**: Handles network interruptions and reconnections
- **Resource Management**: Efficient management of computing resources and connections

## 🚀 Quick Start

### Installation

```bash
pip install easy-compute
```

### Basic Usage

1. Set up a computation node (private network):

```python
from easy_compute import ComputeNode

node = ComputeNode(
    vps_address="your-vps-ip:8080",
    node_id="compute-1"
)

@node.register
async def process_image(image_data: bytes) -> bytes:
    # Your image processing logic
    return processed_image

if __name__ == "__main__":
    node.start()
```

2. Start VPS server (public network):

```python
from easy_compute import VPSServer

server = VPSServer(port=8080)
server.start()
```

3. Client usage:

```python
from easy_compute import Client

client = Client("your-vps-ip:8080")
result = await client.call("process_image", image_data)
```

## 🎯 Advanced Features

### Stream Processing

```python
@node.register(mode="stream")
async def process_video(video_data: bytes) -> AsyncIterator[bytes]:
    for frame in extract_frames(video_data):
        processed = await process_frame(frame)
        yield processed

# Client usage
async for frame in client.stream_call("process_video", video_data):
    display_frame(frame)
```

### Bi-directional Streaming

```python
@node.register(mode="bidirectional")
async def chat_session(
    messages: AsyncIterator[str]
) -> AsyncIterator[str]:
    async for message in messages:
        response = await generate_response(message)
        yield response

# Client usage
async with client.create_session("chat_session") as chat:
    async for response in chat.stream("Hello"):
        print(response)
```

### Multi-modal Processing

```python
from easy_compute.types import MultiModalOutput

@node.register(mode="multimodal")
async def process_document(doc: bytes) -> MultiModalOutput:
    return MultiModalOutput(
        text=extract_text(doc),
        images=extract_images(doc),
        audio=text_to_speech(extract_text(doc))
    )
```

## 🔧 Configuration

### Node Configuration

```python
from easy_compute import NodeConfig

config = NodeConfig(
    vps_address="x.x.x.x:8080",
    node_id="node-1",
    max_workers=4,
    reconnect_interval=5,
    ssl_enabled=True,
    compression_enabled=True
)

node = ComputeNode(config)
```

### VPS Server Configuration

```python
from easy_compute import ServerConfig

config = ServerConfig(
    port=8080,
    max_connections=1000,
    load_balance_strategy="least_load",
    ssl_config=SSLConfig(cert_file="cert.pem", key_file="key.pem")
)

server = VPSServer(config)
```

## 📊 Architecture

### Core Components

- **ComputeNode**: Manages private network computing resources
- **VPSServer**: Handles public network requests and load balancing
- **Client**: Provides easy-to-use API for service consumers
- **ConnectionManager**: Maintains stable connections between nodes

### Data Flow

1. ComputeNode establishes connection with VPSServer
2. Client sends request to VPSServer
3. VPSServer routes request to appropriate ComputeNode
4. ComputeNode processes request and returns result
5. Result is forwarded back to Client

## 🔐 Security

- TLS encryption for all communications
- Node authentication and authorization
- Resource usage limits and rate limiting
- Secure task isolation

## 📈 Performance

- Connection pooling
- Smart load balancing
- Data compression
- Zero-copy data transfer when possible

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test category
pytest tests/test_streaming.py
```

## 📚 API Documentation

Detailed API documentation is available at [docs.easycompute.io](https://docs.easycompute.io).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository

```bash
git clone https://github.com/Qingbolan/easy-compute.git
cd easy-compute
```

2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install development dependencies

```bash
pip install -e ".[dev]"
```

## 📝 Project Structure

```
easy-compute/
├── easy_compute/
│   ├── __init__.py
│   ├── node.py          # ComputeNode implementation
│   ├── server.py        # VPSServer implementation
│   ├── client.py        # Client implementation
│   ├── protocol/        # Protocol definitions
│   ├── transport/       # Transport layer
│   └── utils/           # Utility functions
├── tests/               # Test cases
├── examples/            # Usage examples
├── docs/               # Documentation
└── setup.py           # Package configuration
```

## 🗺️ Roadmap

- [X] Basic RPC functionality
- [X] Stream processing support
- [X] Bi-directional streaming
- [X] Load balancing
- [ ] Metrics collection
- [ ] Web dashboard
- [ ] Kubernetes integration
- [ ] Plugin system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Qingbolan**

- GitHub: [@Qingbolan](https://github.com/Qingbolan)
- Email: silan.hu@u.nus.edu

## 🙏 Acknowledgments

- Kitex Team for the amazing RPC framework
- All contributors who have helped with the project
- The Python community for inspiration and support
