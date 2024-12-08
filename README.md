
# EasyRemote

<div align="center">

![EasyRemote Logo](https://raw.githubusercontent.com/Qingbolan/EasyRemote/master/docs/logo.png)

[![PyPI version](https://badge.fury.io/py/easyremote.svg)](https://badge.fury.io/py/easyremote)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/easyremote)]()
[![Downloads](https://pepy.tech/badge/easyremote)]()

*A lightweight framework for hassle-free remote computing resource sharing.*

English | [中文](./README_zh.md)

</div>

## Why EasyRemote?

Are you tired of:

- Paying expensive cloud GPU fees for AI development?
- Struggling with complex deployment for demos?
- Looking for ways to share computing resources within your team?

EasyRemote lets you expose local computing resources (AI models, data processing functions) as remote services with just a few lines of code. All you need is a cheap VPS!

```python
# It's as simple as this:
from easyremote import remote

@remote
def run_model(input_data):
    return your_ai_model(input_data)  # Runs on your local GPU
```

## Features

- 🚀 **Super Simple**: Turn any function into a remote service with a single decorator
- 💰 **Cost-Effective**: Use your local GPU through an inexpensive VPS
- 🔒 **Private & Secure**: All computation stays on your local machine
- 🌐 **Flexible Deployment**: Perfect for demos, prototypes, and team collaboration

## Quick Start

### 1. Installation

```bash
pip install easyremote
```

### 2. Set Up VPS (Gateway)

```python
from easyremote import Server

server = Server(port=8080)
server.start()
```

### 3. Configure Local Node

```python
from easyremote import ComputeNode

# Connect to your VPS
node = ComputeNode("your-vps-ip:8080")

# Define your remote functions
@node.register
def process_data(data):
    return heavy_computation(data)  # Runs locally

# Start serving
node.serve()
```

### 4. Call Remote Functions

```python
# From anywhere with internet access
from easyremote import Client

client = Client("vps-ip:8080")
result = client.call("process_data", data=my_data)
```

## Advanced Usage

### Async Support

```python
@node.register(async_func=True)
async def async_process(data):
    result = await complex_async_operation(data)
    return result
```

### Streaming Results

```python
@node.register(stream=True)
def stream_results(data):
    for chunk in process_large_dataset(data):
        yield chunk
```

### Real-world Examples

Check out our [examples](./examples/) directory for:

- AI Model Serving
- Data Pipeline Processing
- Team Resource Sharing
- And more!

## Architecture

```
Client -> VPS (Gateway) -> Local Compute Node
                       -> Local Compute Node
                       -> Local Compute Node
```

## Performance

- Efficient binary protocol
- Support for large data transfer
- Automatic connection management

## Roadmap

- [ ] Multi-node clustering support
- [ ] Enhanced security features
- [ ] Web-based management UI
- [ ] More language SDKs
- [ ] Docker support

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md)

## License

[MIT License](LICENSE)

## Contact & Support

- Author: Silan Hu
- Email: silan.hu@u.nus.edu
- GitHub: [Qingbolan](https://github.com/Qingbolan)

## Acknowledgments

Special thanks to all contributors who have helped make EasyRemote better!

---

<div align="center">

*If you find EasyRemote useful, please consider giving it a star ⭐*

</div>
