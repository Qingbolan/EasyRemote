# vps_server_example.py

from flask import Flask, request, jsonify
from easyremote import Server, remote

app = Flask(__name__)
server = Server(port=8080)  # 初始化服务器，监听 8080 端口

# 注册一个远程同步函数
@remote()
def process_data(data: dict) -> dict:
    """这实际上会在计算节点上执行"""
    pass  # 该函数的实现由计算节点处理

# 注册一个远程异步函数
@remote(async_func=True)
async def async_process_data(data: dict) -> dict:
    """这实际上会在计算节点上执行"""
    pass  # 该函数的实现由计算节点处理

# 注册一个远程同步生成器函数（流式）
@remote(stream=True)
def stream_process(data: list) -> list:
    """这实际上会在计算节点上执行"""
    pass  # 该函数的实现由计算节点处理

# 注册一个远程异步生成器函数（流式异步）
@remote(stream=True, async_func=True)
async def async_stream_process(data: list) -> list:
    """这实际上会在计算节点上执行"""
    pass  # 该函数的实现由计算节点处理

@app.route('/process', methods=['POST'])
def process():
    """
    端点调用远程同步函数。
    期望 JSON 负载包含 'value' 键。
    """
    data = request.json
    result = process_data(data)
    return jsonify(result)

@app.route('/async_process', methods=['POST'])
async def async_proc():
    """
    端点调用远程异步函数。
    期望 JSON 负载包含 'value' 键。
    """
    data = request.json
    result = await async_process_data(data)
    return jsonify(result)

@app.route('/stream_process', methods=['POST'])
def stream_proc():
    """
    端点调用远程同步生成器函数（流式）。
    期望 JSON 负载包含 'data' 键（列表）。
    返回处理后的数据块列表。
    """
    data = request.json.get('data', [])
    result_gen = stream_process(data)
    # 将生成器转换为列表
    return jsonify(list(result_gen))

@app.route('/async_stream_process', methods=['POST'])
async def async_stream_proc():
    """
    端点调用远程异步生成器函数（流式异步）。
    期望 JSON 负载包含 'data' 键（列表）。
    返回处理后的数据块列表。
    """
    data = request.json.get('data', [])
    result_gen = async_stream_process(data)
    # 异步收集生成器中的所有数据块
    result = []
    async for item in result_gen:
        result.append(item)
    return jsonify(result)

if __name__ == '__main__':
    # 启动服务器在后台运行
    server.start_background()
    # 运行 Flask 应用
    app.run(host='0.0.0.0', port=5000)
