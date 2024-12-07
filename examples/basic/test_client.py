# examples/basic/test_client.py
import requests

def test_basic_functions():
    # 测试加法
    response = requests.post(
        "http://127.0.0.1:8000/add",
        params={"a": 5, "b": 3}
    )
    print("Add result:", response.json())
    
    # 测试数据处理
    data = {"x": 10, "y": 20}
    response = requests.post(
        "http://127.0.0.1:8000/process",
        json=data
    )
    print("Process result:", response.json())
    
    # 测试流式生成
    response = requests.get(
        "http://127.0.0.1:8000/generate",
        params={"start": 0, "count": 5},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8')
            if data.startswith('data: '):
                print("Generated:", data[6:])
                

if __name__ == "__main__":
    test_basic_functions()