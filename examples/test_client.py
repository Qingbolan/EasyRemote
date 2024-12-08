# test_client.py
import requests
import json
import os

def test_basic_functions():
    # 测试 /add 端点
    try:
        response = requests.post(
            "http://127.0.0.1:8000/add",
            params={"a": 5, "b": 3}
        )
        response.raise_for_status()
        print("Add result:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"测试 /add 时出错: {e}")
    
    # 测试 /process 端点
    try:
        data = {"x": 10, "y": 1000}
        response = requests.post(
            "http://127.0.0.1:8000/process",
            json=data
        )
        response.raise_for_status()
        print("Process result:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"测试 /process 时出错: {e}")

def test_process_photo():
    """测试 /process_photo 端点，上传并处理本地的 test.png 照片。"""
    try:
        url = "http://127.0.0.1:8000/process_photo"
        file_path = "easyremote-logo.png"  # 确保当前目录下有 test.png 文件
        if not os.path.isfile(file_path):
            print(f"测试图片文件 {file_path} 未找到。请将 easyremote-logo.png 放在当前目录。")
            return
        with open(file_path, "rb") as image_file:
            files = {"file": (file_path, image_file, "image/png")}
            response = requests.post(url, files=files)
            response.raise_for_status()
            # 保存处理后的图片
            processed_image_path = "processed_" + file_path
            with open(processed_image_path, "wb") as f:
                f.write(response.content)
            print(f"处理后的照片已保存为 {processed_image_path}")
    except requests.exceptions.RequestException as e:
        print(f"测试 /process_photo 时出错: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")

if __name__ == "__main__":
    test_basic_functions()
    test_process_photo()
