#!/usr/bin/env python3
"""
EasyRemote并发流式任务快速启动脚本
"""
import sys
import subprocess
import os

def check_dependencies():
    """检查依赖项"""
    required_packages = ['rich', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 EasyRemote Concurrent Streaming Demo")
    print("=" * 50)
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 切换到正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("✅ Dependencies checked")
    print("🎯 Starting demo...")
    print()
    
    try:
        # 运行客户端演示
        subprocess.run([sys.executable, "client_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 