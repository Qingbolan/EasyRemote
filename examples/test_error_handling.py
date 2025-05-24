#!/usr/bin/env python3
"""
测试改进后的错误处理和重连机制

这个示例展示：
1. 优雅的断连处理（不再打印错误堆栈）
2. 事件循环关闭时的安全清理
3. 自动重连功能
"""

import asyncio
import time
from easyremote import ComputeNode

def main():
    print("🧪 测试改进后的错误处理和重连机制")
    print("=" * 50)
    
    # 创建节点
    node = ComputeNode(
        vps_address="localhost:8080",
        node_id="error-handling-test",
        reconnect_interval=2,
        max_retry_attempts=5,
        heartbeat_interval=3,
        heartbeat_timeout=10,
        health_check_interval=15
    )
    
    # 注册一个测试函数
    @node.register
    def test_function(message: str = "Hello") -> str:
        return f"Response: {message}"
    
    print(f"✅ 已注册测试函数")
    print(f"🔗 节点ID: {node.node_id}")
    print(f"📡 连接地址: {node.vps_address}")
    
    print(f"\n💡 测试说明:")
    print("   1. 此程序会尝试连接到 localhost:8080")
    print("   2. 如果服务器未运行，会看到优雅的重连尝试")
    print("   3. 不会再看到 'Event loop is closed' 错误")
    print("   4. 连接失败的消息现在是 DEBUG 级别，更简洁")
    print("   5. 按 Ctrl+C 停止测试")
    
    print(f"\n🎯 开始连接测试...")
    print("=" * 50)
    
    try:
        # 启动节点服务
        node.serve(blocking=True)
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在停止节点...")
        node.stop()
        print("✅ 节点已安全停止")
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        node.stop()

if __name__ == "__main__":
    main() 