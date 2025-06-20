# async_stream_server.py
"""
Async and Streaming Test Server
异步和流式功能测试服务器
"""
from easyremote import Server
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the gateway server for async and streaming tests"""
    logger.info("🚀 Starting EasyRemote Server for Async & Streaming Tests")
    logger.info("📡 Server will listen on port 8080")
    logger.info("⚡ Ready to handle async functions and streaming data")
    
    # Create and start the gateway server
    server = Server(
        port=8080,
        max_queue_size=1000,
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        logger.info("✅ Server stopped gracefully")

if __name__ == "__main__":
    main() 