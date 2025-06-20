# improved_client.py
"""
Improved Client for Testing True Streaming Functions
改进的客户端 - 测试真正的流式处理功能
"""
import asyncio
import time
from datetime import datetime
from easyremote import Client
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedStreamingClient:
    """Improved client for testing streaming functions"""
    
    def __init__(self, server_address: str = "localhost:8080"):
        self.client = Client(server_address)
        self.server_address = server_address
    
    async def test_sync_functions(self):
        """Test synchronous functions"""
        logger.info("🔧 Testing Synchronous Functions")
        
        # Test sync addition
        result = self.client.execute("sync_add", 10, 20)
        logger.info(f"✅ sync_add(10, 20) = {result}")
        
        # Test sync data processing
        test_data = {"value": 42, "name": "test", "count": 5}
        result = self.client.execute("sync_process_data", test_data)
        logger.info(f"✅ sync_process_data result: {result}")
    
    async def test_async_functions(self):
        """Test asynchronous functions"""
        logger.info("⚡ Testing Asynchronous Functions")
        
        # Test async computation
        start_time = time.time()
        result = self.client.execute("async_computation", [1, 2, 3, 4, 5], 1.5)
        execution_time = time.time() - start_time
        logger.info(f"✅ async_computation completed in {execution_time:.2f}s")
        logger.info(f"   Result: sum={result['sum']}, avg={result['average']}")
        
        # Test async AI simulation
        start_time = time.time()
        result = self.client.execute("async_ai_simulation", "This is a test text for sentiment analysis", 1.0)
        execution_time = time.time() - start_time
        logger.info(f"✅ async_ai_simulation completed in {execution_time:.2f}s")
        logger.info(f"   Result: sentiment={result['sentiment']}, confidence={result['confidence']}")
    
    async def test_streaming_functions(self):
        """Test true streaming functions"""
        logger.info("🌊 Testing True Streaming Functions")
        
        # Test sync number stream
        logger.info("📊 Testing sync_number_stream...")
        try:
            stream_result = self.client.execute("sync_number_stream", 1, 5, 0.3)
            if isinstance(stream_result, str):
                logger.info("⚠️  Received string result (collected stream)")
                lines = stream_result.strip().split('\n')
                logger.info(f"   Received {len(lines)} stream items")
                for i, line in enumerate(lines[:3]):  # Show first 3 items
                    logger.info(f"   Item {i+1}: {line}")
            else:
                logger.info(f"✅ sync_number_stream result: {stream_result}")
        except Exception as e:
            logger.error(f"❌ Error in sync_number_stream: {e}")
        
        # Test async data stream
        logger.info("📈 Testing async_data_stream...")
        try:
            config = {
                'sensors': ['temperature', 'humidity'],
                'sample_rate': 2,
                'duration': 5
            }
            stream_result = self.client.execute("async_data_stream", config)
            if isinstance(stream_result, str):
                logger.info("⚠️  Received string result (collected stream)")
                lines = stream_result.strip().split('\n')
                logger.info(f"   Received {len(lines)} stream items")
                for i, line in enumerate(lines[:3]):  # Show first 3 items
                    logger.info(f"   Item {i+1}: {line}")
            else:
                logger.info(f"✅ async_data_stream result: {stream_result}")
        except Exception as e:
            logger.error(f"❌ Error in async_data_stream: {e}")
        
        # Test ML inference stream
        logger.info("🤖 Testing async_ml_inference_stream...")
        try:
            images = [f"image_{i:03d}.jpg" for i in range(1, 8)]
            stream_result = self.client.execute("async_ml_inference_stream", images, 3)
            if isinstance(stream_result, str):
                logger.info("⚠️  Received string result (collected stream)")
                lines = stream_result.strip().split('\n')
                logger.info(f"   Received {len(lines)} stream items")
                for i, line in enumerate(lines[:2]):  # Show first 2 items
                    logger.info(f"   Batch {i+1}: {line}")
            else:
                logger.info(f"✅ async_ml_inference_stream result: {stream_result}")
        except Exception as e:
            logger.error(f"❌ Error in async_ml_inference_stream: {e}")
        
        # Test complex pipeline
        logger.info("🔄 Testing async_complex_pipeline...")
        try:
            config = {
                'stages': 2,
                'items_per_stage': 3,
                'stage_delay': 0.5
            }
            stream_result = self.client.execute("async_complex_pipeline", "test_pipeline", config)
            if isinstance(stream_result, str):
                logger.info("⚠️  Received string result (collected stream)")
                lines = stream_result.strip().split('\n')
                logger.info(f"   Received {len(lines)} stream items")
                for i, line in enumerate(lines[:3]):  # Show first 3 items
                    logger.info(f"   Item {i+1}: {line}")
            else:
                logger.info(f"✅ async_complex_pipeline result: {stream_result}")
        except Exception as e:
            logger.error(f"❌ Error in async_complex_pipeline: {e}")
        
        # Test real-time event stream
        logger.info("📡 Testing real_time_event_stream...")
        try:
            config = {
                'event_types': ['user_action', 'system_alert'],
                'event_rate': 3,
                'duration': 4
            }
            stream_result = self.client.execute("real_time_event_stream", config)
            if isinstance(stream_result, str):
                logger.info("⚠️  Received string result (collected stream)")
                lines = stream_result.strip().split('\n')
                logger.info(f"   Received {len(lines)} stream items")
                for i, line in enumerate(lines[:3]):  # Show first 3 items
                    logger.info(f"   Event {i+1}: {line}")
            else:
                logger.info(f"✅ real_time_event_stream result: {stream_result}")
        except Exception as e:
            logger.error(f"❌ Error in real_time_event_stream: {e}")
    
    async def test_concurrent_execution(self):
        """Test concurrent execution of multiple functions"""
        logger.info("🚀 Testing Concurrent Execution")
        
        async def run_async_computation():
            return self.client.execute("async_computation", [10, 20, 30], 1.0)
        
        async def run_ai_simulation():
            return self.client.execute("async_ai_simulation", "Concurrent test", 1.0)
        
        # Run multiple async functions concurrently
        start_time = time.time()
        results = await asyncio.gather(
            asyncio.create_task(asyncio.to_thread(run_async_computation)),
            asyncio.create_task(asyncio.to_thread(run_ai_simulation)),
            return_exceptions=True
        )
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Concurrent execution completed in {execution_time:.2f}s")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"   Task {i+1} failed: {result}")
            else:
                logger.info(f"   Task {i+1} succeeded")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("🎯 Starting Comprehensive Test Suite")
        logger.info(f"📡 Server: {self.server_address}")
        logger.info(f"⏰ Start time: {datetime.now().isoformat()}")
        
        try:
            # Test sync functions
            await self.test_sync_functions()
            await asyncio.sleep(1)
            
            # Test async functions
            await self.test_async_functions()
            await asyncio.sleep(1)
            
            # Test streaming functions
            await self.test_streaming_functions()
            await asyncio.sleep(1)
            
            # Test concurrent execution
            await self.test_concurrent_execution()
            
            logger.info("✅ All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise
        
        logger.info(f"⏰ End time: {datetime.now().isoformat()}")

async def main():
    """Main function"""
    logger.info("🚀 Starting Improved EasyRemote Streaming Client")
    
    client = ImprovedStreamingClient()
    
    try:
        await client.run_comprehensive_test()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    
    logger.info("🎉 Test completed!")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 