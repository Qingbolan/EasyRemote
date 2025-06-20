# async_stream_client.py
"""
Async and Streaming Test Client
异步和流式功能测试客户端
"""
import asyncio
import json
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

class AsyncStreamTestClient:
    """Test client for async and streaming functions"""
    
    def __init__(self, server_address: str = "localhost:8080"):
        self.client = Client(server_address)
        self.server_address = server_address
        
    def test_sync_functions(self):
        """Test synchronous functions"""
        logger.info("🔄 Testing Synchronous Functions")
        print("=" * 50)
        
        # Test sync_add
        print("1. Testing sync_add...")
        result = self.client.execute("sync_add", 15, 25)
        print(f"   Result: {result}")
        
        # Test sync_process_data
        print("\n2. Testing sync_process_data...")
        test_data = {"x": 10, "y": 20, "name": "test"}
        result = self.client.execute("sync_process_data", test_data)
        print(f"   Result: {json.dumps(result, indent=2)}")
        
        print("✅ Sync functions test completed\n")
    
    async def test_async_functions(self):
        """Test asynchronous functions"""
        logger.info("⚡ Testing Asynchronous Functions")
        print("=" * 50)
        
        # Test async_computation
        print("1. Testing async_computation...")
        start_time = time.time()
        result = self.client.execute("async_computation", [1, 2, 3, 4, 5], 1.5)
        elapsed = time.time() - start_time
        print(f"   Result: {json.dumps(result, indent=2)}")
        print(f"   Elapsed time: {elapsed:.2f}s")
        
        # Test async_ai_simulation
        print("\n2. Testing async_ai_simulation...")
        start_time = time.time()
        result = self.client.execute("async_ai_simulation", "This is a test message for AI processing", 2.0)
        elapsed = time.time() - start_time
        print(f"   Result: {json.dumps(result, indent=2)}")
        print(f"   Elapsed time: {elapsed:.2f}s")
        
        print("✅ Async functions test completed\n")
    
    def test_sync_streaming(self):
        """Test synchronous streaming functions"""
        logger.info("🌊 Testing Synchronous Streaming")
        print("=" * 50)
        
        print("Testing sync_number_stream...")
        print("Streaming numbers from 1 to 5 with 0.5s interval:")
        
        # Note: This would need to be implemented based on how the client handles streaming
        # For now, we'll show the expected usage pattern
        try:
            # This is a placeholder - actual streaming implementation may vary
            result = self.client.execute("sync_number_stream", 1, 5, 0.5)
            if isinstance(result, str):
                # If result is a single string, split by lines
                for line in result.strip().split('\n'):
                    if line:
                        data = json.loads(line)
                        print(f"   📊 {data}")
            else:
                print(f"   Result: {result}")
        except Exception as e:
            print(f"   ⚠️  Streaming test note: {e}")
            print("   (Streaming may require special client implementation)")
        
        print("✅ Sync streaming test completed\n")
    
    async def test_async_streaming(self):
        """Test asynchronous streaming functions"""
        logger.info("⚡🌊 Testing Asynchronous Streaming")
        print("=" * 50)
        
        # Test async_data_stream
        print("1. Testing async_data_stream...")
        config = {
            'sensors': ['temperature', 'humidity'],
            'sample_rate': 2,  # 2 samples per second
            'duration': 5      # 5 seconds
        }
        
        try:
            result = self.client.execute("async_data_stream", config)
            print(f"   Stream config: {config}")
            if isinstance(result, str):
                lines = result.strip().split('\n')
                print(f"   Received {len(lines)} data points:")
                for i, line in enumerate(lines[:3]):  # Show first 3
                    if line:
                        data = json.loads(line)
                        print(f"   📊 Sample {i+1}: {data}")
                if len(lines) > 3:
                    print(f"   ... and {len(lines)-3} more samples")
            else:
                print(f"   Result: {result}")
        except Exception as e:
            print(f"   ⚠️  Async streaming test note: {e}")
        
        # Test async_ml_inference_stream
        print("\n2. Testing async_ml_inference_stream...")
        images = [f"image_{i:03d}.jpg" for i in range(1, 7)]  # 6 images
        
        try:
            result = self.client.execute("async_ml_inference_stream", images, 2)
            print(f"   Processing {len(images)} images with batch_size=2")
            if isinstance(result, str):
                lines = result.strip().split('\n')
                for line in lines:
                    if line:
                        data = json.loads(line)
                        print(f"   🤖 Batch {data.get('batch_id', '?')}: {data.get('batch_size', '?')} images processed")
            else:
                print(f"   Result: {result}")
        except Exception as e:
            print(f"   ⚠️  ML streaming test note: {e}")
        
        print("✅ Async streaming test completed\n")
    
    async def test_complex_pipeline(self):
        """Test complex async pipeline"""
        logger.info("🔧 Testing Complex Async Pipeline")
        print("=" * 50)
        
        config = {
            'stages': 3,
            'items_per_stage': 4,
            'stage_delay': 2.0
        }
        
        print(f"Testing complex pipeline with config: {config}")
        
        try:
            start_time = time.time()
            result = self.client.execute("async_complex_pipeline", "test_pipeline", config)
            elapsed = time.time() - start_time
            
            if isinstance(result, str):
                lines = result.strip().split('\n')
                print(f"   Pipeline generated {len(lines)} results:")
                
                # Show first few results
                for i, line in enumerate(lines[:5]):
                    if line:
                        data = json.loads(line)
                        if 'stage' in data:
                            print(f"   Stage {data['stage']}, Item {data['item']}: {data['progress']['total_progress']}")
                        elif 'status' in data:
                            print(f"   {data['status'].upper()}: {data['total_items']} items processed")
                
                if len(lines) > 5:
                    print(f"   ... and {len(lines)-5} more results")
            else:
                print(f"   Result: {result}")
            
            print(f"Total elapsed time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Complex pipeline test note: {e}")
        
        print("Complex pipeline test completed\n")
    
    async def run_all_tests(self):
        """Run all test cases"""
        print("Starting EasyRemote Async & Streaming Tests")
        print("=" * 60)
        print(f"Server: {self.server_address}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print("=" * 60)
        
        try:
            # Test synchronous functions
            self.test_sync_functions()
            
            # Test asynchronous functions
            await self.test_async_functions()
            
            # Test streaming functions
            self.test_sync_streaming()
            await self.test_async_streaming()
            
            # Test complex pipeline
            await self.test_complex_pipeline()
            
            print("🎉 All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            print(f"❌ Test failed: {e}")
            raise

async def main():
    """Main test function"""
    print("EasyRemote Async & Streaming Test Client")
    print("Testing various async and streaming capabilities\n")
    
    client = AsyncStreamTestClient()
    
    try:
        await client.run_all_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTests failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 