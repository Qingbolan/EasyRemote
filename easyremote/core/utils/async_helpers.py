#!/usr/bin/env python3
"""
Async helpers for handling event loops safely
"""
import asyncio
import threading
from typing import Coroutine, Any, Optional
from .logger import ModernLogger

class AsyncHelpers(ModernLogger):
    
    def __init__(self):
        super().__init__(name="AsyncHelpers")
    
    def is_running_in_event_loop(self) -> bool:
        """Check if we're currently running inside an event loop."""
        try:
            loop = asyncio.get_running_loop()
            return loop is not None
        except RuntimeError:
            return False

    def get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """获取现有事件循环或创建新的事件循环"""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def run_async_safely(self, coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """
        安全地运行异步协程，自动处理事件循环冲突
        
        Args:
            coro: 要运行的协程
            timeout: 超时时间（秒）
            
        Returns:
            协程的返回值
            
        Raises:
            asyncio.TimeoutError: 如果超时
            Exception: 协程执行中的异常
        """
        if self.is_running_in_event_loop():
            # 如果已经在事件循环中，不能使用 run_until_complete
            # 需要在新线程中运行
            self.debug("Running coroutine in new thread (already in event loop)")
            
            def run_in_thread():
                # 在新线程中创建新的事件循环
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        else:
            # 可以直接运行
            self.debug("Running coroutine directly")
            loop = self.get_or_create_event_loop()
            try:
                return loop.run_until_complete(coro)
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # 如果还是有冲突，使用线程方式
                    self.warning("Event loop conflict detected, falling back to thread execution")
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(coro)
                        finally:
                            new_loop.close()
                    
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result(timeout=timeout)
                else:
                    raise

    def run_async_in_background(self, coro: Coroutine, daemon: bool = True) -> threading.Thread:
        """
        在后台线程中运行异步协程
        
        Args:
            coro: 要运行的协程
            daemon: 是否为守护线程
            
        Returns:
            运行协程的线程对象
        """
        def run_coro():
            # 在新线程中创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                self.error(f"Error in background coroutine: {e}", exc_info=True)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_coro, daemon=daemon)
        thread.start()
        return thread

    async def run_in_executor(self, func, *args, **kwargs):
        """在线程池执行器中运行同步函数"""
        loop = asyncio.get_event_loop()
        import functools
        wrapped_func = functools.partial(func, **kwargs)
        return await loop.run_in_executor(None, wrapped_func, *args)


class AsyncContextManager(ModernLogger):
    """异步上下文管理器基类"""
    def __init__(self):
        super().__init__(name="AsyncContextManager")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class SafeEventLoop(ModernLogger):
    """安全的事件循环管理器"""
    
    def __init__(self):
        super().__init__(name="SafeEventLoop")
        self._loop = None
        self._thread = None
        self._running = False
    
    def start(self, daemon: bool = True):
        """启动事件循环线程"""
        if self._running:
            return
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()
        
        self._thread = threading.Thread(target=run_loop, daemon=daemon)
        self._thread.start()
        self._running = True
        
        # 等待事件循环启动
        import time
        timeout = 5.0
        start_time = time.time()
        while not self._loop and time.time() - start_time < timeout:
            time.sleep(0.01)
        
        if not self._loop:
            self.error("Failed to start event loop")
            raise RuntimeError("Failed to start event loop")

    def stop(self):
        """停止事件循环"""
        if not self._running:
            return
            
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
    
    def run_coroutine(self, coro: Coroutine, timeout: Optional[float] = None):
        """在事件循环中运行协程"""
        if not self._running or not self._loop:
            raise RuntimeError("Event loop not running")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
    
    @property
    def is_running(self) -> bool:
        return self._running and self._loop and not self._loop.is_closed()


# Convenience functions for global use
def is_running_in_event_loop() -> bool:
    """Check if we're currently running inside an event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False 