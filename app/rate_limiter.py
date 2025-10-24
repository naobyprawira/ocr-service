"""Accelerator Rate Limiter for controlling concurrent OCR operations.

Works with both CPU and GPU execution using PaddlePaddle.
"""

import asyncio
import time
from typing import Optional
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class AcceleratorRateLimiter:
    """Rate limiter for OCR operations with concurrent request limiting and queuing.
    
    Works with both CPU and GPU execution modes. Prevents resource exhaustion
    by limiting concurrent operations and queuing excess requests.
    """
    
    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 10,
        timeout: float = 60.0
    ):
        """
        Initialize accelerator rate limiter.
        
        Args:
            max_concurrent: Maximum number of concurrent OCR operations
            max_queue_size: Maximum number of queued requests
            timeout: Maximum wait time in seconds for acquiring processing access
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_requests = 0
        self._queued_requests = 0
        self._total_processed = 0
        self._total_rejected = 0
        self._total_timeouts = 0
        self._lock = asyncio.Lock()
        
        logger.info(
            f"Accelerator rate limiter initialized: max_concurrent={max_concurrent}, "
            f"max_queue_size={max_queue_size}, timeout={timeout}s"
        )
        
    @asynccontextmanager
    async def acquire(self):
        """
        Async context manager to acquire processing access.
        
        Raises:
            asyncio.TimeoutError: If timeout is reached
            RuntimeError: If queue is full
        """
        # Check queue capacity
        async with self._lock:
            if self._queued_requests >= self.max_queue_size:
                self._total_rejected += 1
                logger.warning(
                    f"Request queue is full. Rejecting request. "
                    f"Active: {self._active_requests}, Queued: {self._queued_requests}"
                )
                raise RuntimeError(
                    f"Request queue is full ({self.max_queue_size} requests waiting). "
                    f"Please try again later."
                )
            self._queued_requests += 1
        
        start_time = time.time()
        acquired = False
        
        try:
            logger.info(
                f"Processing request submitted. Active: {self._active_requests}, "
                f"Queued: {self._queued_requests}"
            )
            
            # Wait for processing access with timeout
            try:
                async with asyncio.timeout(self.timeout):
                    await self._semaphore.acquire()
                    acquired = True
            except asyncio.TimeoutError:
                async with self._lock:
                    self._total_timeouts += 1
                logger.error(
                    f"Processing request timeout after {self.timeout}s. "
                    f"Active: {self._active_requests}, Queued: {self._queued_requests}"
                )
                raise
            
            wait_time = time.time() - start_time
            
            async with self._lock:
                self._active_requests += 1
                self._queued_requests -= 1
            
            logger.info(
                f"Processing started after {wait_time:.2f}s. "
                f"Active: {self._active_requests}, Remaining queue: {self._queued_requests}"
            )
            
            yield
            
            # Mark as successfully processed
            async with self._lock:
                self._total_processed += 1
            
        finally:
            # Clean up
            if acquired:
                async with self._lock:
                    self._active_requests -= 1
                self._semaphore.release()
                
                processing_time = time.time() - start_time
                logger.info(
                    f"Processing completed after {processing_time:.2f}s. "
                    f"Active: {self._active_requests}, Queued: {self._queued_requests}"
                )
            else:
                # Request was queued but never acquired
                async with self._lock:
                    self._queued_requests -= 1
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "active_requests": self._active_requests,
            "queued_requests": self._queued_requests,
            "available_slots": self.max_concurrent - self._active_requests,
            "total_processed": self._total_processed,
            "total_rejected": self._total_rejected,
            "total_timeouts": self._total_timeouts
        }
    
    async def wait_until_available(self, check_interval: float = 0.5) -> None:
        """
        Wait until processing resources are available (non-blocking check).
        
        Args:
            check_interval: How often to check for availability in seconds
        """
        while self._active_requests >= self.max_concurrent:
            await asyncio.sleep(check_interval)


# Global rate limiter instance
accelerator_limiter: Optional[AcceleratorRateLimiter] = None


def init_rate_limiter(
    max_concurrent: int = 3,
    max_queue_size: int = 10,
    timeout: float = 60.0
) -> AcceleratorRateLimiter:
    """
    Initialize the global accelerator rate limiter.
    
    Args:
        max_concurrent: Maximum number of concurrent OCR operations
        max_queue_size: Maximum number of queued requests
        timeout: Maximum wait time in seconds for acquiring processing access
        
    Returns:
        The initialized AcceleratorRateLimiter instance
    """
    global accelerator_limiter
    accelerator_limiter = AcceleratorRateLimiter(
        max_concurrent=max_concurrent,
        max_queue_size=max_queue_size,
        timeout=timeout
    )
    return accelerator_limiter


def get_rate_limiter() -> AcceleratorRateLimiter:
    """
    Get the global accelerator rate limiter instance.
    
    Raises:
        RuntimeError: If rate limiter has not been initialized
    """
    if accelerator_limiter is None:
        raise RuntimeError("Accelerator rate limiter not initialized. Call init_rate_limiter() first.")
    return accelerator_limiter
