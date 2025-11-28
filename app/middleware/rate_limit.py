"""
Jarvis Voice Assistant - Rate Limiting Middleware

Implements per-session and per-IP rate limiting.
"""

import time
from typing import Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.config import settings


@dataclass
class RateLimitEntry:
    """Tracks rate limit state."""
    count: int
    window_start: float


class RateLimiter:
    """
    Rate limiter with sliding window algorithm.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_session: int = 100
    ):
        self._rpm = requests_per_minute
        self._rps = requests_per_session
        self._ip_limits: Dict[str, RateLimitEntry] = defaultdict(
            lambda: RateLimitEntry(0, time.time())
        )
        self._session_limits: Dict[str, RateLimitEntry] = defaultdict(
            lambda: RateLimitEntry(0, time.time())
        )
        self._window_size = 60  # 1 minute window
    
    def check_ip(self, ip: str) -> Tuple[bool, int]:
        """
        Check rate limit for an IP address.
        
        Args:
            ip: The IP address
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        return self._check_limit(self._ip_limits, ip, self._rpm)
    
    def check_session(self, session_id: str) -> Tuple[bool, int]:
        """
        Check rate limit for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        return self._check_limit(self._session_limits, session_id, self._rps)
    
    def _check_limit(
        self,
        limits: Dict[str, RateLimitEntry],
        key: str,
        max_requests: int
    ) -> Tuple[bool, int]:
        """Check and update rate limit for a key."""
        now = time.time()
        entry = limits[key]
        
        # Reset window if expired
        if now - entry.window_start >= self._window_size:
            entry.count = 0
            entry.window_start = now
        
        # Check if allowed
        if entry.count >= max_requests:
            return False, 0
        
        # Increment and allow
        entry.count += 1
        remaining = max_requests - entry.count
        
        return True, remaining
    
    def get_retry_after(self, key: str, is_session: bool = False) -> int:
        """
        Get seconds until rate limit resets.
        
        Args:
            key: IP or session ID
            is_session: Whether this is a session limit
            
        Returns:
            Seconds until reset
        """
        limits = self._session_limits if is_session else self._ip_limits
        
        if key not in limits:
            return 0
        
        entry = limits[key]
        elapsed = time.time() - entry.window_start
        
        return max(0, int(self._window_size - elapsed))


# Global rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=settings.rate_limit_requests_per_minute,
    requests_per_session=settings.rate_limit_requests_per_session
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/api/health", "/api/ready"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check IP rate limit
        ip_allowed, ip_remaining = rate_limiter.check_ip(client_ip)
        
        if not ip_allowed:
            retry_after = rate_limiter.get_retry_after(client_ip)
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "message": "Too many requests. Please wait before trying again."
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Check session rate limit if session ID present
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            session_allowed, session_remaining = rate_limiter.check_session(session_id)
            
            if not session_allowed:
                retry_after = rate_limiter.get_retry_after(session_id, is_session=True)
                logger.warning(f"Rate limit exceeded for session: {session_id}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Session rate limit exceeded",
                        "retry_after": retry_after,
                        "message": "Too many requests for this session."
                    },
                    headers={"Retry-After": str(retry_after)}
                )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(ip_remaining)
        
        return response

