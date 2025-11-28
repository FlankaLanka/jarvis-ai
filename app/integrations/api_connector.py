"""
Jarvis Voice Assistant - Public API Connector

Generic connector for public APIs with caching and refresh scheduling.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import httpx
from loguru import logger

from app.config import settings


@dataclass
class CachedResponse:
    """Cached API response with timestamp."""
    data: Any
    fetched_at: datetime
    expires_at: datetime


@dataclass
class APIEndpoint:
    """Configuration for an API endpoint."""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    refresh_interval: int = 180  # seconds
    transform: Optional[Callable[[Any], Any]] = None


class APIConnector:
    """
    Generic connector for public APIs.
    
    Features:
    1. Configurable endpoints
    2. Response caching
    3. Automatic refresh scheduling
    4. Data validation
    """
    
    def __init__(self):
        self._endpoints: Dict[str, APIEndpoint] = {}
        self._cache: Dict[str, CachedResponse] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=30.0)
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """
        Register an API endpoint.
        
        Args:
            endpoint: The endpoint configuration
        """
        self._endpoints[endpoint.name] = endpoint
        logger.info(f"Registered API endpoint: {endpoint.name}")
    
    async def fetch(
        self,
        endpoint_name: str,
        force_refresh: bool = False,
        **params
    ) -> Optional[Any]:
        """
        Fetch data from an API endpoint.
        
        Args:
            endpoint_name: Name of the registered endpoint
            force_refresh: Force a fresh fetch, ignoring cache
            **params: Additional parameters to merge with endpoint params
            
        Returns:
            The API response data or None on error
        """
        if endpoint_name not in self._endpoints:
            logger.error(f"Unknown endpoint: {endpoint_name}")
            return None
        
        endpoint = self._endpoints[endpoint_name]
        
        # Check cache
        if not force_refresh and endpoint_name in self._cache:
            cached = self._cache[endpoint_name]
            if cached.expires_at > datetime.utcnow():
                logger.debug(f"Using cached response for {endpoint_name}")
                return cached.data
        
        # Fetch fresh data
        try:
            merged_params = {**endpoint.params, **params}
            
            logger.debug(f"Fetching from {endpoint_name}: {endpoint.url}")
            
            if endpoint.method.upper() == "GET":
                response = await self._client.get(
                    endpoint.url,
                    headers=endpoint.headers,
                    params=merged_params
                )
            else:
                response = await self._client.post(
                    endpoint.url,
                    headers=endpoint.headers,
                    params=merged_params
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Apply transform if defined
            if endpoint.transform:
                data = endpoint.transform(data)
            
            # Cache the response
            now = datetime.utcnow()
            self._cache[endpoint_name] = CachedResponse(
                data=data,
                fetched_at=now,
                expires_at=now + timedelta(seconds=endpoint.refresh_interval)
            )
            
            logger.debug(f"Fetched and cached response for {endpoint_name}")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {endpoint_name}: {e}")
            # Return stale cache if available
            if endpoint_name in self._cache:
                logger.warning(f"Returning stale cache for {endpoint_name}")
                return self._cache[endpoint_name].data
            return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint_name}: {e}")
            return None
    
    async def fetch_all(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch data from all registered endpoints.
        
        Args:
            force_refresh: Force fresh fetch for all endpoints
            
        Returns:
            Dictionary of endpoint names to response data
        """
        results = {}
        
        for name in self._endpoints:
            data = await self.fetch(name, force_refresh=force_refresh)
            results[name] = data
        
        return results
    
    def get_cached(self, endpoint_name: str) -> Optional[Any]:
        """
        Get cached data without fetching.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Cached data or None
        """
        if endpoint_name in self._cache:
            return self._cache[endpoint_name].data
        return None
    
    def clear_cache(self, endpoint_name: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            endpoint_name: Specific endpoint to clear, or None for all
        """
        if endpoint_name:
            self._cache.pop(endpoint_name, None)
            logger.debug(f"Cleared cache for {endpoint_name}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cache")
    
    def start_refresh_scheduler(self):
        """Start the background refresh scheduler."""
        async def refresh_loop():
            while True:
                await asyncio.sleep(settings.api_refresh_interval_seconds)
                logger.debug("Running scheduled API refresh")
                await self.fetch_all(force_refresh=True)
        
        self._refresh_task = asyncio.create_task(refresh_loop())
        logger.info("API refresh scheduler started")
    
    def stop_refresh_scheduler(self):
        """Stop the background refresh scheduler."""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
            logger.info("API refresh scheduler stopped")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# Global API connector instance
api_connector = APIConnector()

