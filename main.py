"""
Jarvis Voice Assistant - Main Application Entry Point

This FastAPI application serves both the static frontend files and the API endpoints.
All API routes are under /api/ and the frontend is served from the static/ directory.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.api.router import api_router
from app.config import settings
from app.services.session import session_manager
from app.integrations.api_connector import api_connector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Starting Jarvis Voice Assistant...")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Start background tasks
    session_manager.start_cleanup_task()
    api_connector.start_refresh_scheduler()
    logger.info(f"API auto-refresh enabled (every {settings.api_refresh_interval_seconds}s)")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Jarvis Voice Assistant...")
    session_manager.stop_cleanup_task()
    api_connector.stop_refresh_scheduler()
    await api_connector.close()


# Create FastAPI application
app = FastAPI(
    title="Jarvis Voice Assistant",
    description="Real-time, audio-first voice assistant with verified responses",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiting middleware
from app.middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# Mount API routes under /api
app.include_router(api_router, prefix="/api")

# Determine static files directory
static_dir = Path(__file__).parent / "static"

# Mount static files if the directory exists
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
    
    @app.get("/")
    async def serve_index():
        """Serve the frontend index.html."""
        return FileResponse(static_dir / "index.html")
    
    @app.get("/{path:path}")
    async def serve_static(path: str):
        """Serve static files or fall back to index.html for SPA routing."""
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")
else:
    @app.get("/")
    async def no_frontend():
        """Placeholder when frontend is not built."""
        return {
            "message": "Jarvis API is running. Frontend not built yet.",
            "api_docs": "/docs",
            "health": "/api/health"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

