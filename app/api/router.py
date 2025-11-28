"""
Jarvis Voice Assistant - Main API Router

Aggregates all API routes and provides the main router for the application.
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.voice import router as voice_router
from app.api.session import router as session_router

# Main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice"])
api_router.include_router(session_router, prefix="/session", tags=["Session"])

