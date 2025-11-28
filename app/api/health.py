"""
Jarvis Voice Assistant - Health Check Endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the status of the application and its services.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "api": "online",
            "openai_configured": bool(settings.openai_api_key),
            "github_configured": bool(settings.github_token),
        }
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    Returns whether the application is ready to handle requests.
    """
    is_ready = bool(settings.openai_api_key)
    
    if is_ready:
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "reason": "OpenAI API key not configured"}

