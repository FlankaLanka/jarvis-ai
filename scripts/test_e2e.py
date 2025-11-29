#!/usr/bin/env python3
"""
End-to-end test for Jarvis voice assistant with vector DB and GitHub integration.

This simulates the voice flow without actual audio recording.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from loguru import logger
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.api.voice import get_github_context


async def simulate_voice_query(query: str, session_id: str = "test_session"):
    """Simulate a voice query through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"🎤 User Query: {query}")
    print('='*60)
    
    # 1. Get conversation history
    history = memory_service.get_history(session_id)
    
    # 2. Retrieve vector DB context
    vectordb_context = None
    try:
        vectordb_context = await memory_service.get_combined_context(
            query=query,
            session_id=session_id,
            include_memories=True,
            include_indexed_content=True
        )
        if vectordb_context:
            print(f"📚 Vector DB context: {len(vectordb_context)} chars")
    except Exception as e:
        print(f"⚠️ Vector DB error: {e}")
    
    # 3. Retrieve GitHub context
    github_context = None
    try:
        github_context = await get_github_context(query, vectordb_context)
        if github_context:
            print(f"📁 GitHub context: {len(github_context)} chars")
    except Exception as e:
        print(f"⚠️ GitHub error: {e}")
    
    # 4. Combine contexts
    combined_context = None
    if github_context and vectordb_context:
        combined_context = f"{github_context}\n\n{vectordb_context}"
    elif github_context:
        combined_context = github_context
    elif vectordb_context:
        combined_context = vectordb_context
    
    if combined_context:
        print(f"📦 Combined context: {len(combined_context)} chars")
    
    # 5. Generate LLM response
    print("\n🤖 Jarvis Response:")
    print("-" * 40)
    
    full_response = ""
    async for token in llm_service.generate_response_stream(
        query=query,
        conversation_history=history,
        session_id=session_id,
        model="gpt-4o-mini",  # Use mini for faster tests
        context=combined_context
    ):
        print(token, end="", flush=True)
        full_response += token
    
    print("\n" + "-" * 40)
    
    # 6. Save to memory
    memory_service.add_exchange(session_id, query, full_response)
    
    return full_response


async def main():
    """Run end-to-end tests."""
    print("="*60)
    print("🚀 Jarvis Voice Assistant - End-to-End Test")
    print("="*60)
    print("Testing with Vector DB + GitHub integration")
    print()
    
    session_id = "e2e_test_session"
    
    # Test queries
    test_queries = [
        "What is the status of the construction project?",
        "Who is responsible for the structural work?",
        "What tasks are currently in progress?",
        "When is the project expected to be completed?",
    ]
    
    for query in test_queries:
        await simulate_voice_query(query, session_id)
        print()
    
    print("\n" + "="*60)
    print("✅ End-to-end test complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

