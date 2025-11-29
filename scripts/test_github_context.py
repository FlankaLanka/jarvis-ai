#!/usr/bin/env python3
"""Test the GitHub context retrieval function."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from app.api.voice import get_github_context
from loguru import logger

async def test():
    print("=== Test 3: GitHub Context Retrieval ===")
    
    # Test construction project query
    query = "What is the status of the construction project?"
    print(f"Query: {query}")
    context = await get_github_context(query)
    if context:
        print(f"Context length: {len(context)} chars")
        print(f"First 800 chars:")
        print(context[:800])
        print("...")
    else:
        print("No context found")
    
    print()
    
    # Test tasks query
    query2 = "What tasks are in progress?"
    print(f"Query: {query2}")
    context2 = await get_github_context(query2)
    if context2:
        print(f"Context length: {len(context2)} chars")
        print(f"First 800 chars:")
        print(context2[:800])
        print("...")
    else:
        print("No context found")
    
    print()
    
    # Test team query
    query3 = "Who is the project manager?"
    print(f"Query: {query3}")
    context3 = await get_github_context(query3)
    if context3:
        print(f"Context length: {len(context3)} chars")
    else:
        print("No context found")

if __name__ == "__main__":
    asyncio.run(test())


