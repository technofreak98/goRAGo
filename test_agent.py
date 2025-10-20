#!/usr/bin/env python3
"""Test script for the LangGraph agent system."""

import asyncio
import json
from app.services.langgraph_agent import LangGraphAgent

async def test_agent():
    """Test the agent with different types of queries."""
    
    # Initialize agent
    print("Initializing LangGraph agent...")
    agent = LangGraphAgent()
    
    # Test queries
    test_queries = [
        "What places did Mark Twain visit?",  # Document query
        "What's the weather in Rome?",  # Weather query
        "I want to visit places Twain went to in Italy - what's the weather?",  # Combined query
        "Explain quantum physics"  # Guardrails query
    ]
    
    print("\nTesting agent with different query types...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 50)
        
        try:
            response = await agent.process_query(query)
            
            print(f"Route: {response.route_taken}")
            print(f"Answer: {response.answer[:200]}...")
            print(f"Processing time: {response.processing_time_ms:.2f}ms")
            print(f"Sources: {len(response.sources)}")
            print(f"Weather data: {len(response.weather_data)}")
            print(f"Locations: {len(response.locations)}")
            
            if response.error:
                print(f"Error: {response.error}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "="*60 + "\n")
    
    # Test health check
    print("Testing health check...")
    health = await agent.health_check()
    print(f"Health status: {health}")

if __name__ == "__main__":
    asyncio.run(test_agent())
