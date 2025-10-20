"""Agent router for LangGraph-based intelligent query processing."""

import logging
from fastapi import APIRouter, HTTPException
from app.models import AgentQuery, AgentResponse
from app.services.langgraph_agent import LangGraphAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Initialize the agent (singleton pattern)
_agent_instance = None

def get_agent() -> LangGraphAgent:
    """Get or create the agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LangGraphAgent()
    return _agent_instance


@router.post("/query", response_model=AgentResponse)
async def process_agent_query(query: AgentQuery):
    """
    Process a user query through the LangGraph agent.
    
    This endpoint handles intelligent routing of queries to:
    - Document retrieval for literature questions
    - Weather API for weather queries
    - Combined processing for travel + weather questions
    - Guardrails for out-of-scope queries
    
    Args:
        query: User's question
        
    Returns:
        AgentResponse with answer, sources, and metadata
    """
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get agent instance
        agent = get_agent()
        
        # Process the query
        response = await agent.process_query(query.query)
        
        # Log the response
        logger.info(f"Agent query processed: route={response.route_taken}, time={response.processing_time_ms:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing agent query: {e}")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")


@router.get("/health")
async def agent_health_check():
    """Check health of the agent and all its components."""
    try:
        agent = get_agent()
        health_status = await agent.health_check()
        
        # Determine overall health
        all_healthy = all(health_status.values())
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "components": health_status,
            "agent_type": "LangGraph Router Agent"
        }
        
    except Exception as e:
        logger.error(f"Error in agent health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "agent_type": "LangGraph Router Agent"
        }


@router.get("/info")
async def get_agent_info():
    """Get information about the agent capabilities."""
    return {
        "agent_type": "LangGraph Router Agent",
        "capabilities": [
            "Document retrieval and literature questions",
            "Weather information for travel planning",
            "Combined literature + weather queries",
            "Intelligent query routing",
            "Location extraction from documents",
            "Enhanced query preprocessing"
        ],
        "supported_routes": [
            "document - Questions about literature and places in books",
            "weather - Current weather conditions",
            "combined - Literature + weather for travel planning",
            "guardrails - Out-of-scope query handling"
        ],
        "example_queries": [
            "What places did Mark Twain visit?",
            "What's the weather in Rome?",
            "I want to visit places Twain went to in Italy - what's the weather?",
            "Tell me about quantum physics (will be handled by guardrails)"
        ]
    }
