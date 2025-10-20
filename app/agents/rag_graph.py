import logging
import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from app.agents.router_node import router_node
from app.agents.retrieval_node import retrieval_node
from app.agents.weather_node import weather_node
from app.agents.guardrail_node import guardrail_node
from app.agents.generation_node import generation_node
from app.utils.cost_tracker import cost_tracker
from app.utils.latency_tracker import latency_tracker
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class RAGGraph:
    """Main LangGraph workflow for the RAG system."""
    
    def __init__(self):
        """Initialize RAG graph."""
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        try:
            # Define the state schema
            from typing import TypedDict, Annotated
            from operator import add
            
            class AgentState(TypedDict):
                query: Annotated[str, lambda x, y: y]  # Keep the latest query value
                route: str
                confidence: float
                reasoning: str
                document_context: str
                weather_context: str
                document_sources: Annotated[List[Dict], add]
                weather_sources: Annotated[List[Dict], add]
                extracted_places: List[str]  # Places extracted from documents
                final_answer: str
                sources: Annotated[List[Dict], add]
            
            # Create the state graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("router", router_node.route_query)
            workflow.add_node("documents", retrieval_node.retrieve_documents)
            workflow.add_node("weather", weather_node.retrieve_weather)
            workflow.add_node("guardrail", guardrail_node.handle_out_of_scope)
            workflow.add_node("generation", generation_node.generate_response)
            
            # Set entry point
            workflow.set_entry_point("router")
            
            # Add single conditional edge from router
            workflow.add_conditional_edges(
                "router",
                self._route_from_router,
                {
                    "weather": "weather",
                    "documents": "documents", 
                    "guardrail": "guardrail",
                    "__end__": END
                }
            )
            
            # Add edges from weather node
            workflow.add_edge("weather", "generation")
            
            # Add conditional edges from documents node
            workflow.add_conditional_edges(
                "documents",
                self._route_from_documents,
                {
                    "weather": "weather",
                    "generation": "generation",
                    "__end__": END
                }
            )
            
            # Add edges from guardrail node
            workflow.add_edge("guardrail", END)
            
            # Add edge from generation node
            workflow.add_edge("generation", END)
            
            # Compile the graph
            self.graph = workflow.compile()
            
            logger.info("RAG graph compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to build RAG graph: {e}")
            raise RuntimeError(f"Graph construction failed: {e}")
    
    def _route_from_router(self, state: Dict[str, Any]) -> str:
        """Route from router node based on the route decision."""
        route = state.get("route", "")
        
        if route == "weather_only":
            return "weather"
        elif route == "document_only":
            return "documents"
        elif route == "combined":
            # For combined queries, we need to go to documents first to extract places
            # Then weather will use those places
            return "documents"
        elif route == "out_of_scope":
            return "guardrail"
        else:
            return "__end__"
    
    def _route_from_documents(self, state: Dict[str, Any]) -> str:
        """Route from documents node based on the original route decision."""
        route = state.get("route", "")
        
        if route == "combined":
            # For combined queries, go to weather after documents
            return "weather"
        elif route == "document_only":
            # For document-only queries, go directly to generation
            return "generation"
        else:
            return "__end__"
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG graph.
        
        Args:
            query: User query to process
            session_id: Optional session ID for conversation tracking
            
        Returns:
            Dict with response and metadata
        """
        workflow_start_time = time.time()
        initial_cost = cost_tracker.get_total_cost()
        
        try:
            if not self.graph:
                raise RuntimeError("Graph not initialized")
            
            # Initial state
            initial_state = {
                "query": query,
                "route": "",
                "confidence": 0.0,
                "reasoning": "",
                "document_context": "",
                "weather_context": "",
                "document_sources": [],
                "weather_sources": [],
                "extracted_places": [],
                "final_answer": "",
                "sources": []
            }
            
            
            # Execute the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Calculate workflow metrics
            workflow_duration = (time.time() - workflow_start_time) * 1000
            final_cost = cost_tracker.get_total_cost()
            workflow_cost = final_cost - initial_cost
            
            # Get latency summary
            latency_summary = latency_tracker.get_metrics_summary()
            
            # Format response
            response = {
                "answer": result.get("final_answer", ""),
                "route": result.get("route", ""),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "sources": result.get("sources", []),
                "session_id": session_id,
                "workflow_metrics": {
                    "total_duration_ms": workflow_duration,
                    "total_cost": workflow_cost,
                    "api_calls": cost_tracker.get_call_count(),
                    "steps_completed": latency_summary.get("total_steps", 0),
                    "success_rate": latency_summary.get("success_rate", 0.0)
                }
            }
            
            # Log only significant workflows
            if workflow_duration > 5000 or workflow_cost > 0.01:  # Only log if > 5s or > $0.01
                logger.info(f"Workflow completed: {workflow_duration:.0f}ms, ${workflow_cost:.4f}, {cost_tracker.get_call_count()} calls")
            
            return response
            
        except Exception as e:
            workflow_duration = (time.time() - workflow_start_time) * 1000
            final_cost = cost_tracker.get_total_cost()
            workflow_cost = final_cost - initial_cost
            
            logger.error(f"Workflow failed: {e} ({workflow_duration:.0f}ms, ${workflow_cost:.4f})")
            
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "route": "error",
                "confidence": 0.0,
                "reasoning": f"Processing error: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "workflow_metrics": {
                    "total_duration_ms": workflow_duration,
                    "total_cost": workflow_cost,
                    "api_calls": cost_tracker.get_call_count(),
                    "steps_completed": 0,
                    "success_rate": 0.0
                }
            }
    
    def health_check(self) -> bool:
        """
        Check if the graph is healthy and ready to process queries.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.graph:
                return False
            
            # Simple check - just verify graph is compiled
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global RAG graph instance
rag_graph = RAGGraph()
