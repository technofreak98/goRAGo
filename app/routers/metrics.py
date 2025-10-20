"""Metrics router for exposing workflow performance data."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from app.utils.cost_tracker import cost_tracker
from app.utils.latency_tracker import latency_tracker
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/costs")
async def get_cost_metrics() -> Dict[str, Any]:
    """Get cost metrics for all API calls."""
    try:
        cost_summary = cost_tracker.get_metrics_summary()
        
        metrics_logger.log_workflow_step(
            "metrics_cost_request",
            "Cost metrics requested",
            workflow_context={"total_cost": cost_summary["total_cost"]}
        )
        
        return {
            "status": "success",
            "data": cost_summary
        }
        
    except Exception as e:
        logger.error(f"Error retrieving cost metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cost metrics: {str(e)}")


@router.get("/latency")
async def get_latency_metrics() -> Dict[str, Any]:
    """Get latency metrics for all workflow steps."""
    try:
        latency_summary = latency_tracker.get_metrics_summary()
        
        metrics_logger.log_workflow_step(
            "metrics_latency_request",
            "Latency metrics requested",
            workflow_context={"total_steps": latency_summary["total_steps"]}
        )
        
        return {
            "status": "success",
            "data": latency_summary
        }
        
    except Exception as e:
        logger.error(f"Error retrieving latency metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latency metrics: {str(e)}")


@router.get("/combined")
async def get_combined_metrics() -> Dict[str, Any]:
    """Get combined cost and latency metrics."""
    try:
        cost_summary = cost_tracker.get_metrics_summary()
        latency_summary = latency_tracker.get_metrics_summary()
        
        combined_metrics = {
            "costs": cost_summary,
            "latency": latency_summary,
            "summary": {
                "total_cost": cost_summary["total_cost"],
                "total_duration_ms": latency_summary["total_duration_ms"],
                "api_calls": cost_summary["call_count"],
                "workflow_steps": latency_summary["total_steps"],
                "success_rate": latency_summary["success_rate"],
                "average_cost_per_call": cost_summary["average_cost_per_call"],
                "average_duration_per_step": latency_summary["average_duration_ms"]
            }
        }
        
        metrics_logger.log_workflow_step(
            "metrics_combined_request",
            "Combined metrics requested",
            workflow_context={
                "total_cost": cost_summary["total_cost"],
                "total_steps": latency_summary["total_steps"]
            }
        )
        
        return {
            "status": "success",
            "data": combined_metrics
        }
        
    except Exception as e:
        logger.error(f"Error retrieving combined metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve combined metrics: {str(e)}")


@router.post("/reset")
async def reset_metrics() -> Dict[str, str]:
    """Reset all metrics."""
    try:
        cost_tracker.reset()
        latency_tracker.reset()
        
        metrics_logger.log_workflow_step(
            "metrics_reset",
            "All metrics reset",
            workflow_context={"reset": True}
        )
        
        return {
            "status": "success",
            "message": "All metrics have been reset"
        }
        
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")


@router.get("/health")
async def metrics_health_check() -> Dict[str, Any]:
    """Health check for metrics system."""
    try:
        # Check if trackers are working
        cost_summary = cost_tracker.get_metrics_summary()
        latency_summary = latency_tracker.get_metrics_summary()
        
        return {
            "status": "healthy",
            "cost_tracker": "operational",
            "latency_tracker": "operational",
            "current_metrics": {
                "total_cost": cost_summary["total_cost"],
                "total_steps": latency_summary["total_steps"]
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
