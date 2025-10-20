"""Latency tracking utility for workflow steps."""

import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency metrics for workflow steps."""
    step_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_message: Optional[str] = None


class LatencyTracker:
    """Tracks latency for workflow steps."""
    
    def __init__(self):
        """Initialize latency tracker."""
        self.metrics_history = []
        self.step_timers = {}
    
    def start_step(self, step_name: str) -> str:
        """
        Start timing a step.
        
        Args:
            step_name: Name of the step being timed
            
        Returns:
            Timer ID for tracking
        """
        timer_id = f"{step_name}_{int(time.time() * 1000000)}"
        self.step_timers[timer_id] = {
            "step_name": step_name,
            "start_time": time.time(),
            "start_datetime": datetime.now()
        }
        
        return timer_id
    
    def end_step(self, timer_id: str, success: bool = True, error_message: Optional[str] = None) -> LatencyMetrics:
        """
        End timing a step.
        
        Args:
            timer_id: Timer ID from start_step
            success: Whether the step completed successfully
            error_message: Error message if step failed
            
        Returns:
            LatencyMetrics object
        """
        if timer_id not in self.step_timers:
            logger.warning(f"Timer ID {timer_id} not found")
            return None
        
        timer_data = self.step_timers.pop(timer_id)
        end_time = time.time()
        duration_ms = (end_time - timer_data["start_time"]) * 1000
        
        metrics = LatencyMetrics(
            step_name=timer_data["step_name"],
            start_time=timer_data["start_datetime"],
            end_time=datetime.now(),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        self.metrics_history.append(metrics)
        
        # Only log slow steps or errors to reduce noise
        if duration_ms > 1000 or not success:  # Only log if > 1 second or failed
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Step {timer_data['step_name']} completed in {duration_ms:.2f}ms - {status}")
            
            if error_message:
                logger.error(f"Step {timer_data['step_name']} error: {error_message}")
        
        return metrics
    
    def get_step_metrics(self, step_name: str) -> list[LatencyMetrics]:
        """Get all metrics for a specific step."""
        return [m for m in self.metrics_history if m.step_name == step_name]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics_history:
            return {
                "total_steps": 0,
                "total_duration_ms": 0.0,
                "average_duration_ms": 0.0,
                "success_rate": 0.0,
                "steps": {}
            }
        
        total_duration = sum(m.duration_ms for m in self.metrics_history)
        successful_steps = sum(1 for m in self.metrics_history if m.success)
        success_rate = successful_steps / len(self.metrics_history) if self.metrics_history else 0.0
        
        # Group by step name
        steps = {}
        for metric in self.metrics_history:
            if metric.step_name not in steps:
                steps[metric.step_name] = {
                    "count": 0,
                    "total_duration_ms": 0.0,
                    "average_duration_ms": 0.0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "min_duration_ms": float('inf'),
                    "max_duration_ms": 0.0
                }
            
            step_data = steps[metric.step_name]
            step_data["count"] += 1
            step_data["total_duration_ms"] += metric.duration_ms
            step_data["min_duration_ms"] = min(step_data["min_duration_ms"], metric.duration_ms)
            step_data["max_duration_ms"] = max(step_data["max_duration_ms"], metric.duration_ms)
            
            if metric.success:
                step_data["success_count"] += 1
        
        # Calculate averages and success rates
        for step_data in steps.values():
            step_data["average_duration_ms"] = step_data["total_duration_ms"] / step_data["count"]
            step_data["success_rate"] = step_data["success_count"] / step_data["count"]
            if step_data["min_duration_ms"] == float('inf'):
                step_data["min_duration_ms"] = 0.0
        
        return {
            "total_steps": len(self.metrics_history),
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / len(self.metrics_history),
            "success_rate": success_rate,
            "steps": steps
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics_history = []
        self.step_timers = {}
        logger.debug("Latency tracker reset")


def track_latency(step_name: str):
    """
    Decorator to track latency of a function.
    
    Args:
        step_name: Name of the step for tracking
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                timer_id = latency_tracker.start_step(step_name)
                try:
                    result = await func(*args, **kwargs)
                    latency_tracker.end_step(timer_id, success=True)
                    return result
                except Exception as e:
                    latency_tracker.end_step(timer_id, success=False, error_message=str(e))
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                timer_id = latency_tracker.start_step(step_name)
                try:
                    result = func(*args, **kwargs)
                    latency_tracker.end_step(timer_id, success=True)
                    return result
                except Exception as e:
                    latency_tracker.end_step(timer_id, success=False, error_message=str(e))
                    raise
            return sync_wrapper
    return decorator


# Global latency tracker instance
latency_tracker = LatencyTracker()
