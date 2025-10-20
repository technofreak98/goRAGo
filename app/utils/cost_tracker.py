"""Cost tracking utility for OpenAI API calls."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CostMetrics:
    """Cost metrics for API calls."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost: float
    completion_cost: float
    total_cost: float
    timestamp: datetime


class CostTracker:
    """Tracks costs for OpenAI API calls."""
    
    # OpenAI pricing as of 2024 (per 1K tokens)
    PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
        "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
        "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.call_count = 0
        self.metrics_history = []
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> CostMetrics:
        """
        Calculate cost for an API call.
        
        Args:
            model: Model name used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            CostMetrics object with cost breakdown
        """
        # Get pricing for model, fallback to gpt-4o-mini if not found
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        
        # Calculate costs
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        # Create metrics object
        metrics = CostMetrics(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            timestamp=datetime.now()
        )
        
        # Update totals
        self.total_cost += total_cost
        self.call_count += 1
        self.metrics_history.append(metrics)
        
        # Only log significant costs to reduce noise
        if total_cost > 0.001:  # Only log if cost > $0.001
            logger.debug(f"API Cost: {model} - ${total_cost:.4f} ({metrics.total_tokens} tokens)")
        
        return metrics
    
    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return self.total_cost
    
    def get_call_count(self) -> int:
        """Get total number of API calls."""
        return self.call_count
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics_history:
            return {
                "total_cost": 0.0,
                "call_count": 0,
                "average_cost_per_call": 0.0,
                "models_used": [],
                "total_tokens": 0
            }
        
        models_used = list(set(metric.model for metric in self.metrics_history))
        total_tokens = sum(metric.total_tokens for metric in self.metrics_history)
        average_cost = self.total_cost / self.call_count if self.call_count > 0 else 0.0
        
        return {
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "average_cost_per_call": average_cost,
            "models_used": models_used,
            "total_tokens": total_tokens,
            "cost_by_model": self._get_cost_by_model()
        }
    
    def _get_cost_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by model."""
        model_costs = {}
        
        for metric in self.metrics_history:
            if metric.model not in model_costs:
                model_costs[metric.model] = {
                    "total_cost": 0.0,
                    "call_count": 0,
                    "total_tokens": 0
                }
            
            model_costs[metric.model]["total_cost"] += metric.total_cost
            model_costs[metric.model]["call_count"] += 1
            model_costs[metric.model]["total_tokens"] += metric.total_tokens
        
        return model_costs
    
    def reset(self):
        """Reset all metrics."""
        self.total_cost = 0.0
        self.call_count = 0
        self.metrics_history = []
        logger.debug("Cost tracker reset")


# Global cost tracker instance
cost_tracker = CostTracker()
