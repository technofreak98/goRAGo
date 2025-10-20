"""Simplified logging configuration for the RAG system."""

import logging
import sys
from typing import Dict, Any, Optional
from app.utils.cost_tracker import cost_tracker
from app.utils.latency_tracker import latency_tracker


class SimpleFormatter(logging.Formatter):
    """Simple formatter with optional metrics."""
    
    def __init__(self, include_metrics: bool = False):
        super().__init__()
        self.include_metrics = include_metrics
    
    def format(self, record):
        """Format log record with optional metrics."""
        base_msg = super().format(record)
        
        if self.include_metrics and hasattr(record, 'metrics'):
            metrics = record.metrics
            base_msg += f" [Cost: ${metrics.get('cost', 0):.4f}, Duration: {metrics.get('duration_ms', 0):.1f}ms]"
        
        return base_msg


class MetricsLogger:
    """Simplified logger for workflow metrics."""
    
    def __init__(self, name: str):
        """Initialize metrics logger."""
        self.logger = logging.getLogger(name)
        self.cost_tracker = cost_tracker
        self.latency_tracker = latency_tracker
    
    def log_workflow_step(self, step_name: str, message: str, level: int = logging.INFO, 
                         workflow_context: Optional[Dict[str, Any]] = None):
        """Log a workflow step with minimal overhead."""
        # Only log important steps, not every operation
        if level >= logging.INFO:
            self.logger.log(level, f"[{step_name}] {message}")
    
    def log_api_call(self, model: str, prompt_tokens: int, completion_tokens: int = 0):
        """Log API call with cost tracking (minimal logging)."""
        cost_metrics = self.cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
        # Only log significant costs
        if cost_metrics.total_cost > 0.001:  # Only log if cost > $0.001
            self.logger.info(f"API Call: {model} - ${cost_metrics.total_cost:.4f} ({cost_metrics.total_tokens} tokens)")
    
    def get_logger(self):
        """Get the underlying logger."""
        return self.logger


def setup_logging(log_level: str = "INFO", include_metrics: bool = False):
    """
    Setup simplified logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        include_metrics: Whether to include metrics in log messages
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatter
    formatter = SimpleFormatter(include_metrics=include_metrics)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers - reduce noise
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, Metrics: {include_metrics}")


def get_metrics_logger(name: str) -> MetricsLogger:
    """Get a metrics logger instance."""
    return MetricsLogger(name)
