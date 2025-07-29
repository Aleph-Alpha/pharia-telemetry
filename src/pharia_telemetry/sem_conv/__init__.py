"""
Semantic conventions package for OpenTelemetry.

This package contains semantic convention utilities and helpers for different
OpenTelemetry instrumentation domains, starting with GenAI operations.
"""

# Re-export all GenAI functionality
from pharia_telemetry.sem_conv.baggage import Baggage, BaggageKeys, Spans

# Re-export constants
from pharia_telemetry.sem_conv.gen_ai import (
    # Data structures
    DataContext,
    GenAI,
    create_agent_creation_span,
    create_agent_invocation_span,
    # Convenience functions for specific operations
    create_chat_span,
    create_embeddings_span,
    # Core GenAI functions
    create_genai_span,
    create_tool_execution_span,
    set_genai_span_response,
    set_genai_span_usage,
)

# Re-export setup function from utils
from pharia_telemetry.setup import setup_telemetry

__all__ = [
    # Core GenAI functions
    "create_genai_span",
    "set_genai_span_usage",
    "set_genai_span_response",
    # Setup function
    "setup_telemetry",
    # Data structures
    "DataContext",
    # Convenience functions for specific operations
    "create_chat_span",
    "create_embeddings_span",
    "create_tool_execution_span",
    "create_agent_creation_span",
    "create_agent_invocation_span",
    # Constants
    "GenAI",
    "Baggage",
    "BaggageKeys",
    "Spans",
]
