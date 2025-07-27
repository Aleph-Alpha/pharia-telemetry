"""
OpenTelemetry constants for Pharia services.

This package provides standardized constants for OpenTelemetry instrumentation
including baggage keys, span attributes, and semantic conventions.

Modules:
- telemetry: General telemetry constants, baggage keys, and non-GenAI span attributes
- gen_ai: GenAI-specific constants following OpenTelemetry semantic conventions
"""

# Import main telemetry constants
# Import GenAI constants
from pharia_telemetry.constants.gen_ai import (
    # Legacy aliases for backwards compatibility
    ATTR_GEN_AI_AGENT_ID,
    ATTR_GEN_AI_AGENT_NAME,
    ATTR_GEN_AI_CONVERSATION_ID,
    ATTR_GEN_AI_OPERATION_NAME,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
    ATTR_GEN_AI_RESPONSE_ID,
    ATTR_GEN_AI_RESPONSE_MODEL,
    ATTR_GEN_AI_SYSTEM,
    ATTR_GEN_AI_TOKEN_TYPE,
    ATTR_GEN_AI_TOOL_CALL_ID,
    ATTR_GEN_AI_TOOL_DESCRIPTION,
    ATTR_GEN_AI_TOOL_NAME,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAI,
)
from pharia_telemetry.constants.telemetry import (
    Baggage,
    BaggageKeys,  # Legacy alias
    Spans,
)

__all__ = [
    # General telemetry constants
    "Baggage",
    "Spans",
    "BaggageKeys",  # Legacy alias
    # GenAI constants
    "GenAI",
    # Legacy GenAI attribute constants (for backwards compatibility)
    "ATTR_GEN_AI_AGENT_ID",
    "ATTR_GEN_AI_AGENT_NAME",
    "ATTR_GEN_AI_CONVERSATION_ID",
    "ATTR_GEN_AI_OPERATION_NAME",
    "ATTR_GEN_AI_REQUEST_MODEL",
    "ATTR_GEN_AI_RESPONSE_MODEL",
    "ATTR_GEN_AI_SYSTEM",
    "ATTR_GEN_AI_USAGE_INPUT_TOKENS",
    "ATTR_GEN_AI_USAGE_OUTPUT_TOKENS",
    "ATTR_GEN_AI_RESPONSE_ID",
    "ATTR_GEN_AI_RESPONSE_FINISH_REASONS",
    "ATTR_GEN_AI_TOKEN_TYPE",
    "ATTR_GEN_AI_TOOL_NAME",
    "ATTR_GEN_AI_TOOL_DESCRIPTION",
    "ATTR_GEN_AI_TOOL_CALL_ID",
]
