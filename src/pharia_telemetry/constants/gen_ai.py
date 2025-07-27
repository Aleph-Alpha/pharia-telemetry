"""
OpenTelemetry GenAI semantic conventions constants.

This module provides comprehensive GenAI constants following the OpenTelemetry
semantic conventions specification for generative AI operations.

Based on:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

import logging

# Import GenAI semantic convention constants from OpenTelemetry
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    # Core GenAI attributes
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID,
    # OpenAI specific attributes
    GEN_AI_OPENAI_REQUEST_SERVICE_TIER,
    GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
    GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_SEED,
    # Request/Response content attributes
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    # Response attributes
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOKEN_TYPE,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_DESCRIPTION,
    # Tool attributes
    GEN_AI_TOOL_NAME,
    # Usage and token attributes
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)


class GenAI:
    """
    OpenTelemetry GenAI semantic convention attribute keys.

    These constants provide standardized attribute keys for GenAI operations
    following the OpenTelemetry semantic conventions specification.
    """

    # =========================================================================
    # Core GenAI Attributes
    # =========================================================================

    # Agent attributes
    AGENT_ID: str = GEN_AI_AGENT_ID
    AGENT_NAME: str = GEN_AI_AGENT_NAME

    # Conversation attributes
    CONVERSATION_ID: str = GEN_AI_CONVERSATION_ID

    # Operation attributes
    OPERATION_NAME: str = GEN_AI_OPERATION_NAME
    SYSTEM: str = GEN_AI_SYSTEM

    # Model attributes
    REQUEST_MODEL: str = GEN_AI_REQUEST_MODEL
    RESPONSE_MODEL: str = GEN_AI_RESPONSE_MODEL

    # =========================================================================
    # Request Attributes
    # =========================================================================

    REQUEST_TEMPERATURE: str = GEN_AI_REQUEST_TEMPERATURE
    REQUEST_TOP_P: str = GEN_AI_REQUEST_TOP_P
    REQUEST_MAX_TOKENS: str = GEN_AI_REQUEST_MAX_TOKENS
    REQUEST_SEED: str = GEN_AI_REQUEST_SEED

    # =========================================================================
    # Response Attributes
    # =========================================================================

    RESPONSE_ID: str = GEN_AI_RESPONSE_ID
    RESPONSE_FINISH_REASONS: str = GEN_AI_RESPONSE_FINISH_REASONS
    OUTPUT_TYPE: str = GEN_AI_OUTPUT_TYPE

    # =========================================================================
    # Usage and Token Attributes
    # =========================================================================

    USAGE_INPUT_TOKENS: str = GEN_AI_USAGE_INPUT_TOKENS
    USAGE_OUTPUT_TOKENS: str = GEN_AI_USAGE_OUTPUT_TOKENS
    TOKEN_TYPE: str = GEN_AI_TOKEN_TYPE

    # =========================================================================
    # Tool Attributes
    # =========================================================================

    TOOL_NAME: str = GEN_AI_TOOL_NAME
    TOOL_DESCRIPTION: str = GEN_AI_TOOL_DESCRIPTION
    TOOL_CALL_ID: str = GEN_AI_TOOL_CALL_ID

    # =========================================================================
    # OpenAI Specific Attributes
    # =========================================================================

    OPENAI_REQUEST_SERVICE_TIER: str = GEN_AI_OPENAI_REQUEST_SERVICE_TIER
    OPENAI_RESPONSE_SERVICE_TIER: str = GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT: str = GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT

    # =========================================================================
    # Standard Values
    # =========================================================================

    class Values:
        """Standard values for GenAI attributes following OpenTelemetry conventions."""

        class OperationName:
            """
            Standard GenAI operation names.

            Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
            """

            CHAT: str = "chat"
            GENERATE_CONTENT: str = "generate_content"
            EXECUTE_TOOL: str = "execute_tool"
            CREATE_AGENT: str = "create_agent"
            EMBEDDINGS: str = "embeddings"
            INVOKE_AGENT: str = "invoke_agent"
            TEXT_COMPLETION: str = "text_completion"
            IMAGE_GENERATION: str = "image_generation"
            AUDIO_GENERATION: str = "audio_generation"
            VIDEO_GENERATION: str = "video_generation"
            CODE_GENERATION: str = "code_generation"
            DATA_ANALYSIS: str = "data_analysis"
            DATA_VISUALIZATION: str = "data_visualization"

        class System:
            """
            Standard GenAI system identifiers.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            ANTHROPIC: str = "anthropic"
            AWS_BEDROCK: str = "aws.bedrock"
            AZURE_AI_INFERENCE: str = "azure.ai.inference"
            AZURE_AI_OPENAI: str = "azure.ai.openai"
            COHERE: str = "cohere"
            DEEPSEEK: str = "deepseek"
            GCP_GEMINI: str = "gcp.gemini"
            GCP_GEN_AI: str = "gcp.gen_ai"
            GCP_VERTEX_AI: str = "gcp.vertex_ai"
            GROQ: str = "groq"
            IBM_WATSONX_AI: str = "ibm.watsonx.ai"
            MISTRAL_AI: str = "mistral_ai"
            OPENAI: str = "openai"
            PERPLEXITY: str = "perplexity"
            XAI: str = "xai"

            # Pharia-specific systems
            PHARIA_AI: str = "pharia_ai"
            CUSTOM: str = "_OTHER"

        class OutputType:
            """
            Standard output type values.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            IMAGE: str = "image"
            JSON: str = "json"
            SPEECH: str = "speech"
            TEXT: str = "text"

        class TokenType:
            """
            Standard token type values.

            Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
            """

            INPUT: str = "input"
            OUTPUT: str = "output"

        class PhariaAgentId:
            """Standard GenAI agent IDs for Pharia services."""

            QA_CHAT: str = "pharia_qa_chat"
            AGENTIC_CHAT: str = "pharia_agentic_chat"
            TRANSLATION: str = "pharia_translation"
            TRANSCRIPTION: str = "pharia_transcription"
            EASY_LANGUAGE: str = "pharia_easy_language"
            SIGN_LANGUAGE: str = "pharia_sign_language"
            FILE_UPLOAD: str = "pharia_file_upload"
            DOCUMENT_PROCESSING: str = "pharia_document_processing"
            AGENT_CREATION: str = "pharia_agent_creation"
            CUSTOM_AGENT: str = "pharia_custom_agent"


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# For backwards compatibility with existing code
Spans = GenAI

# Legacy attribute aliases
ATTR_GEN_AI_AGENT_ID = GenAI.AGENT_ID
ATTR_GEN_AI_AGENT_NAME = GenAI.AGENT_NAME
ATTR_GEN_AI_CONVERSATION_ID = GenAI.CONVERSATION_ID
ATTR_GEN_AI_OPERATION_NAME = GenAI.OPERATION_NAME
ATTR_GEN_AI_REQUEST_MODEL = GenAI.REQUEST_MODEL
ATTR_GEN_AI_RESPONSE_MODEL = GenAI.RESPONSE_MODEL
ATTR_GEN_AI_SYSTEM = GenAI.SYSTEM
ATTR_GEN_AI_USAGE_INPUT_TOKENS = GenAI.USAGE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = GenAI.USAGE_OUTPUT_TOKENS
ATTR_GEN_AI_RESPONSE_ID = GenAI.RESPONSE_ID
ATTR_GEN_AI_RESPONSE_FINISH_REASONS = GenAI.RESPONSE_FINISH_REASONS
ATTR_GEN_AI_TOKEN_TYPE = GenAI.TOKEN_TYPE
ATTR_GEN_AI_TOOL_NAME = GenAI.TOOL_NAME
ATTR_GEN_AI_TOOL_DESCRIPTION = GenAI.TOOL_DESCRIPTION
ATTR_GEN_AI_TOOL_CALL_ID = GenAI.TOOL_CALL_ID
