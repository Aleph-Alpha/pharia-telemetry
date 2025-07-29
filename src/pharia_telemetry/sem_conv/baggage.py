"""
OpenTelemetry baggage keys and general telemetry constants.

This module provides standardized OpenTelemetry baggage keys and semantic conventions
for proper trace correlation and context propagation across Pharia services.

For GenAI-specific constants, see pharia_telemetry.sem_conv.gen_ai_constants module.

Based on OpenTelemetry semantic conventions:
- https://opentelemetry.io/docs/specs/semconv/general/general-attributes/
- https://opentelemetry.io/docs/specs/semconv/general/trace/
"""

import logging

logger = logging.getLogger(__name__)


class Baggage:
    """
    OpenTelemetry baggage keys for trace correlation.

    These keys enable context propagation across service boundaries and should be
    used consistently across all Pharia services for proper correlation.
    """

    # App-level correlation IDs (cross-cutting context)
    USER_ID: str = "app.user.id"
    SESSION_ID: str = "app.session.id"
    USER_INTENT: str = "app.user.intent"  # High-level user intent

    # Pharia-specific feature and policy context
    FEATURE_FLAGS: str = "pharia.feature.flags"
    FEATURE_SET: str = "pharia.feature.set"
    PHARIA_USER_INTENT: str = "pharia.user.intent"
    PRIVACY_POLICY_USER_MESSAGES: str = "pharia.privacy.policy.user.messages"
    PRIVACY_POLICY_PROMPT_CONTENT: str = "pharia.privacy.policy.prompt.content"
    PRIVACY_POLICY_PROMPT_COMPLETION: str = "pharia.privacy.policy.prompt.completion"

    # Data correlation IDs (plural forms with dots)
    DATA_NAMESPACES: str = "pharia.data.namespaces"
    DATA_COLLECTIONS: str = "pharia.data.collections"
    DATA_INDEXES: str = "pharia.data.indexes"
    DATA_DATASET_IDS: str = "pharia.data.dataset.ids"

    # Conversation hierarchy (fine-grained session management)
    CHAT_QA_CONVERSATION_ID: str = "pharia.chat.qa.conversation.id"
    CHAT_AGENT_CONVERSATION_ID: str = "pharia.chat.agent.conversation.id"
    CHAT_QA_TRACE_ID: str = "pharia.chat.qa.trace.id"
    TRANSLATION_REQUEST_ID: str = "pharia.translation.request.id"
    TRANSCRIPTION_FILE_ID: str = "pharia.transcription.file.id"

    class Values:
        """Standard baggage values for cross-cutting context."""

        class UserIntent:
            """Standard user intent values for context propagation."""

            QA_CHAT: str = "pharia_qa_chat"
            AGENTIC_CHAT: str = "pharia_agentic_chat"
            TRANSLATION: str = "pharia_translation"
            TRANSCRIPTION: str = "pharia_transcription"
            EASY_LANGUAGE: str = "pharia_easy_language"
            SIGN_LANGUAGE: str = "pharia_sign_language"
            FILE_UPLOAD: str = "pharia_file_upload"
            DOCUMENT_PROCESSING: str = "pharia_document_processing"
            AGENT_CREATION: str = "pharia_agent_creation"


class Spans:
    """
    OpenTelemetry span attribute keys for general (non-GenAI) operations.

    For GenAI-specific span attributes, use pharia_telemetry.sem_conv.gen_ai_constants.GenAI
    """

    # Intelligence Layer correlation (Pharia-specific)
    INTELLIGENCE_LAYER_TRACE_ID: str = "pharia.intelligence.layer.trace.id"
    INTELLIGENCE_LAYER_SPAN_ID: str = "pharia.intelligence.layer.span.id"


# =============================================================================
# Backwards Compatibility
# =============================================================================


# Legacy BaggageKeys class alias for backwards compatibility
class BaggageKeys(Baggage):
    """Legacy alias for Baggage class. Use Baggage instead."""

    pass
