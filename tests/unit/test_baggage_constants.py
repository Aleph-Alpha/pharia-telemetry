"""Tests for telemetry constants module."""

from pharia_telemetry.sem_conv.baggage import Baggage, BaggageKeys, Spans


class TestBaggageConstants:
    """Test baggage key constants."""

    def test_app_level_baggage_keys(self):
        """Test app-level baggage keys."""
        assert Baggage.USER_ID == "app.user.id"
        assert Baggage.SESSION_ID == "app.session.id"
        assert Baggage.USER_INTENT == "app.user.intent"

    def test_pharia_specific_baggage_keys(self):
        """Test Pharia-specific baggage keys."""
        assert Baggage.FEATURE_FLAGS == "pharia.feature.flags"
        assert Baggage.FEATURE_SET == "pharia.feature.set"
        assert Baggage.PHARIA_USER_INTENT == "pharia.user.intent"
        assert (
            Baggage.PRIVACY_POLICY_USER_MESSAGES
            == "pharia.privacy.policy.user.messages"
        )
        assert (
            Baggage.PRIVACY_POLICY_PROMPT_CONTENT
            == "pharia.privacy.policy.prompt.content"
        )
        assert (
            Baggage.PRIVACY_POLICY_PROMPT_COMPLETION
            == "pharia.privacy.policy.prompt.completion"
        )

    def test_data_correlation_baggage_keys(self):
        """Test data correlation baggage keys."""
        assert Baggage.DATA_NAMESPACES == "pharia.data.namespaces"
        assert Baggage.DATA_COLLECTIONS == "pharia.data.collections"
        assert Baggage.DATA_INDEXES == "pharia.data.indexes"
        assert Baggage.DATA_DATASET_IDS == "pharia.data.dataset.ids"

    def test_conversation_hierarchy_baggage_keys(self):
        """Test conversation hierarchy baggage keys."""
        assert Baggage.CHAT_QA_CONVERSATION_ID == "pharia.chat.qa.conversation.id"
        assert Baggage.CHAT_AGENT_CONVERSATION_ID == "pharia.chat.agent.conversation.id"
        assert Baggage.CHAT_QA_TRACE_ID == "pharia.chat.qa.trace.id"
        assert Baggage.TRANSLATION_REQUEST_ID == "pharia.translation.request.id"
        assert Baggage.TRANSCRIPTION_FILE_ID == "pharia.transcription.file.id"

    def test_user_intent_values(self):
        """Test user intent value constants."""
        assert Baggage.Values.UserIntent.QA_CHAT == "pharia_qa_chat"
        assert Baggage.Values.UserIntent.AGENTIC_CHAT == "pharia_agentic_chat"
        assert Baggage.Values.UserIntent.TRANSLATION == "pharia_translation"
        assert Baggage.Values.UserIntent.TRANSCRIPTION == "pharia_transcription"
        assert Baggage.Values.UserIntent.EASY_LANGUAGE == "pharia_easy_language"
        assert Baggage.Values.UserIntent.SIGN_LANGUAGE == "pharia_sign_language"
        assert Baggage.Values.UserIntent.FILE_UPLOAD == "pharia_file_upload"
        assert (
            Baggage.Values.UserIntent.DOCUMENT_PROCESSING
            == "pharia_document_processing"
        )
        assert Baggage.Values.UserIntent.AGENT_CREATION == "pharia_agent_creation"


class TestSpanConstants:
    """Test span attribute constants."""

    def test_intelligence_layer_attributes(self):
        """Test intelligence layer span attributes."""
        assert Spans.INTELLIGENCE_LAYER_TRACE_ID == "pharia.intelligence.layer.trace.id"
        assert Spans.INTELLIGENCE_LAYER_SPAN_ID == "pharia.intelligence.layer.span.id"


class TestBackwardsCompatibility:
    """Test backwards compatibility aliases."""

    def test_baggage_keys_alias(self):
        """Test BaggageKeys legacy alias."""
        # Test that BaggageKeys is an alias for Baggage
        assert BaggageKeys.USER_ID == Baggage.USER_ID
        assert BaggageKeys.SESSION_ID == Baggage.SESSION_ID
        assert BaggageKeys.FEATURE_FLAGS == Baggage.FEATURE_FLAGS
        assert BaggageKeys.FEATURE_SET == Baggage.FEATURE_SET
        assert BaggageKeys.PHARIA_USER_INTENT == Baggage.PHARIA_USER_INTENT
        assert (
            BaggageKeys.PRIVACY_POLICY_USER_MESSAGES
            == Baggage.PRIVACY_POLICY_USER_MESSAGES
        )

    def test_direct_imports(self):
        """Test that direct import paths work correctly."""
        from pharia_telemetry.sem_conv.baggage import Baggage as ImportedBaggage
        from pharia_telemetry.sem_conv.baggage import BaggageKeys as ImportedBaggageKeys
        from pharia_telemetry.sem_conv.baggage import Spans as ImportedSpans

        assert ImportedBaggage.FEATURE_FLAGS == "pharia.feature.flags"
        assert (
            ImportedSpans.INTELLIGENCE_LAYER_TRACE_ID
            == "pharia.intelligence.layer.trace.id"
        )
        assert ImportedBaggageKeys.USER_ID == "app.user.id"
