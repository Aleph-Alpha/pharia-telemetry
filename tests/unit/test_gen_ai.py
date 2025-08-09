"""Tests for GenAI convenience functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from pharia_telemetry.sem_conv.gen_ai import (
    DataContext,
    GenAI,
    create_agent_creation_span,
    create_agent_creation_span_async,
    create_agent_creation_span_sync,
    create_agent_invocation_span,
    create_agent_invocation_span_async,
    create_agent_invocation_span_sync,
    # Smart convenience functions
    create_chat_span,
    # Async convenience functions
    create_chat_span_async,
    # Sync convenience functions
    create_chat_span_sync,
    create_embeddings_span,
    create_embeddings_span_async,
    create_embeddings_span_sync,
    create_genai_span,
    create_tool_execution_span,
    create_tool_execution_span_async,
    create_tool_execution_span_sync,
    # Span attribute setters
    set_genai_span_response,
    set_genai_span_usage,
)

# Import additional components for test assertions
try:
    from opentelemetry.trace import NonRecordingSpan
except ImportError:
    NonRecordingSpan = None

# Import SpanKind for test assertions
try:
    from opentelemetry.trace import SpanKind
except ImportError:
    SpanKind = None


class TestCreateGenAISpan:
    """Test the create_genai_span convenience function."""

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_genai_span_basic(self, mock_get_tracer):
        """Test basic GenAI span creation."""
        # Mock tracer and span with proper context manager
        mock_span = Mock()
        mock_tracer = Mock()

        # Create a proper context manager mock
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_span
        context_manager.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = context_manager
        mock_get_tracer.return_value = mock_tracer

        # Test basic span creation
        with create_genai_span(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_name="Test Agent",
            model="gpt-4",
        ) as span:
            assert span == mock_span

        # Verify tracer was called with correct span name and attributes
        expected_attributes = {
            GenAI.OPERATION_NAME: "chat",
            GenAI.REQUEST_MODEL: "gpt-4",
            GenAI.AGENT_NAME: "Test Agent",
        }
        mock_tracer.start_as_current_span.assert_called_once_with(
            "chat gpt-4", kind=SpanKind.CLIENT, attributes=expected_attributes
        )

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_genai_span_with_agent(self, mock_get_tracer):
        """Test GenAI span creation with agent information."""
        # Mock tracer and span with proper context manager
        mock_span = Mock()
        mock_tracer = Mock()

        # Create a proper context manager mock
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_span
        context_manager.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = context_manager
        mock_get_tracer.return_value = mock_tracer

        # Test span creation with agent info
        with create_genai_span(
            operation_name=GenAI.Values.OperationName.INVOKE_AGENT,
            model="gpt-4",
            agent_name="Math Tutor",
            agent_id="agent_123",
            conversation_id="conv_456",
        ) as span:
            assert span == mock_span

        # Verify all attributes were passed to start_as_current_span
        call_args = mock_tracer.start_as_current_span.call_args
        span_name, kwargs = call_args[0][0], call_args[1]
        attributes = kwargs["attributes"]

        assert span_name == "invoke_agent Math Tutor"
        assert attributes[GenAI.AGENT_NAME] == "Math Tutor"
        assert attributes[GenAI.AGENT_ID] == "agent_123"
        assert attributes[GenAI.CONVERSATION_ID] == "conv_456"
        assert attributes[GenAI.REQUEST_MODEL] == "gpt-4"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_genai_span_execute_tool(self, mock_get_tracer):
        """Test GenAI span creation for tool execution."""
        # Mock tracer and span with proper context manager
        mock_span = Mock()
        mock_tracer = Mock()

        # Create a proper context manager mock
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_span
        context_manager.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = context_manager
        mock_get_tracer.return_value = mock_tracer

        # Test tool execution span
        with create_genai_span(
            operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
            tool_name="calculator",
            agent_name="Tool Agent",
            additional_attributes={
                "gen_ai.tool.description": "Performs calculations",
            },
        ) as span:
            assert span == mock_span

        # Verify span name and attributes were passed correctly
        expected_attributes = {
            GenAI.OPERATION_NAME: "execute_tool",
            GenAI.AGENT_NAME: "Tool Agent",
            GenAI.TOOL_NAME: "calculator",
            "gen_ai.tool.description": "Performs calculations",
        }
        mock_tracer.start_as_current_span.assert_called_once_with(
            "execute_tool calculator",
            kind=SpanKind.CLIENT,
            attributes=expected_attributes,
        )

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", False)
    def test_create_genai_span_no_otel(self):
        """Test GenAI span creation when OpenTelemetry is not available."""
        with create_genai_span(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_name="Test Agent",
            model="gpt-4",
        ) as span:
            # Should return a NonRecordingSpan when no OTEL available
            assert isinstance(span, NonRecordingSpan)

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_genai_span_no_tracer(self, mock_get_tracer):
        """Test GenAI span creation when tracer is not available."""
        mock_get_tracer.return_value = None

        with create_genai_span(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_name="Test Agent",
            model="gpt-4",
        ) as span:
            # Should return a NonRecordingSpan when no tracer available
            assert isinstance(span, NonRecordingSpan)


class TestSetGenAISpanUsage:
    """Test the set_genai_span_usage convenience function."""

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_usage(self, mock_get_current_span):
        """Test setting usage information on a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span

        set_genai_span_usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        # Verify span attributes were set
        expected_calls = [
            (GenAI.USAGE_INPUT_TOKENS, 100),
            (GenAI.USAGE_OUTPUT_TOKENS, 50),
            ("gen_ai.usage.total_tokens", 150),
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_usage_partial(self, mock_get_current_span):
        """Test setting partial usage information on a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span

        set_genai_span_usage(input_tokens=100)

        # Verify only input tokens were set
        mock_span.set_attribute.assert_called_with(GenAI.USAGE_INPUT_TOKENS, 100)

        # Verify output tokens weren't set
        assert not any(
            call[0][0] == GenAI.USAGE_OUTPUT_TOKENS
            for call in mock_span.set_attribute.call_args_list
        )

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", False)
    def test_set_genai_span_usage_no_otel(self):
        """Test setting usage when OpenTelemetry is not available."""
        # Should not raise an exception
        set_genai_span_usage(input_tokens=100)

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_usage_none_span(self, mock_get_current_span):
        """Test setting usage when no current span."""
        mock_get_current_span.return_value = None

        # Should not raise an exception
        set_genai_span_usage(input_tokens=100)

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_usage_auto_total(self, mock_get_current_span):
        """Test automatic total calculation."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span

        set_genai_span_usage(
            input_tokens=100,
            output_tokens=50,
            # total_tokens not provided - should be calculated
        )

        # Verify total was calculated automatically
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)


class TestSetGenAISpanResponse:
    """Test the set_genai_span_response convenience function."""

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_response(self, mock_get_current_span):
        """Test setting response information on a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span

        set_genai_span_response(
            response_id="resp_123",
            model="gpt-4-0613",
            finish_reasons=["stop", "length"],
            system_fingerprint="fp_123",
        )

        # Verify span attributes were set
        expected_calls = [
            (GenAI.RESPONSE_ID, "resp_123"),
            (GenAI.RESPONSE_MODEL, "gpt-4-0613"),
            (GenAI.RESPONSE_FINISH_REASONS, ["stop", "length"]),
            (GenAI.OPENAI_RESPONSE_SYSTEM_FINGERPRINT, "fp_123"),
        ]

        for attr_name, attr_value in expected_calls:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_response_partial(self, mock_get_current_span):
        """Test setting partial response information on a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_current_span.return_value = mock_span

        set_genai_span_response(response_id="resp_123")

        # Verify only response ID was set
        mock_span.set_attribute.assert_called_with(GenAI.RESPONSE_ID, "resp_123")

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", False)
    def test_set_genai_span_response_no_otel(self):
        """Test setting response when OpenTelemetry is not available."""
        # Should not raise an exception
        set_genai_span_response(response_id="resp_123")


class TestSmartConvenienceSpans:
    """Tests for smart (auto sync/async) convenience functions."""

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_chat_span_smart(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        dc = DataContext(collections=["kb"])
        with create_chat_span(model="gpt-4", conversation_id="c-1", data_context=dc):
            pass

        call = mock_tracer.start_as_current_span.call_args
        name, kwargs = call[0][0], call[1]
        attrs = kwargs["attributes"]
        assert name == "chat gpt-4"
        assert attrs[GenAI.OPERATION_NAME] == "chat"
        assert attrs[GenAI.REQUEST_MODEL] == "gpt-4"
        assert attrs[GenAI.CONVERSATION_ID] == "c-1"
        assert attrs["pharia.data.collections"] == ["kb"]

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_embeddings_span_smart(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_embeddings_span(model="embed-1"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "embeddings embed-1"
        assert attrs[GenAI.OPERATION_NAME] == "embeddings"
        assert attrs[GenAI.REQUEST_MODEL] == "embed-1"
        assert attrs[GenAI.AGENT_NAME] == "embeddings_agent"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_tool_execution_span_smart(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_tool_execution_span("search", conversation_id="conv"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "execute_tool search"
        assert attrs[GenAI.OPERATION_NAME] == "execute_tool"
        assert attrs[GenAI.TOOL_NAME] == "search"
        assert attrs[GenAI.CONVERSATION_ID] == "conv"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_agent_creation_span_smart(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_agent_creation_span(agent_name="AgentX", agent_id="aid"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "create_agent AgentX"
        assert attrs[GenAI.OPERATION_NAME] == "create_agent"
        assert attrs[GenAI.AGENT_NAME] == "AgentX"
        assert attrs[GenAI.AGENT_ID] == "aid"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_agent_invocation_span_smart(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_agent_invocation_span(agent_name="QA Assistant", model="m-1"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "invoke_agent QA Assistant"
        assert attrs[GenAI.OPERATION_NAME] == "invoke_agent"
        assert attrs[GenAI.AGENT_NAME] == "QA Assistant"
        assert attrs[GenAI.REQUEST_MODEL] == "m-1"


class TestSyncConvenienceSpans:
    """Tests for explicit sync convenience functions (mypy-safe)."""

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_chat_span_sync(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_chat_span_sync(model="gpt-4"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "chat gpt-4"
        assert attrs[GenAI.OPERATION_NAME] == "chat"
        assert attrs[GenAI.REQUEST_MODEL] == "gpt-4"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_embeddings_span_sync(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_embeddings_span_sync(model="embed-2"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "embeddings embed-2"
        assert attrs[GenAI.OPERATION_NAME] == "embeddings"
        assert attrs[GenAI.REQUEST_MODEL] == "embed-2"
        assert attrs[GenAI.AGENT_NAME] == "embeddings_agent"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_tool_execution_span_sync(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_tool_execution_span_sync("web_search", conversation_id="c-2"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "execute_tool web_search"
        assert attrs[GenAI.OPERATION_NAME] == "execute_tool"
        assert attrs[GenAI.TOOL_NAME] == "web_search"
        assert attrs[GenAI.CONVERSATION_ID] == "c-2"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_agent_creation_span_sync(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_agent_creation_span_sync(agent_name="Builder", agent_id="id-1"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "create_agent Builder"
        assert attrs[GenAI.OPERATION_NAME] == "create_agent"
        assert attrs[GenAI.AGENT_NAME] == "Builder"
        assert attrs[GenAI.AGENT_ID] == "id-1"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_create_agent_invocation_span_sync(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        with create_agent_invocation_span_sync(agent_name="Helper", model="m-2"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "invoke_agent Helper"
        assert attrs[GenAI.OPERATION_NAME] == "invoke_agent"
        assert attrs[GenAI.AGENT_NAME] == "Helper"
        assert attrs[GenAI.REQUEST_MODEL] == "m-2"


class TestAsyncConvenienceSpans:
    """Tests for explicit async convenience functions."""

    @pytest.mark.asyncio
    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    async def test_create_chat_span_async(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        async with create_chat_span_async(model="gpt-4", conversation_id="c-3"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "chat gpt-4"
        assert attrs[GenAI.OPERATION_NAME] == "chat"
        assert attrs[GenAI.CONVERSATION_ID] == "c-3"

    @pytest.mark.asyncio
    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    async def test_create_embeddings_span_async(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        async with create_embeddings_span_async(model="embed-3"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "embeddings embed-3"
        assert attrs[GenAI.OPERATION_NAME] == "embeddings"
        assert attrs[GenAI.AGENT_NAME] == "embeddings_agent"

    @pytest.mark.asyncio
    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    async def test_create_tool_execution_span_async(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        async with create_tool_execution_span_async("calc", conversation_id="conv-4"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "execute_tool calc"
        assert attrs[GenAI.OPERATION_NAME] == "execute_tool"
        assert attrs[GenAI.TOOL_NAME] == "calc"
        assert attrs[GenAI.CONVERSATION_ID] == "conv-4"

    @pytest.mark.asyncio
    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    async def test_create_agent_creation_span_async(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        async with create_agent_creation_span_async(
            agent_name="Creator", agent_id="id-2"
        ):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "create_agent Creator"
        assert attrs[GenAI.OPERATION_NAME] == "create_agent"
        assert attrs[GenAI.AGENT_NAME] == "Creator"
        assert attrs[GenAI.AGENT_ID] == "id-2"

    @pytest.mark.asyncio
    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    async def test_create_agent_invocation_span_async(self, mock_get_tracer):
        mock_span = Mock()
        mock_tracer = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_span
        ctx.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = ctx
        mock_get_tracer.return_value = mock_tracer

        async with create_agent_invocation_span_async(agent_name="Runner", model="m-3"):
            pass

        name = mock_tracer.start_as_current_span.call_args[0][0]
        kwargs = mock_tracer.start_as_current_span.call_args[1]
        attrs = kwargs["attributes"]
        assert name == "invoke_agent Runner"
        assert attrs[GenAI.OPERATION_NAME] == "invoke_agent"
        assert attrs[GenAI.AGENT_NAME] == "Runner"
        assert attrs[GenAI.REQUEST_MODEL] == "m-3"

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    @patch("opentelemetry.trace.get_current_span")
    def test_set_genai_span_response_none_span(self, mock_get_current_span):
        """Test setting response when no current span."""
        mock_get_current_span.return_value = None

        # Should not raise an exception
        set_genai_span_response(response_id="resp_123")
