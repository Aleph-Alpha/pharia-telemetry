"""
Tests for smart sync/async GenAI span functionality.

Tests the auto-detection logic and shared code functionality.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from pharia_telemetry.sem_conv.gen_ai import (
    DataContext,
    GenAI,
    _build_span_name_and_attributes,
    _is_async_context,
    create_agent_creation_span,
    create_agent_invocation_span,
    create_chat_span,
    create_embeddings_span,
    create_genai_span,
    create_genai_span_async,
    create_genai_span_sync,
    create_tool_execution_span,
)


class TestAsyncDetection:
    """Test async context detection."""

    def test_is_async_context_false_in_sync(self):
        """Test that _is_async_context returns False in sync context."""
        assert _is_async_context() is False

    @pytest.mark.asyncio
    async def test_is_async_context_true_in_async(self):
        """Test that _is_async_context returns True in async context."""
        assert _is_async_context() is True

    def test_is_async_context_false_in_sync_even_with_loop(self):
        """Do not misdetect sync code as async when a loop exists in the thread."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            assert _is_async_context() is False
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class TestSharedLogic:
    """Test shared span creation logic."""

    def test_build_span_name_and_attributes_chat(self):
        """Test building span name and attributes for chat operation."""
        span_name, attributes = _build_span_name_and_attributes(
            operation_name=GenAI.Values.OperationName.CHAT,
            model="gpt-4",
            agent_id="test-agent",
            conversation_id="conv-123",
        )

        assert span_name == "chat gpt-4"
        assert attributes[GenAI.OPERATION_NAME] == GenAI.Values.OperationName.CHAT
        assert attributes[GenAI.REQUEST_MODEL] == "gpt-4"
        assert attributes[GenAI.AGENT_ID] == "test-agent"
        assert attributes[GenAI.CONVERSATION_ID] == "conv-123"

    def test_build_span_name_and_attributes_tool(self):
        """Test building span name and attributes for tool operation."""
        span_name, attributes = _build_span_name_and_attributes(
            operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
            tool_name="web_search",
            agent_id="tool-agent",
        )

        assert span_name == "execute_tool web_search"
        assert (
            attributes[GenAI.OPERATION_NAME] == GenAI.Values.OperationName.EXECUTE_TOOL
        )
        assert attributes[GenAI.TOOL_NAME] == "web_search"
        assert attributes[GenAI.AGENT_ID] == "tool-agent"

    def test_build_span_name_and_attributes_with_data_context(self):
        """Test building attributes with data context."""
        data_context = DataContext(
            collections=["docs"],
            dataset_ids=["dataset1"],
            namespaces=["pharia"],
            indexes=["vector_index"],
        )

        span_name, attributes = _build_span_name_and_attributes(
            operation_name=GenAI.Values.OperationName.EMBEDDINGS,
            model="text-embedding-3-small",
            data_context=data_context,
        )

        assert span_name == "embeddings text-embedding-3-small"
        assert attributes["pharia.data.collections"] == ["docs"]
        assert attributes["pharia.data.dataset.ids"] == ["dataset1"]
        assert attributes["pharia.data.namespaces"] == ["pharia"]
        assert attributes["pharia.data.indexes"] == ["vector_index"]

    def test_build_span_name_and_attributes_with_additional(self):
        """Test building attributes with additional attributes."""
        additional_attrs = {"custom_field": "custom_value", "numeric_field": 42}

        span_name, attributes = _build_span_name_and_attributes(
            operation_name=GenAI.Values.OperationName.CHAT,
            model="claude-3",
            additional_attributes=additional_attrs,
        )

        assert attributes["custom_field"] == "custom_value"
        assert attributes["numeric_field"] == 42


class TestSmartSpanSelection:
    """Test smart span selection based on context (default behavior)."""

    @patch("pharia_telemetry.sem_conv.gen_ai._is_async_context")
    @patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync")
    @patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span_async")
    def test_create_genai_span_sync_context(
        self, mock_async_span, mock_sync_span, mock_is_async
    ):
        """Test that default create_genai_span uses sync version in sync context."""
        mock_is_async.return_value = False
        mock_sync_span.return_value = MagicMock()

        create_genai_span(GenAI.Values.OperationName.CHAT, model="gpt-4")

        mock_sync_span.assert_called_once_with(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_id=None,
            agent_name=None,
            model="gpt-4",
            conversation_id=None,
            tool_name=None,
            data_context=None,
            span_kind=None,
            additional_attributes=None,
        )
        mock_async_span.assert_not_called()

    @patch("pharia_telemetry.sem_conv.gen_ai._is_async_context")
    @patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync")
    @patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span_async")
    def test_create_genai_span_async_context(
        self, mock_async_span, mock_sync_span, mock_is_async
    ):
        """Test that default create_genai_span uses async version in async context."""
        mock_is_async.return_value = True
        mock_async_span.return_value = MagicMock()

        create_genai_span(GenAI.Values.OperationName.CHAT, model="claude-3")

        mock_async_span.assert_called_once_with(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_id=None,
            agent_name=None,
            model="claude-3",
            conversation_id=None,
            tool_name=None,
            data_context=None,
            span_kind=None,
            additional_attributes=None,
        )
        mock_sync_span.assert_not_called()

    @patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span")
    def test_create_chat_span_delegates_to_smart_default(self, mock_create_genai_span):
        """Test that create_chat_span delegates to smart default create_genai_span."""
        mock_create_genai_span.return_value = MagicMock()

        create_chat_span(
            model="gpt-4", conversation_id="test-conv", agent_id="custom-agent"
        )

        mock_create_genai_span.assert_called_once_with(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_id="custom-agent",
            agent_name=None,
            model="gpt-4",
            conversation_id="test-conv",
            data_context=None,
            additional_attributes=None,
        )


class TestIntegration:
    """Integration tests for smart span functionality."""

    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    def test_smart_span_sync_integration(self, mock_get_tracer):
        """Test smart span (default create_chat_span) in actual sync context."""
        # Mock tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )
        mock_get_tracer.return_value = mock_tracer

        # Use smart span (default) in sync context
        with create_chat_span(model="gpt-4", conversation_id="test") as span:
            assert span is mock_span

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "chat gpt-4"  # span name

    @patch("pharia_telemetry.sem_conv.gen_ai.get_tracer")
    @pytest.mark.asyncio
    async def test_smart_span_async_integration(self, mock_get_tracer):
        """Test smart span (default create_chat_span) in actual async context."""
        # Mock tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )
        mock_get_tracer.return_value = mock_tracer

        # Use smart span (default) in async context
        async with create_chat_span(model="claude-3", conversation_id="test") as span:
            assert span is mock_span

        # Verify tracer was called
        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == "chat claude-3"  # span name


class TestSmartLogicComprehensive:
    """Comprehensive tests for smart context detection and span selection logic."""

    def test_smart_default_is_truly_smart(self):
        """Test that the default create_genai_span function is smart."""
        # This test verifies that calling create_genai_span in sync context
        # actually detects it's sync and behaves appropriately

        # Mock the detection to ensure we're testing the right path
        with patch(
            "pharia_telemetry.sem_conv.gen_ai._is_async_context"
        ) as mock_is_async:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync"
            ) as mock_sync:
                mock_is_async.return_value = False
                mock_sync.return_value = MagicMock()

                # Call the default function
                create_genai_span(GenAI.Values.OperationName.CHAT, model="test")

                # Verify it chose the sync path
                mock_sync.assert_called_once()
                mock_is_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_smart_default_detects_async(self):
        """Test that the default create_genai_span function detects async context."""
        # This test verifies that calling create_genai_span in async context
        # actually detects it's async and behaves appropriately

        # Mock the async function to ensure we're testing the right path
        with patch(
            "pharia_telemetry.sem_conv.gen_ai._is_async_context"
        ) as mock_is_async:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.create_genai_span_async"
            ) as mock_async:
                mock_is_async.return_value = True
                mock_async.return_value = MagicMock()

                # Call the default function in async context
                create_genai_span(GenAI.Values.OperationName.CHAT, model="test")

                # Verify it chose the async path
                mock_async.assert_called_once()
                mock_is_async.assert_called_once()

    def test_all_convenience_functions_are_smart(self):
        """Test that all convenience functions use smart detection by default."""

        # Mock the smart default function
        with patch("pharia_telemetry.sem_conv.gen_ai.create_genai_span") as mock_smart:
            mock_smart.return_value = MagicMock()

            # Test each convenience function calls the smart default
            create_chat_span(model="test")
            create_embeddings_span(model="test")
            create_tool_execution_span("test_tool")
            create_agent_creation_span(agent_name="test_agent")
            create_agent_invocation_span(model="test")

            # Verify all called the smart default
            assert mock_smart.call_count == 5

    def test_explicit_functions_bypass_smart_logic(self):
        """Test that explicit sync/async functions bypass smart detection."""
        # Mock the smart detection function
        with patch(
            "pharia_telemetry.sem_conv.gen_ai._is_async_context"
        ) as mock_is_async:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.get_tracer"
            ) as mock_get_tracer:
                mock_get_tracer.return_value = None  # No tracer available

                # Call explicit sync function
                with create_genai_span_sync(GenAI.Values.OperationName.CHAT):
                    pass  # Should work without checking async context

                # Verify smart detection was NOT called for explicit function
                mock_is_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_async_function_bypasses_smart_logic(self):
        """Test that explicit async function bypasses smart detection."""
        # Mock the smart detection function
        with patch(
            "pharia_telemetry.sem_conv.gen_ai._is_async_context"
        ) as mock_is_async:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.get_tracer"
            ) as mock_get_tracer:
                mock_get_tracer.return_value = None  # No tracer available

                # Call explicit async function
                async with create_genai_span_async(GenAI.Values.OperationName.CHAT):
                    pass  # Should work without checking async context

                # Verify smart detection was NOT called for explicit function
                mock_is_async.assert_not_called()

    def test_smart_detection_with_all_parameters(self):
        """Test smart detection works with all possible parameters."""
        data_context = DataContext(collections=["test"], indexes=["idx"])
        additional_attrs = {"custom": "value"}

        with patch(
            "pharia_telemetry.sem_conv.gen_ai._is_async_context"
        ) as mock_is_async:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync"
            ) as mock_sync:
                mock_is_async.return_value = False
                mock_sync.return_value = MagicMock()

                # Call with all parameters
                create_genai_span(
                    operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
                    agent_id="test-agent",
                    agent_name="Test Agent",
                    model="test-model",
                    conversation_id="conv-123",
                    tool_name="test-tool",
                    data_context=data_context,
                    span_kind=None,
                    additional_attributes=additional_attrs,
                )

                # Verify all parameters were passed through correctly
                mock_sync.assert_called_once_with(
                    operation_name=GenAI.Values.OperationName.EXECUTE_TOOL,
                    agent_id="test-agent",
                    agent_name="Test Agent",
                    model="test-model",
                    conversation_id="conv-123",
                    tool_name="test-tool",
                    data_context=data_context,
                    span_kind=None,
                    additional_attributes=additional_attrs,
                )

    def test_legacy_smart_functions_work(self):
        """Test that the default smart functions work correctly."""
        with patch(
            "pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync"
        ) as mock_sync:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai._is_async_context"
            ) as mock_is_async:
                mock_is_async.return_value = False
                mock_sync.return_value = MagicMock()

                # Call the default smart function
                create_genai_span(
                    operation_name=GenAI.Values.OperationName.CHAT, model="test"
                )

                # Verify it delegated to the sync function
                mock_sync.assert_called_once_with(
                    operation_name=GenAI.Values.OperationName.CHAT,
                    agent_id=None,
                    agent_name=None,
                    model="test",
                    conversation_id=None,
                    tool_name=None,
                    data_context=None,
                    span_kind=None,
                    additional_attributes=None,
                )

    def test_smart_logic_performance(self):
        """Test that smart detection doesn't add significant overhead."""
        import time

        # Mock functions to avoid actual span creation
        with patch(
            "pharia_telemetry.sem_conv.gen_ai.create_genai_span_sync"
        ) as mock_sync:
            with patch(
                "pharia_telemetry.sem_conv.gen_ai.create_genai_span_async"
            ) as mock_async:
                mock_sync.return_value = MagicMock()
                mock_async.return_value = MagicMock()

                # Time multiple calls
                start_time = time.time()
                for _ in range(100):
                    create_genai_span(GenAI.Values.OperationName.CHAT, model="test")
                end_time = time.time()

                # Should be very fast (less than 0.1 seconds for 100 calls)
                total_time = end_time - start_time
                assert total_time < 0.1, (
                    f"Smart detection too slow: {total_time:.3f}s for 100 calls"
                )


@pytest.mark.asyncio
async def test_async_context_detection_in_pytest():
    """Test that async context detection works correctly in pytest async context."""
    # This should detect we're in an async context
    assert _is_async_context() is True


if __name__ == "__main__":
    # Run sync tests
    pytest.main([__file__, "-v"])
