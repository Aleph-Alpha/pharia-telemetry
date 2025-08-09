"""
Unit tests for custom OpenTelemetry span processors.

Tests the BaggageSpanProcessor to ensure baggage is correctly
added to spans as attributes.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from pharia_telemetry.baggage.processors import BaggageSpanProcessor


class TestBaggageSpanProcessor:
    """Test suite for BaggageSpanProcessor."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.processor = BaggageSpanProcessor()
        self.mock_span = Mock()
        self.mock_span.set_attribute = Mock()

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_on_start_adds_baggage_as_attributes(self, mock_baggage: Mock) -> None:
        """Test that baggage items are added as span attributes."""
        # Arrange
        mock_baggage.get_all.return_value = {
            "app.user.id": "user123",
            "app.session.id": "session456",
            "aa.chat.qa.conversation.id": "conv789",
        }

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        mock_baggage.get_all.assert_called_once()
        expected_calls = [
            ("app.user.id", "user123"),
            ("app.session.id", "session456"),
            ("aa.chat.qa.conversation.id", "conv789"),
        ]

        actual_calls = [
            call.args for call in self.mock_span.set_attribute.call_args_list
        ]
        assert len(actual_calls) == 3
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_on_start_skips_none_values(self, mock_baggage: Mock) -> None:
        """Test that None baggage values are skipped."""
        # Arrange
        mock_baggage.get_all.return_value = {
            "app.user.id": "user123",
            "app.session.id": None,  # Should be skipped
            "aa.chat.qa.conversation.id": "conv789",
        }

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        expected_calls = [
            ("app.user.id", "user123"),
            ("aa.chat.qa.conversation.id", "conv789"),
        ]

        actual_calls = [
            call.args for call in self.mock_span.set_attribute.call_args_list
        ]
        assert len(actual_calls) == 2
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_on_start_converts_values_to_strings(self, mock_baggage: Mock) -> None:
        """Test that baggage values are converted to strings."""
        # Arrange
        mock_baggage.get_all.return_value = {
            "app.user.id": 123,  # Integer
            "app.session.timeout": 3600.5,  # Float
            "app.feature.enabled": True,  # Boolean
        }

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        expected_calls = [
            ("app.user.id", "123"),
            ("app.session.timeout", "3600.5"),
            ("app.feature.enabled", "True"),
        ]

        actual_calls = [
            call.args for call in self.mock_span.set_attribute.call_args_list
        ]
        assert len(actual_calls) == 3
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_on_start_empty_baggage(self, mock_baggage: Mock) -> None:
        """Test behavior when baggage is empty."""
        # Arrange
        mock_baggage.get_all.return_value = {}

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        mock_baggage.get_all.assert_called_once()
        self.mock_span.set_attribute.assert_not_called()

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", False)
    def test_on_start_otel_not_available(self) -> None:
        """Test behavior when OpenTelemetry is not available."""
        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        self.mock_span.set_attribute.assert_not_called()

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    @patch("pharia_telemetry.baggage.processors.logger")
    def test_on_start_handles_exceptions(
        self, mock_logger: Mock, mock_baggage: Mock
    ) -> None:
        """Test that exceptions during baggage processing are handled."""
        # Arrange
        mock_baggage.get_all.side_effect = Exception("Baggage error")

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        mock_logger.error.assert_called_once()
        assert (
            "Failed to set baggage attributes on span"
            in mock_logger.error.call_args[0][0]
        )
        self.mock_span.set_attribute.assert_not_called()

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_on_start_with_parent_context(self, mock_baggage: Mock) -> None:
        """Test that parent_context parameter is accepted but not used."""
        # Arrange
        mock_baggage.get_all.return_value = {"app.user.id": "user123"}
        mock_context = Mock()

        # Act
        self.processor.on_start(self.mock_span, parent_context=mock_context)

        # Assert
        mock_baggage.get_all.assert_called_once()
        self.mock_span.set_attribute.assert_called_once_with("app.user.id", "user123")

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_prefix_filter_functionality(self, mock_baggage: Mock) -> None:
        """Test prefix filter functionality."""
        # Arrange
        processor = BaggageSpanProcessor(prefix_filter="app.")
        mock_baggage.get_all.return_value = {
            "app.user.id": "user123",
            "app.session.id": "session456",
            "aa.chat.qa.conversation.id": "conv789",  # Should be filtered out
        }

        # Act
        processor.on_start(self.mock_span)

        # Assert
        expected_calls = [
            ("app.user.id", "user123"),
            ("app.session.id", "session456"),
        ]

        actual_calls = [
            call.args for call in self.mock_span.set_attribute.call_args_list
        ]
        assert len(actual_calls) == 2
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    def test_on_end_no_op(self) -> None:
        """Test that on_end does nothing."""
        # Arrange
        mock_readable_span = Mock()

        # Act
        self.processor.on_end(mock_readable_span)

        # Assert - No exceptions should be raised

    def test_shutdown_no_op(self) -> None:
        """Test that shutdown does nothing."""
        # Act
        self.processor.shutdown()

        # Assert - No exceptions should be raised

    def test_force_flush_returns_true(self) -> None:
        """Test that force_flush always returns True."""
        # Act
        result = self.processor.force_flush()

        # Assert
        assert result is True

    def test_force_flush_with_timeout(self) -> None:
        """Test that force_flush accepts timeout parameter."""
        # Act
        result = self.processor.force_flush(timeout_millis=5000)

        # Assert
        assert result is True

    @patch("pharia_telemetry.baggage.processors.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.processors.baggage")
    def test_integration_realistic_baggage(self, mock_baggage: Mock) -> None:
        """Test with realistic baggage data from Pharia applications."""
        # Arrange
        realistic_baggage = {
            "app.user.id": "user_550e8400-e29b-41d4-a716-446655440000",
            "app.session.id": "session_6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "aa.chat.qa.conversation.id": "conv_123456789",
            "aa.chat.agent.conversation.id": "agent_conv_987654321",
            "aa.data.namespaces": "production,analytics",
            "aa.data.collections": "documents,embeddings",
            "aa.transcription.file.id": "trans_job_abc123",
        }
        mock_baggage.get_all.return_value = realistic_baggage

        # Act
        self.processor.on_start(self.mock_span)

        # Assert
        assert self.mock_span.set_attribute.call_count == len(realistic_baggage)

        # Verify all expected attributes were set
        actual_calls = {
            call.args[0]: call.args[1]
            for call in self.mock_span.set_attribute.call_args_list
        }
        assert actual_calls == realistic_baggage


@pytest.mark.integration
class TestBaggageSpanProcessorIntegration:
    """Integration tests for BaggageSpanProcessor with real OpenTelemetry components."""

    @pytest.fixture
    def tracer_provider(self) -> tuple[Any, Any]:
        """Create a test tracer provider."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            provider = TracerProvider()
            exporter = InMemorySpanExporter()
            processor = SimpleSpanProcessor(exporter)
            provider.add_span_processor(processor)

            return provider, exporter
        except ImportError:
            pytest.skip("OpenTelemetry not available for integration tests")

    def test_baggage_processor_with_real_tracer(
        self, tracer_provider: tuple[Any, Any]
    ) -> None:
        """Test BaggageSpanProcessor with a real OpenTelemetry tracer."""
        try:
            from opentelemetry import baggage, context
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            # Create a fresh provider with correct processor order
            provider = TracerProvider()
            exporter = InMemorySpanExporter()

            # CRITICAL: Add baggage processor FIRST, then export processor
            baggage_processor = BaggageSpanProcessor()
            provider.add_span_processor(baggage_processor)
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            # Create a tracer
            tracer = provider.get_tracer(__name__)

            # Set baggage in context and create span within that context
            # Create context with baggage
            ctx = context.get_current()
            ctx = baggage.set_baggage("app.user.id", "test_user_123", context=ctx)
            ctx = baggage.set_baggage("app.session.id", "test_session_456", context=ctx)

            # Use the context when creating the span
            token = context.attach(ctx)
            try:
                # Create a span - it should automatically get baggage attributes
                with tracer.start_as_current_span("test_span") as span:
                    span.set_attribute("manual.attribute", "manual_value")
            finally:
                context.detach(token)

            # Verify the span was exported with baggage attributes
            exported_spans = exporter.get_finished_spans()
            assert len(exported_spans) == 1

            exported_span = exported_spans[0]
            attributes = dict(exported_span.attributes or {})

            # Check that baggage was automatically added
            assert attributes.get("app.user.id") == "test_user_123"
            assert attributes.get("app.session.id") == "test_session_456"

            # Check that manual attributes are also present
            assert attributes.get("manual.attribute") == "manual_value"

        except ImportError:
            pytest.skip("OpenTelemetry not available for integration tests")
