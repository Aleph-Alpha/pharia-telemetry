"""Tests for baggage propagation utilities."""

from unittest.mock import Mock, patch

from pharia_telemetry.baggage.propagation import (
    get_all_baggage,
    get_baggage_item,
    set_baggage_item,
    set_baggage_span_attributes,
    set_gen_ai_span_attributes,
)
from pharia_telemetry.sem_conv.baggage import Baggage
from pharia_telemetry.sem_conv.gen_ai import GenAI


class TestSetBaggageSpanAttributes:
    """Test cases for set_baggage_span_attributes function."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    @patch("pharia_telemetry.baggage.propagation.baggage.get_all")
    def test_sets_baggage_as_span_attributes(
        self, mock_get_all: Mock, mock_get_current_span: Mock
    ) -> None:
        """Test that baggage items are set as span attributes."""
        # Setup
        mock_span = Mock()
        mock_get_current_span.return_value = mock_span
        mock_get_all.return_value = {
            Baggage.USER_ID: "user123",
            Baggage.SESSION_ID: "session456",
        }

        # Execute
        set_baggage_span_attributes()

        # Verify
        mock_span.set_attribute.assert_any_call(Baggage.USER_ID, "user123")
        mock_span.set_attribute.assert_any_call(Baggage.SESSION_ID, "session456")

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    @patch("pharia_telemetry.baggage.propagation.baggage.get_all")
    def test_with_provided_span(
        self, mock_get_all: Mock, mock_get_current_span: Mock
    ) -> None:
        """Test function works with explicitly provided span."""
        # Setup
        mock_current_span = Mock()
        mock_provided_span = Mock()
        mock_get_current_span.return_value = mock_current_span
        mock_get_all.return_value = {Baggage.USER_ID: "user123"}

        # Execute
        set_baggage_span_attributes(span=mock_provided_span)

        # Verify
        mock_provided_span.set_attribute.assert_called_with(Baggage.USER_ID, "user123")
        mock_current_span.set_attribute.assert_not_called()

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", False)
    def test_otel_not_available(self) -> None:
        """Test function handles gracefully when OpenTelemetry is not available."""
        # Execute - should not raise exception
        set_baggage_span_attributes()

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    def test_no_active_span(self, mock_get_current_span: Mock) -> None:
        """Test function handles gracefully when no span is active."""
        # Setup
        mock_get_current_span.return_value = None

        # Execute - should not raise exception
        set_baggage_span_attributes()


class TestSetGenAiSpanAttributes:
    """Test cases for set_gen_ai_span_attributes function."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    def test_sets_genai_attributes(self, mock_get_current_span: Mock) -> None:
        """Test that GenAI attributes are set correctly."""
        # Setup
        mock_span = Mock()
        mock_get_current_span.return_value = mock_span

        # Execute
        set_gen_ai_span_attributes(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
        )

        # Verify
        mock_span.set_attribute.assert_any_call(
            GenAI.OPERATION_NAME,
            GenAI.Values.OperationName.CHAT,
        )
        mock_span.set_attribute.assert_any_call(
            GenAI.AGENT_ID,
            GenAI.Values.PhariaAgentId.QA_CHAT,
        )

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    def test_sets_optional_attributes(self, mock_get_current_span: Mock) -> None:
        """Test that optional attributes are set when provided."""
        # Setup
        mock_span = Mock()
        mock_get_current_span.return_value = mock_span

        # Execute
        set_gen_ai_span_attributes(
            operation_name=GenAI.Values.OperationName.CHAT,
            agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
            conversation_id="conv123",
            model_name="gpt-4",
        )

        # Verify
        mock_span.set_attribute.assert_any_call(GenAI.CONVERSATION_ID, "conv123")
        mock_span.set_attribute.assert_any_call(GenAI.REQUEST_MODEL, "gpt-4")

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", False)
    def test_otel_not_available(self) -> None:
        """Test function handles gracefully when OpenTelemetry is not available."""
        # Execute - should not raise exception
        set_gen_ai_span_attributes("chat", "qa_chat")


class TestSetBaggageItem:
    """Test cases for set_baggage_item function."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.set_baggage")
    def test_sets_baggage_item(self, mock_set_baggage: Mock) -> None:
        """Test that baggage item is set."""
        # Execute
        set_baggage_item(Baggage.USER_ID, "user123")

        # Verify
        mock_set_baggage.assert_called_once_with(Baggage.USER_ID, "user123")

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", False)
    def test_otel_not_available(self) -> None:
        """Test function handles gracefully when OpenTelemetry is not available."""
        # Execute - should not raise exception
        set_baggage_item(Baggage.USER_ID, "user123")


class TestGetBaggageItem:
    """Test cases for get_baggage_item function."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.get_baggage")
    def test_gets_baggage_item(self, mock_get_baggage: Mock) -> None:
        """Test that baggage item is retrieved."""
        # Setup
        mock_get_baggage.return_value = "user123"

        # Execute
        result = get_baggage_item(Baggage.USER_ID)

        # Verify
        mock_get_baggage.assert_called_once_with(Baggage.USER_ID)
        assert result == "user123"

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", False)
    def test_otel_not_available(self) -> None:
        """Test function returns None when OpenTelemetry is not available."""
        # Execute
        result = get_baggage_item(Baggage.USER_ID)

        # Verify
        assert result is None


class TestGetAllBaggage:
    """Test cases for get_all_baggage function."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.get_all")
    def test_gets_all_baggage(self, mock_get_all: Mock) -> None:
        """Test that all baggage items are retrieved."""
        # Setup
        expected_baggage = {
            Baggage.USER_ID: "user123",
            Baggage.SESSION_ID: "session456",
        }
        mock_get_all.return_value = expected_baggage

        # Execute
        result = get_all_baggage()

        # Verify
        mock_get_all.assert_called_once()
        assert result == expected_baggage

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", False)
    def test_otel_not_available(self) -> None:
        """Test function returns empty dict when OpenTelemetry is not available."""
        # Execute
        result = get_all_baggage()

        # Verify
        assert result == {}


class TestErrorHandling:
    """Test error handling in baggage propagation functions."""

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.set_baggage")
    @patch("pharia_telemetry.baggage.propagation.logger")
    def test_set_baggage_item_handles_exceptions(
        self, mock_logger: Mock, mock_set_baggage: Mock
    ) -> None:
        """Test that exceptions in set_baggage_item are handled."""
        # Setup
        mock_set_baggage.side_effect = Exception("Baggage error")

        # Execute
        set_baggage_item(Baggage.USER_ID, "user123")

        # Verify
        mock_logger.error.assert_called_once()

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.get_baggage")
    @patch("pharia_telemetry.baggage.propagation.logger")
    def test_get_baggage_item_handles_exceptions(
        self, mock_logger: Mock, mock_get_baggage: Mock
    ) -> None:
        """Test that exceptions in get_baggage_item are handled."""
        # Setup
        mock_get_baggage.side_effect = Exception("Baggage error")

        # Execute
        result = get_baggage_item(Baggage.USER_ID)

        # Verify
        mock_logger.error.assert_called_once()
        assert result is None

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.baggage.get_all")
    @patch("pharia_telemetry.baggage.propagation.logger")
    def test_get_all_baggage_handles_exceptions(
        self, mock_logger: Mock, mock_get_all: Mock
    ) -> None:
        """Test that exceptions in get_all_baggage are handled."""
        # Setup
        mock_get_all.side_effect = Exception("Baggage error")

        # Execute
        result = get_all_baggage()

        # Verify
        mock_logger.error.assert_called_once()
        assert result == {}

    @patch("pharia_telemetry.baggage.propagation.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.baggage.propagation.trace.get_current_span")
    @patch("pharia_telemetry.baggage.propagation.logger")
    def test_set_gen_ai_span_attributes_handles_exceptions(
        self, mock_logger: Mock, mock_get_current_span: Mock
    ) -> None:
        """Test that exceptions in set_gen_ai_span_attributes are handled."""
        # Setup
        mock_get_current_span.side_effect = Exception("Span error")

        # Execute
        set_gen_ai_span_attributes("chat", "qa_chat")

        # Verify
        mock_logger.error.assert_called_once()
