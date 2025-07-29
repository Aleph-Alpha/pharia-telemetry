"""Tests for context injectors."""

from unittest.mock import Mock, patch

from pharia_telemetry.logging.injectors import (
    BaggageContextInjector,
    CompositeContextInjector,
    TraceContextInjector,
    create_baggage_injector,
    create_full_context_injector,
    create_trace_injector,
)
from pharia_telemetry.sem_conv.baggage import BaggageKeys


class TestTraceContextInjector:
    """Test TraceContextInjector."""

    def test_init_defaults(self) -> None:
        """Test injector initialization with defaults."""
        injector = TraceContextInjector()

        assert injector.include_trace_id is True
        assert injector.include_span_id is True
        assert injector.trace_id_key == "trace_id"
        assert injector.span_id_key == "span_id"

    def test_init_custom_config(self) -> None:
        """Test injector initialization with custom configuration."""
        injector = TraceContextInjector(
            include_trace_id=False,
            include_span_id=True,
            trace_id_key="custom_trace",
            span_id_key="custom_span",
        )

        assert injector.include_trace_id is False
        assert injector.include_span_id is True
        assert injector.trace_id_key == "custom_trace"
        assert injector.span_id_key == "custom_span"

    @patch("pharia_telemetry.logging.injectors.OTEL_AVAILABLE", False)
    def test_inject_no_otel(self) -> None:
        """Test injection when OpenTelemetry is not available."""
        injector = TraceContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result == {"message": "test"}

    @patch("pharia_telemetry.logging.injectors.trace.get_current_span")
    def test_inject_no_span(self, mock_get_span: Mock) -> None:
        """Test injection when no current span is available."""
        mock_get_span.return_value = None

        injector = TraceContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result == {"message": "test"}

    @patch("pharia_telemetry.logging.injectors.trace.get_current_span")
    def test_inject_with_trace_context(self, mock_get_span: Mock) -> None:
        """Test injection with valid trace context."""
        # Mock span with valid context
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span.get_span_context.return_value = mock_span_context
        mock_get_span.return_value = mock_span

        injector = TraceContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result["message"] == "test"
        assert result["trace_id"] == "12345678901234567890123456789012"
        assert result["span_id"] == "1234567890123456"

    @patch("pharia_telemetry.logging.injectors.trace.get_current_span")
    def test_inject_custom_keys(self, mock_get_span: Mock) -> None:
        """Test injection with custom key names."""
        # Mock span with valid context
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span.get_span_context.return_value = mock_span_context
        mock_get_span.return_value = mock_span

        injector = TraceContextInjector(
            trace_id_key="custom_trace",
            span_id_key="custom_span",
        )
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result["custom_trace"] == "12345678901234567890123456789012"
        assert result["custom_span"] == "1234567890123456"

    @patch("pharia_telemetry.logging.injectors.trace.get_current_span")
    def test_inject_selective_inclusion(self, mock_get_span: Mock) -> None:
        """Test selective inclusion of trace ID and span ID."""
        # Mock span with valid context
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span.get_span_context.return_value = mock_span_context
        mock_get_span.return_value = mock_span

        # Only include trace ID
        injector = TraceContextInjector(include_trace_id=True, include_span_id=False)
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert "trace_id" in result
        assert "span_id" not in result


class TestBaggageContextInjector:
    """Test BaggageContextInjector."""

    def test_init_defaults(self) -> None:
        """Test injector initialization with defaults."""
        injector = BaggageContextInjector()

        assert injector.prefix_filter is None
        assert injector.exclude_keys == set()

    def test_init_custom_config(self) -> None:
        """Test injector initialization with custom configuration."""
        exclude_keys = {"exclude1", "exclude2"}
        injector = BaggageContextInjector(
            prefix_filter="app.",
            exclude_keys=exclude_keys,
        )

        assert injector.prefix_filter == "app."
        assert injector.exclude_keys == exclude_keys

    @patch("pharia_telemetry.logging.injectors.OTEL_AVAILABLE", False)
    def test_inject_no_otel(self) -> None:
        """Test injection when OpenTelemetry is not available."""
        injector = BaggageContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result == {"message": "test"}

    @patch("pharia_telemetry.logging.injectors.baggage.get_all")
    def test_inject_with_baggage(self, mock_get_all: Mock) -> None:
        """Test injection with baggage values."""
        mock_get_all.return_value = {
            BaggageKeys.USER_ID: "user123",
            BaggageKeys.SESSION_ID: "session456",
            "custom.key": "custom_value",
        }

        injector = BaggageContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result["message"] == "test"
        assert result[BaggageKeys.USER_ID] == "user123"
        assert result[BaggageKeys.SESSION_ID] == "session456"
        assert result["custom.key"] == "custom_value"

    @patch("pharia_telemetry.logging.injectors.baggage.get_all")
    def test_inject_with_prefix_filter(self, mock_get_all: Mock) -> None:
        """Test injection with prefix filter."""
        mock_get_all.return_value = {
            "app.user.id": "user123",
            "app.session.id": "session456",
            "system.metric": "value",
            "other.key": "other_value",
        }

        injector = BaggageContextInjector(prefix_filter="app.")
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result["app.user.id"] == "user123"
        assert result["app.session.id"] == "session456"
        assert "system.metric" not in result
        assert "other.key" not in result

    @patch("pharia_telemetry.logging.injectors.baggage.get_all")
    def test_inject_with_exclude_keys(self, mock_get_all: Mock) -> None:
        """Test injection with excluded keys."""
        mock_get_all.return_value = {
            BaggageKeys.USER_ID: "user123",
            BaggageKeys.SESSION_ID: "session456",
            "sensitive.key": "sensitive_value",
        }

        injector = BaggageContextInjector(exclude_keys={"sensitive.key"})
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result[BaggageKeys.USER_ID] == "user123"
        assert result[BaggageKeys.SESSION_ID] == "session456"
        assert "sensitive.key" not in result

    @patch("pharia_telemetry.logging.injectors.baggage.get_all")
    def test_inject_skips_empty_values(self, mock_get_all: Mock) -> None:
        """Test that empty values are skipped."""
        mock_get_all.return_value = {
            BaggageKeys.USER_ID: "user123",
            "empty.key": "",
            "none.key": None,
        }

        injector = BaggageContextInjector()
        log_dict = {"message": "test"}

        result = injector.inject(log_dict)

        assert result[BaggageKeys.USER_ID] == "user123"
        assert "empty.key" not in result
        assert "none.key" not in result


class TestCompositeContextInjector:
    """Test CompositeContextInjector."""

    def test_init(self) -> None:
        """Test composite injector initialization."""
        injector1 = TraceContextInjector()
        injector2 = BaggageContextInjector()

        composite = CompositeContextInjector([injector1, injector2])

        assert len(composite.injectors) == 2
        assert composite.injectors[0] is injector1
        assert composite.injectors[1] is injector2

    @patch("pharia_telemetry.logging.injectors.trace.get_current_span")
    @patch("pharia_telemetry.logging.injectors.baggage.get_all")
    def test_inject_combines_all(self, mock_get_all: Mock, mock_get_span: Mock) -> None:
        """Test that composite injector combines all contexts."""
        # Mock trace context
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x12345678901234567890123456789012
        mock_span_context.span_id = 0x1234567890123456
        mock_span.get_span_context.return_value = mock_span_context
        mock_get_span.return_value = mock_span

        # Mock baggage context
        mock_get_all.return_value = {
            BaggageKeys.USER_ID: "user123",
        }

        trace_injector = TraceContextInjector()
        baggage_injector = BaggageContextInjector()
        composite = CompositeContextInjector([trace_injector, baggage_injector])

        log_dict = {"message": "test"}
        result = composite.inject(log_dict)

        assert result["message"] == "test"
        assert result["trace_id"] == "12345678901234567890123456789012"
        assert result["span_id"] == "1234567890123456"
        assert result[BaggageKeys.USER_ID] == "user123"

    def test_inject_preserves_original_dict(self) -> None:
        """Test that injection preserves the original dictionary."""
        injector1 = TraceContextInjector(include_trace_id=False, include_span_id=False)
        composite = CompositeContextInjector([injector1])

        original_dict = {"message": "test", "existing": "value"}
        result = composite.inject(original_dict)

        # Original dict should be unchanged
        assert original_dict == {"message": "test", "existing": "value"}
        # Result should include original values
        assert result["message"] == "test"
        assert result["existing"] == "value"


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_trace_injector(self) -> None:
        """Test create_trace_injector factory."""
        injector = create_trace_injector(
            include_trace_id=False,
            trace_id_key="custom_trace",
        )

        assert isinstance(injector, TraceContextInjector)
        assert injector.include_trace_id is False
        assert injector.trace_id_key == "custom_trace"

    def test_create_baggage_injector(self) -> None:
        """Test create_baggage_injector factory."""
        exclude_keys = {"sensitive"}
        injector = create_baggage_injector(
            prefix_filter="app.",
            exclude_keys=exclude_keys,
        )

        assert isinstance(injector, BaggageContextInjector)
        assert injector.prefix_filter == "app."
        assert injector.exclude_keys == exclude_keys

    def test_create_full_context_injector(self) -> None:
        """Test create_full_context_injector factory."""
        injector = create_full_context_injector(
            include_trace_id=True,
            include_span_id=False,
            include_baggage=True,
            baggage_prefix_filter="app.",
        )

        assert isinstance(injector, CompositeContextInjector)
        assert len(injector.injectors) == 2  # trace + baggage

    def test_create_full_context_injector_trace_only(self) -> None:
        """Test create_full_context_injector with trace only."""
        injector = create_full_context_injector(
            include_trace_id=True,
            include_span_id=True,
            include_baggage=False,
        )

        assert isinstance(injector, CompositeContextInjector)
        assert len(injector.injectors) == 1  # trace only

    def test_create_full_context_injector_baggage_only(self) -> None:
        """Test create_full_context_injector with baggage only."""
        injector = create_full_context_injector(
            include_trace_id=False,
            include_span_id=False,
            include_baggage=True,
        )

        assert isinstance(injector, CompositeContextInjector)
        assert len(injector.injectors) == 1  # baggage only
