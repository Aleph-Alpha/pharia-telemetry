"""Tests for structlog utilities and correlation functionality."""

import pytest


class TestStructlogIntegration:
    """Test integration with structlog using the injectors directly."""

    def test_injectors_in_structlog_chain(self) -> None:
        """Test that injectors work correctly in a structlog processor chain."""
        import structlog

        from pharia_telemetry.logging import create_full_context_injector

        # Create a simple structlog processor using our injectors
        class ContextProcessor:
            def __init__(self) -> None:
                self.injector = create_full_context_injector(
                    include_span_id=False, baggage_prefix_filter="app."
                )

            def __call__(self, logger, method_name, event_dict):
                return self.injector.inject(event_dict)

        # Configure structlog with our processor
        structlog.configure(
            processors=[
                ContextProcessor(),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        logger = structlog.get_logger()

        # This should work without errors (even without OpenTelemetry)
        try:
            logger.info("Test message", extra_field="extra_value")
            # If we get here without exception, the integration works
            assert True
        except Exception as e:
            pytest.fail(f"Structlog integration failed: {e}")


class TestUsageExamples:
    """Test realistic usage examples."""

    def test_framework_agnostic_usage(self) -> None:
        """Test using injectors directly for framework-agnostic logging."""
        from pharia_telemetry.logging import (
            BaggageContextInjector,
            CompositeContextInjector,
            TraceContextInjector,
        )

        # Create individual injectors
        trace_injector = TraceContextInjector(include_span_id=False)
        baggage_injector = BaggageContextInjector(prefix_filter="app.")

        # Use individually
        log_dict = {"message": "test", "level": "info"}
        result = trace_injector.inject(log_dict)
        result = baggage_injector.inject(result)

        assert result["message"] == "test"

        # Or use composite
        composite = CompositeContextInjector([trace_injector, baggage_injector])
        result = composite.inject({"message": "composite test"})

        assert result["message"] == "composite test"

    def test_factory_usage(self) -> None:
        """Test using factory functions for common configurations."""
        from pharia_telemetry.logging import (
            create_baggage_injector,
            create_full_context_injector,
            create_trace_injector,
        )

        # Individual factories
        trace_injector = create_trace_injector(include_span_id=False)
        baggage_injector = create_baggage_injector(prefix_filter="app.")

        # Full context factory
        full_injector = create_full_context_injector(
            include_span_id=False,
            baggage_prefix_filter="app.",
        )

        # All should work without errors
        log_dict = {"message": "factory test"}
        result1 = trace_injector.inject(log_dict)
        result2 = baggage_injector.inject(log_dict)
        result3 = full_injector.inject(log_dict)

        assert result1["message"] == "factory test"
        assert result2["message"] == "factory test"
        assert result3["message"] == "factory test"

    def test_structlog_custom_processor(self) -> None:
        """Test creating a custom structlog processor using injectors."""
        from pharia_telemetry.logging import create_full_context_injector

        class CustomStructlogProcessor:
            """Example of how users can create their own structlog processor."""

            def __init__(self, **kwargs) -> None:
                self.injector = create_full_context_injector(**kwargs)

            def __call__(self, logger, method_name, event_dict):
                return self.injector.inject(event_dict)

        # Test the custom processor
        processor = CustomStructlogProcessor(
            include_span_id=False, baggage_prefix_filter="app."
        )

        event_dict = {"message": "custom processor test"}
        result = processor(None, "info", event_dict)

        assert result["message"] == "custom processor test"
        # Processor should work even without OpenTelemetry configured
