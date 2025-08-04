"""Tests for structlog processors."""

import pytest

from pharia_telemetry.logging.processor import (
    StructlogBaggageProcessor,
    StructlogCompositeProcessor,
    StructlogTraceProcessor,
    create_structlog_baggage_processor,
    create_structlog_full_context_processor,
    create_structlog_trace_processor,
)


class TestStructlogProcessors:
    """Test structlog processor classes."""

    def test_structlog_trace_processor_init(self):
        """Test StructlogTraceProcessor initialization."""
        processor = StructlogTraceProcessor()
        assert processor.injector is not None

    def test_structlog_trace_processor_call(self):
        """Test StructlogTraceProcessor __call__ method."""
        processor = StructlogTraceProcessor()
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        assert isinstance(result, dict)
        assert "message" in result

    def test_structlog_baggage_processor_init(self):
        """Test StructlogBaggageProcessor initialization."""
        processor = StructlogBaggageProcessor()
        assert processor.injector is not None

    def test_structlog_baggage_processor_call(self):
        """Test StructlogBaggageProcessor __call__ method."""
        processor = StructlogBaggageProcessor()
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        assert isinstance(result, dict)
        assert "message" in result

    def test_structlog_composite_processor_init(self):
        """Test StructlogCompositeProcessor initialization."""
        from pharia_telemetry.logging.injectors import TraceContextInjector

        injectors = [TraceContextInjector()]
        processor = StructlogCompositeProcessor(injectors)
        assert processor.injector is not None

    def test_structlog_composite_processor_call(self):
        """Test StructlogCompositeProcessor __call__ method."""
        from pharia_telemetry.logging.injectors import TraceContextInjector

        injectors = [TraceContextInjector()]
        processor = StructlogCompositeProcessor(injectors)
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        assert isinstance(result, dict)
        assert "message" in result


class TestStructlogFactories:
    """Test structlog factory functions."""

    def test_create_structlog_trace_processor(self):
        """Test create_structlog_trace_processor factory."""
        # Skip if structlog not available
        try:
            processor = create_structlog_trace_processor()
            assert isinstance(processor, StructlogTraceProcessor)
        except ImportError:
            pytest.skip("structlog not available")

    def test_create_structlog_baggage_processor(self):
        """Test create_structlog_baggage_processor factory."""
        # Skip if structlog not available
        try:
            processor = create_structlog_baggage_processor()
            assert isinstance(processor, StructlogBaggageProcessor)
        except ImportError:
            pytest.skip("structlog not available")

    def test_create_structlog_full_context_processor(self):
        """Test create_structlog_full_context_processor factory."""
        # Skip if structlog not available
        try:
            processor = create_structlog_full_context_processor()
            assert isinstance(processor, StructlogCompositeProcessor)
        except ImportError:
            pytest.skip("structlog not available")

    def test_factory_functions_require_structlog(self):
        """Test that factory functions check for structlog availability."""
        # Mock STRUCTLOG_AVAILABLE as False
        import pharia_telemetry.logging.processor as processor_module

        original_value = processor_module.STRUCTLOG_AVAILABLE
        processor_module.STRUCTLOG_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="structlog is not available"):
                create_structlog_trace_processor()

            with pytest.raises(ImportError, match="structlog is not available"):
                create_structlog_baggage_processor()

            with pytest.raises(ImportError, match="structlog is not available"):
                create_structlog_full_context_processor()
        finally:
            # Restore original value
            processor_module.STRUCTLOG_AVAILABLE = original_value
