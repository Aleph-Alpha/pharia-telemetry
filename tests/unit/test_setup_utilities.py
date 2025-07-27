"""
Tests for OpenTelemetry setup utilities and configuration.
"""

from unittest.mock import Mock, patch

from pharia_telemetry.utils.setup import (
    add_baggage_span_processor,
    add_console_exporter,
    add_otlp_exporter,
    create_tracer_provider,
    get_tracer,
    setup_basic_tracing,
    setup_otel_propagators,
)


class TestSetupOtelPropagators:
    """Test setup_otel_propagators function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.propagate", create=True)
    @patch("pharia_telemetry.utils.setup.CompositePropagator", create=True)
    @patch("pharia_telemetry.utils.setup.TraceContextTextMapPropagator", create=True)
    @patch("pharia_telemetry.utils.setup.W3CBaggagePropagator", create=True)
    def test_setup_otel_propagators(
        self,
        mock_baggage_prop: Mock,
        mock_trace_prop: Mock,
        mock_composite: Mock,
        mock_propagate: Mock,
    ) -> None:
        """Test that OpenTelemetry propagators are configured correctly."""
        # Setup mocks
        mock_composite_instance = Mock()
        mock_composite.return_value = mock_composite_instance

        # Test the function
        setup_otel_propagators()

        # Verify propagators were created
        mock_trace_prop.assert_called_once()
        mock_baggage_prop.assert_called_once()
        mock_composite.assert_called_once()

        # Verify propagator was set up
        mock_propagate.set_global_textmap.assert_called_once_with(
            mock_composite_instance
        )

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_setup_otel_propagators_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        setup_otel_propagators()
        mock_logger.warning.assert_called_once()


class TestCreateTracerProvider:
    """Test create_tracer_provider function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.Resource", create=True)
    @patch("pharia_telemetry.utils.setup.TracerProvider", create=True)
    @patch("pharia_telemetry.utils.setup.trace", create=True)
    @patch.dict(
        "os.environ",
        {
            "HOSTNAME": "test-host",
            "ENVIRONMENT": "test",
        },
        clear=False,
    )
    def test_create_tracer_provider_success(
        self,
        mock_trace: Mock,
        mock_tracer_provider: Mock,
        mock_resource: Mock,
    ) -> None:
        """Test successful tracer provider creation."""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance

        # Test the function
        result = create_tracer_provider(
            service_name="test-service",
            service_version="1.0.0",
            deployment_environment="production",
        )

        # Verify resource creation
        mock_resource.assert_called_once_with(
            attributes={
                "service.name": "test-service",
                "service.version": "1.0.0",
                "service.instance.id": "test-host",
                "deployment.environment": "production",
            }
        )

        # Verify tracer provider creation
        mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)

        # Verify tracer provider is set globally
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider_instance)

        # Verify return value
        assert result == mock_provider_instance

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_create_tracer_provider_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        result = create_tracer_provider("test-service")
        assert result is None
        mock_logger.warning.assert_called_once()

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.Resource", create=True)
    @patch("pharia_telemetry.utils.setup.TracerProvider", create=True)
    @patch("pharia_telemetry.utils.setup.trace", create=True)
    def test_create_tracer_provider_with_additional_attributes(
        self,
        mock_trace: Mock,
        mock_tracer_provider: Mock,
        mock_resource: Mock,
    ) -> None:
        """Test tracer provider creation with additional resource attributes."""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance

        additional_attrs = {"custom.attribute": "custom_value"}

        # Test the function
        create_tracer_provider(
            service_name="test-service",
            additional_resource_attributes=additional_attrs,
        )

        # Verify additional attributes were included
        call_args = mock_resource.call_args[1]["attributes"]
        assert call_args["custom.attribute"] == "custom_value"
        assert call_args["service.name"] == "test-service"


class TestAddBaggageSpanProcessor:
    """Test add_baggage_span_processor function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    def test_add_baggage_span_processor(self) -> None:
        """Test adding BaggageSpanProcessor to tracer provider."""
        mock_provider = Mock()

        # Test the function
        add_baggage_span_processor(mock_provider)

        # Verify processor was added
        mock_provider.add_span_processor.assert_called_once()
        processor = mock_provider.add_span_processor.call_args[0][0]
        assert "BaggageSpanProcessor" in str(type(processor))

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_add_baggage_span_processor_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        mock_provider = Mock()
        add_baggage_span_processor(mock_provider)
        mock_logger.warning.assert_called_once()


class TestAddOtlpExporter:
    """Test add_otlp_exporter function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.OTLPSpanExporter", create=True)
    @patch("pharia_telemetry.utils.setup.BatchSpanProcessor", create=True)
    def test_add_otlp_exporter(
        self, mock_batch_processor: Mock, mock_otlp_exporter: Mock
    ) -> None:
        """Test adding OTLP exporter to tracer provider."""
        mock_provider = Mock()
        mock_exporter_instance = Mock()
        mock_otlp_exporter.return_value = mock_exporter_instance
        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance

        # Test the function
        add_otlp_exporter(mock_provider)

        # Verify exporter and processor creation
        mock_otlp_exporter.assert_called_once_with(endpoint=None, headers=None)
        mock_batch_processor.assert_called_once_with(mock_exporter_instance)
        mock_provider.add_span_processor.assert_called_once_with(
            mock_processor_instance
        )

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.OTLPSpanExporter", create=True)
    @patch("pharia_telemetry.utils.setup.BatchSpanProcessor", create=True)
    @patch.dict(
        "os.environ",
        {"OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer%20token,X-Custom=value"},
        clear=False,
    )
    def test_add_otlp_exporter_with_env_headers(
        self, mock_batch_processor: Mock, mock_otlp_exporter: Mock
    ) -> None:
        """Test OTLP exporter with headers from environment."""
        mock_provider = Mock()
        mock_exporter_instance = Mock()
        mock_otlp_exporter.return_value = mock_exporter_instance

        # Test the function
        add_otlp_exporter(mock_provider)

        # Verify headers were parsed and URL-decoded
        call_args = mock_otlp_exporter.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer token"  # %20 decoded to space
        assert headers["X-Custom"] == "value"

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_add_otlp_exporter_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        mock_provider = Mock()
        add_otlp_exporter(mock_provider)
        mock_logger.warning.assert_called_once()


class TestAddConsoleExporter:
    """Test add_console_exporter function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.ConsoleSpanExporter", create=True)
    @patch("pharia_telemetry.utils.setup.BatchSpanProcessor", create=True)
    def test_add_console_exporter(
        self, mock_batch_processor: Mock, mock_console_exporter: Mock
    ) -> None:
        """Test adding console exporter to tracer provider."""
        mock_provider = Mock()
        mock_exporter_instance = Mock()
        mock_console_exporter.return_value = mock_exporter_instance
        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance

        # Test the function
        add_console_exporter(mock_provider)

        # Verify exporter and processor creation
        mock_console_exporter.assert_called_once()
        mock_batch_processor.assert_called_once_with(mock_exporter_instance)
        mock_provider.add_span_processor.assert_called_once_with(
            mock_processor_instance
        )

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_add_console_exporter_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        mock_provider = Mock()
        add_console_exporter(mock_provider)
        mock_logger.warning.assert_called_once()


class TestSetupBasicTracing:
    """Test setup_basic_tracing function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.setup_otel_propagators")
    @patch("pharia_telemetry.utils.setup.create_tracer_provider")
    @patch("pharia_telemetry.utils.setup.add_baggage_span_processor")
    @patch("pharia_telemetry.utils.setup.add_otlp_exporter")
    @patch("pharia_telemetry.utils.setup.add_console_exporter")
    @patch.dict(
        "os.environ",
        {"ENVIRONMENT": "production"},  # Should not enable console exporter
        clear=False,
    )
    def test_setup_basic_tracing_success(
        self,
        mock_add_console: Mock,
        mock_add_otlp: Mock,
        mock_add_baggage: Mock,
        mock_create_provider: Mock,
        mock_setup_propagators: Mock,
    ) -> None:
        """Test successful basic tracing setup."""
        # Setup mocks
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Test the function
        result = setup_basic_tracing("test-service", service_version="1.0.0")

        # Verify all setup functions are called in order
        mock_setup_propagators.assert_called_once()
        mock_create_provider.assert_called_once_with(
            service_name="test-service",
            service_version="1.0.0",
            additional_resource_attributes=None,
        )
        mock_add_baggage.assert_called_once_with(mock_provider)
        mock_add_otlp.assert_called_once_with(mock_provider)

        # Console exporter should NOT be enabled in production
        mock_add_console.assert_not_called()

        assert result == mock_provider

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.setup_otel_propagators")
    @patch("pharia_telemetry.utils.setup.create_tracer_provider")
    @patch("pharia_telemetry.utils.setup.add_baggage_span_processor")
    @patch("pharia_telemetry.utils.setup.add_otlp_exporter")
    @patch("pharia_telemetry.utils.setup.add_console_exporter")
    @patch.dict(
        "os.environ",
        {"ENVIRONMENT": "development"},  # Should enable console exporter
        clear=False,
    )
    def test_setup_basic_tracing_with_console_exporter(
        self,
        mock_add_console: Mock,
        mock_add_otlp: Mock,
        mock_add_baggage: Mock,
        mock_create_provider: Mock,
        mock_setup_propagators: Mock,
    ) -> None:
        """Test basic tracing setup with console exporter in development."""
        # Setup mocks
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Test the function
        setup_basic_tracing("test-service")

        # Console exporter should be enabled in development
        mock_add_console.assert_called_once_with(mock_provider)

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.setup_otel_propagators")
    @patch("pharia_telemetry.utils.setup.create_tracer_provider")
    def test_setup_basic_tracing_no_provider(
        self, mock_create_provider: Mock, mock_setup_propagators: Mock
    ) -> None:
        """Test setup when tracer provider returns None."""
        # Setup mocks
        mock_create_provider.return_value = None

        result = setup_basic_tracing("test-service")

        # Verify propagators are set up but result is None
        mock_setup_propagators.assert_called_once()
        mock_create_provider.assert_called_once()
        assert result is None

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.setup_otel_propagators")
    @patch("pharia_telemetry.utils.setup.create_tracer_provider")
    def test_setup_basic_tracing_with_disabled_features(
        self, mock_create_provider: Mock, mock_setup_propagators: Mock
    ) -> None:
        """Test setup with disabled features."""
        # Setup mocks
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Test the function with disabled features
        result = setup_basic_tracing(
            "test-service",
            enable_baggage_processor=False,
            enable_otlp_exporter=False,
            enable_console_exporter=False,
        )

        # Verify only basic setup is called
        mock_setup_propagators.assert_called_once()
        mock_create_provider.assert_called_once()
        assert result == mock_provider

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_setup_basic_tracing_not_available(self, mock_logger: Mock) -> None:
        """Test behavior when OpenTelemetry is not available."""
        result = setup_basic_tracing("test-service")
        assert result is None
        mock_logger.warning.assert_called_once()


class TestGetTracer:
    """Test get_tracer function."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.trace")
    def test_get_tracer_default(self, mock_trace: Mock) -> None:
        """Test getting tracer with default name."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        result = get_tracer()

        mock_trace.get_tracer.assert_called_once_with("pharia_telemetry.utils.setup")
        assert result == mock_tracer

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.trace")
    def test_get_tracer_custom_name(self, mock_trace: Mock) -> None:
        """Test getting tracer with custom name."""
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        result = get_tracer("custom.tracer")

        mock_trace.get_tracer.assert_called_once_with("custom.tracer")
        assert result == mock_tracer

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", False)
    def test_get_tracer_not_available(self) -> None:
        """Test behavior when OpenTelemetry is not available."""
        result = get_tracer()
        assert result is None


class TestErrorHandling:
    """Test error handling in setup utilities."""

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.OTLPSpanExporter", create=True)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_add_otlp_exporter_handles_exceptions(
        self, mock_logger: Mock, mock_otlp_exporter: Mock
    ) -> None:
        """Test that exceptions in add_otlp_exporter are handled."""
        mock_provider = Mock()
        mock_otlp_exporter.side_effect = Exception("OTLP error")

        add_otlp_exporter(mock_provider)

        mock_logger.error.assert_called_once()

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.ConsoleSpanExporter", create=True)
    @patch("pharia_telemetry.utils.setup.logger")
    def test_add_console_exporter_handles_exceptions(
        self, mock_logger: Mock, mock_console_exporter: Mock
    ) -> None:
        """Test that exceptions in add_console_exporter are handled."""
        mock_provider = Mock()
        mock_console_exporter.side_effect = Exception("Console error")

        add_console_exporter(mock_provider)

        mock_logger.error.assert_called_once()

    @patch("pharia_telemetry.utils.setup.OTEL_AVAILABLE", True)
    @patch("pharia_telemetry.utils.setup.setup_otel_propagators")
    @patch("pharia_telemetry.utils.setup.logger")
    def test_setup_basic_tracing_handles_exceptions(
        self, mock_logger: Mock, mock_setup_propagators: Mock
    ) -> None:
        """Test that exceptions in setup_basic_tracing are handled."""
        mock_setup_propagators.side_effect = Exception("Setup error")

        result = setup_basic_tracing("test-service")

        assert result is None
        mock_logger.error.assert_called_once()
