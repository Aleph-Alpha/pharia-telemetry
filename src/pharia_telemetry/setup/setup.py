"""
OpenTelemetry setup utilities for Pharia services.

This module provides a single, consolidated setup function for OpenTelemetry tracing.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

try:
    import opentelemetry  # noqa: F401

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - tracing will be disabled")


def _setup_propagators() -> None:
    """Configure W3C TraceContext and Baggage propagators."""
    from opentelemetry import propagate
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    propagate.set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )
    logger.debug("Configured OpenTelemetry propagators: TraceContext, Baggage")


def _build_resource_attrs(
    service_name: str,
    service_version: str | None,
    environment: str | None,
) -> dict[str, str]:
    """Build the OTel resource attributes dictionary."""
    attrs: dict[str, str] = {
        "service.name": service_name,
        "service.instance.id": os.getenv("HOSTNAME") or "unknown",
        "deployment.environment": environment or os.getenv("ENVIRONMENT") or "unknown",
    }
    if service_version:
        attrs["service.version"] = service_version
    return attrs


def _create_tracer_provider(resource_attrs: dict[str, str]) -> TracerProvider:
    """Create and register a TracerProvider with the given resource attributes."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource(attributes=resource_attrs))
    trace.set_tracer_provider(provider)
    return provider


def _add_baggage_processor(provider: TracerProvider) -> None:
    """Add the BaggageSpanProcessor to the provider."""
    try:
        from pharia_telemetry.baggage.processors import BaggageSpanProcessor

        provider.add_span_processor(BaggageSpanProcessor())
        logger.debug("BaggageSpanProcessor added to tracer provider")
    except Exception as e:
        logger.error("Failed to add BaggageSpanProcessor: %s", e)


def _parse_otlp_headers() -> dict[str, str] | None:
    """Parse OTEL_EXPORTER_OTLP_HEADERS into a dict, or return None if unset."""
    headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if not headers_str:
        return None

    headers: dict[str, str] = {}
    for header in headers_str.split(","):
        if "=" in header:
            key, value = header.split("=", 1)
            headers[key.strip()] = unquote(value.strip())
    return headers or None


def _add_otlp_exporter(provider: TracerProvider) -> None:
    """Add the OTLP span exporter based on protocol environment variables."""
    otlp_protocol = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
    )

    if otlp_protocol == "grpc":
        pip_package = "opentelemetry-exporter-otlp-proto-grpc"
    else:
        pip_package = "opentelemetry-exporter-otlp-proto-http"

    try:
        if otlp_protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )

        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            headers=_parse_otlp_headers(),
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.debug("OTLP %s exporter added to tracer provider", otlp_protocol)
    except ImportError:
        logger.warning(
            "OTLP exporter for protocol '%s' not available. "
            "Install '%s' to enable trace export.",
            otlp_protocol,
            pip_package,
        )
    except Exception as e:
        logger.warning("Failed to add OTLP exporter: %s", e)


def _should_enable_console_exporter(environment: str | None) -> bool:
    """Determine whether the console exporter should be enabled via auto-detection."""
    is_debug_logging = os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
    is_console_exporter = os.getenv("OTEL_TRACES_EXPORTER") == "console"
    is_dev_environment = (
        environment == "development" or os.getenv("ENVIRONMENT") == "development"
    )
    return is_debug_logging or is_console_exporter or is_dev_environment


def _add_console_exporter(provider: TracerProvider) -> None:
    """Add a console span exporter to the provider."""
    try:
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console trace exporter enabled for development/debugging")
    except Exception as e:
        logger.error("Failed to add console exporter: %s", e)


def setup_telemetry(
    service_name: str,
    service_version: str | None = None,
    environment: str | None = None,
    enable_baggage_processor: bool = True,
    enable_console_exporter: bool | None = None,
) -> bool:
    """
    Set up OpenTelemetry tracing with sensible defaults for Pharia services.

    NOTE: If your application uses Pydantic Logfire, use Pydantic Logfire.configure() instead.
    This function is primarily for applications that don't use Pydantic Logfire.

    This is the main entry point for most applications - it configures everything
    needed for distributed tracing and context propagation.

    Args:
        service_name: Name of the service (required)
        service_version: Version of the service (optional)
        environment: Environment name (dev, staging, prod) (optional)
        enable_baggage_processor: Whether to enable automatic baggage->span attributes (default: True)
        enable_console_exporter: Whether to enable console output for debugging (default: auto-detect)

    Returns:
        True if setup succeeded, False if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - tracing disabled")
        return False

    try:
        _setup_propagators()

        resource_attrs = _build_resource_attrs(
            service_name, service_version, environment
        )
        provider = _create_tracer_provider(resource_attrs)
        logger.info(
            "OpenTelemetry TracerProvider configured for service: %s", service_name
        )

        if enable_baggage_processor:
            _add_baggage_processor(provider)

        _add_otlp_exporter(provider)

        if enable_console_exporter is None:
            enable_console_exporter = _should_enable_console_exporter(environment)

        if enable_console_exporter:
            _add_console_exporter(provider)

        logger.info("OpenTelemetry tracing setup complete for %s", service_name)
        return True

    except Exception as e:
        logger.error("Failed to setup telemetry: %s", e)
        return False


def get_tracer(name: str = __name__) -> trace.Tracer | None:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name for the tracer, defaults to current module

    Returns:
        OpenTelemetry Tracer instance, or None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not available - returning None tracer")
        return None

    from opentelemetry import trace

    return trace.get_tracer(name)
