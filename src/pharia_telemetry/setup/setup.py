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

logger = logging.getLogger(__name__)

try:
    import opentelemetry  # noqa: F401

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - tracing will be disabled")


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
        # Setup propagators
        from opentelemetry import propagate
        from opentelemetry.baggage.propagation import W3CBaggagePropagator
        from opentelemetry.propagators.composite import CompositePropagator
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        propagators = [
            TraceContextTextMapPropagator(),  # W3C TraceContext
            W3CBaggagePropagator(),  # W3C Baggage for user context propagation
        ]
        composite_propagator = CompositePropagator(propagators)
        propagate.set_global_textmap(composite_propagator)
        logger.debug("Configured OpenTelemetry propagators: TraceContext, Baggage")

        # Build resource attributes (ensure all values are strings, not None)
        resource_attrs: dict[str, str] = {
            "service.name": service_name,
            "service.instance.id": os.getenv("HOSTNAME") or "unknown",
            "deployment.environment": environment
            or os.getenv("ENVIRONMENT")
            or "unknown",
        }

        if service_version:
            resource_attrs["service.version"] = service_version

        # Create tracer provider
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource(attributes=resource_attrs)
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        logger.info(
            "OpenTelemetry TracerProvider configured for service: %s", service_name
        )

        # Add baggage processor
        if enable_baggage_processor:
            try:
                from pharia_telemetry.baggage.processors import BaggageSpanProcessor

                baggage_processor = BaggageSpanProcessor()
                provider.add_span_processor(baggage_processor)
                logger.debug("BaggageSpanProcessor added to tracer provider")
            except Exception as e:
                logger.error("Failed to add BaggageSpanProcessor: %s", e)

        # Add OTLP exporter (always enabled by default)
        try:
            from opentelemetry.exporter.otlp.proto.grpc import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Parse headers from environment
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            headers = {}
            if headers_str:
                for header in headers_str.split(","):
                    if "=" in header:
                        key, value = header.split("=", 1)
                        headers[key.strip()] = unquote(value.strip())

            otlp_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                headers=headers if headers else None,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.debug("OTLP exporter added to tracer provider")
        except Exception as e:
            logger.error("Failed to add OTLP exporter: %s", e)

        # Add console exporter (auto-detect if not explicitly set)
        if enable_console_exporter is None:
            is_debug_logging = os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
            is_console_exporter = os.getenv("OTEL_TRACES_EXPORTER") == "console"
            is_dev_environment = (
                environment == "development"
                or os.getenv("ENVIRONMENT") == "development"
            )
            enable_console_exporter = (
                is_debug_logging or is_console_exporter or is_dev_environment
            )

        if enable_console_exporter:
            try:
                from opentelemetry.sdk.trace.export import (
                    BatchSpanProcessor,
                    ConsoleSpanExporter,
                )

                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(console_processor)
                logger.info("Console trace exporter enabled for development/debugging")
            except Exception as e:
                logger.error("Failed to add console exporter: %s", e)

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
