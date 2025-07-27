"""
Generic OpenTelemetry setup utilities.

This module provides utilities for setting up OpenTelemetry tracing in any Python application,
including propagator configuration, tracer provider setup, and basic instrumentation.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

try:
    from opentelemetry import propagate, trace
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.exporter.otlp.proto.grpc import (  # type: ignore[import-not-found]
        OTLPSpanExporter,
    )
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available - tracing will be disabled")


def setup_otel_propagators() -> None:
    """
    Configure OpenTelemetry propagators to support trace context and baggage.

    Sets up W3C TraceContext and W3C Baggage propagators for cross-service communication.
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - skipping propagator setup")
        return

    propagators = [
        TraceContextTextMapPropagator(),  # W3C TraceContext
        W3CBaggagePropagator(),  # W3C Baggage for user context propagation
    ]

    composite_propagator = CompositePropagator(propagators)
    propagate.set_global_textmap(composite_propagator)

    logger.debug("Configured OpenTelemetry propagators: TraceContext, Baggage")


def create_tracer_provider(
    service_name: str,
    service_version: str | None = None,
    service_instance_id: str | None = None,
    deployment_environment: str | None = None,
    additional_resource_attributes: dict[str, str] | None = None,
) -> TracerProvider | None:
    """
    Create and configure an OpenTelemetry TracerProvider.

    Args:
        service_name: Name of the service
        service_version: Version of the service (optional)
        service_instance_id: Instance ID (optional, defaults to HOSTNAME)
        deployment_environment: Deployment environment (optional, defaults to ENVIRONMENT env var)
        additional_resource_attributes: Additional resource attributes

    Returns:
        TracerProvider if successful, None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - skipping tracer provider setup")
        return None

    # Build resource attributes
    resource_attrs = {
        "service.name": service_name,
    }

    if service_version:
        resource_attrs["service.version"] = service_version

    if service_instance_id:
        resource_attrs["service.instance.id"] = service_instance_id
    else:
        resource_attrs["service.instance.id"] = os.getenv("HOSTNAME", "unknown")

    if deployment_environment:
        resource_attrs["deployment.environment"] = deployment_environment
    else:
        resource_attrs["deployment.environment"] = os.getenv("ENVIRONMENT", "unknown")

    if additional_resource_attributes:
        resource_attrs.update(additional_resource_attributes)

    # Create resource
    resource = Resource(attributes=resource_attrs)
    provider = TracerProvider(resource=resource)

    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry TracerProvider configured for service: %s", service_name)

    return provider


def add_baggage_span_processor(tracer_provider: TracerProvider) -> None:
    """
    Add BaggageSpanProcessor to a tracer provider.

    Args:
        tracer_provider: The tracer provider to add the processor to
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - skipping baggage processor")
        return

    try:
        from pharia_telemetry.baggage.processors import BaggageSpanProcessor

        baggage_processor = BaggageSpanProcessor()
        tracer_provider.add_span_processor(baggage_processor)
        logger.debug("BaggageSpanProcessor added to tracer provider")
    except Exception as e:
        logger.error("Failed to add BaggageSpanProcessor: %s", e)


def add_otlp_exporter(
    tracer_provider: TracerProvider,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """
    Add OTLP exporter to a tracer provider.

    Args:
        tracer_provider: The tracer provider to add the exporter to
        endpoint: OTLP endpoint URL (defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var)
        headers: OTLP headers (defaults to OTEL_EXPORTER_OTLP_HEADERS env var)
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - skipping OTLP exporter")
        return

    try:
        # Parse headers from environment if not provided
        if headers is None:
            headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            headers = {}
            if headers_str:
                for header in headers_str.split(","):
                    if "=" in header:
                        key, value = header.split("=", 1)
                        # URL-decode the header value to handle %20 -> space conversion
                        headers[key.strip()] = unquote(value.strip())

        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers if headers else None,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.debug("OTLP exporter added to tracer provider")
    except Exception as e:
        logger.error("Failed to add OTLP exporter: %s", e)


def add_console_exporter(tracer_provider: TracerProvider) -> None:
    """
    Add console exporter to a tracer provider for development/debugging.

    Args:
        tracer_provider: The tracer provider to add the exporter to
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - skipping console exporter")
        return

    try:
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)
        logger.debug("Console exporter added to tracer provider")
    except Exception as e:
        logger.error("Failed to add console exporter: %s", e)


def setup_basic_tracing(
    service_name: str,
    service_version: str | None = None,
    enable_baggage_processor: bool = True,
    enable_otlp_exporter: bool = True,
    enable_console_exporter: bool | None = None,
    additional_resource_attributes: dict[str, str] | None = None,
) -> TracerProvider | None:
    """
    Set up basic OpenTelemetry tracing with common configuration.

    This is a convenience function that sets up:
    - Propagators (TraceContext + Baggage)
    - TracerProvider with resource attributes
    - BaggageSpanProcessor (optional)
    - OTLP exporter (optional)
    - Console exporter (optional, auto-enabled in development)

    Args:
        service_name: Name of the service
        service_version: Version of the service
        enable_baggage_processor: Whether to add BaggageSpanProcessor
        enable_otlp_exporter: Whether to add OTLP exporter
        enable_console_exporter: Whether to add console exporter (auto-detect if None)
        additional_resource_attributes: Additional resource attributes

    Returns:
        TracerProvider if successful, None if OpenTelemetry is not available
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - tracing disabled")
        return None

    try:
        # Setup propagators
        setup_otel_propagators()

        # Create tracer provider
        provider = create_tracer_provider(
            service_name=service_name,
            service_version=service_version,
            additional_resource_attributes=additional_resource_attributes,
        )

        if provider is None:
            return None

        # Add baggage processor
        if enable_baggage_processor:
            add_baggage_span_processor(provider)

        # Add OTLP exporter
        if enable_otlp_exporter:
            add_otlp_exporter(provider)

        # Add console exporter (auto-detect if not specified)
        if enable_console_exporter is None:
            is_debug_logging = os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
            is_console_exporter = os.getenv("OTEL_TRACES_EXPORTER") == "console"
            is_dev_environment = os.getenv("ENVIRONMENT") == "development"
            enable_console_exporter = (
                is_debug_logging or is_console_exporter or is_dev_environment
            )

        if enable_console_exporter:
            add_console_exporter(provider)
            logger.info("Console trace exporter enabled for development/debugging")

        logger.info("Basic OpenTelemetry tracing setup complete for %s", service_name)
        return provider

    except Exception as e:
        logger.error("Failed to setup basic tracing: %s", e)
        return None


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

    return trace.get_tracer(name)
