"""Tests for OTLP exporter protocol selection in setup_telemetry."""

from __future__ import annotations

import builtins
import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Reset the global tracer provider after each test."""
    from opentelemetry import trace

    yield
    trace.set_tracer_provider(trace._PROXY_TRACER_PROVIDER)  # type: ignore[attr-defined]


def _make_import_blocker(blocked_modules: set[str]):
    """Return a side_effect for builtins.__import__ that blocks specific modules."""
    _real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        for blocked in blocked_modules:
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"Mocked: No module named '{name}'")
        return _real_import(name, *args, **kwargs)

    return _import


class TestOtlpExporterProtocolSelection:
    """Test that setup_telemetry selects the correct OTLP exporter based on env vars."""

    def test_default_protocol_is_http(self, monkeypatch):
        """When no protocol env var is set, http/protobuf is used by default."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        mock_exporter_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OTLPSpanExporter = mock_exporter_cls

        with patch.dict(
            "sys.modules",
            {"opentelemetry.exporter.otlp.proto.http.trace_exporter": mock_module},
        ):
            from pharia_telemetry.setup.setup import setup_telemetry

            result = setup_telemetry(
                "test-svc", enable_console_exporter=False, enable_baggage_processor=False
            )

        assert result is True
        mock_exporter_cls.assert_called_once()

    def test_grpc_protocol_env_var(self, monkeypatch):
        """When OTEL_EXPORTER_OTLP_PROTOCOL=grpc, the gRPC exporter is used."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        mock_exporter_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OTLPSpanExporter = mock_exporter_cls

        with patch.dict(
            "sys.modules",
            {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": mock_module},
        ):
            from pharia_telemetry.setup.setup import setup_telemetry

            result = setup_telemetry(
                "test-svc", enable_console_exporter=False, enable_baggage_processor=False
            )

        assert result is True
        mock_exporter_cls.assert_called_once()

    def test_traces_protocol_takes_precedence(self, monkeypatch):
        """OTEL_EXPORTER_OTLP_TRACES_PROTOCOL takes precedence over OTEL_EXPORTER_OTLP_PROTOCOL."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")

        mock_exporter_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OTLPSpanExporter = mock_exporter_cls

        with patch.dict(
            "sys.modules",
            {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": mock_module},
        ):
            from pharia_telemetry.setup.setup import setup_telemetry

            result = setup_telemetry(
                "test-svc", enable_console_exporter=False, enable_baggage_processor=False
            )

        assert result is True
        mock_exporter_cls.assert_called_once()

    def test_warning_when_http_exporter_not_installed(self, monkeypatch, caplog):
        """When http/protobuf is selected but the package is missing, a warning is logged."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        blocker = _make_import_blocker(
            {"opentelemetry.exporter.otlp.proto.http.trace_exporter"}
        )

        with patch("builtins.__import__", side_effect=blocker):
            with caplog.at_level(logging.WARNING):
                from pharia_telemetry.setup.setup import setup_telemetry

                result = setup_telemetry(
                    "test-svc",
                    enable_console_exporter=False,
                    enable_baggage_processor=False,
                )

        assert result is True
        assert "OTLP exporter for protocol 'http/protobuf' not available" in caplog.text
        assert "opentelemetry-exporter-otlp-proto-http" in caplog.text

    def test_warning_when_grpc_exporter_not_installed(self, monkeypatch, caplog):
        """When grpc is selected but the package is missing, a warning is logged."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        blocker = _make_import_blocker(
            {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter"}
        )

        with patch("builtins.__import__", side_effect=blocker):
            with caplog.at_level(logging.WARNING):
                from pharia_telemetry.setup.setup import setup_telemetry

                result = setup_telemetry(
                    "test-svc",
                    enable_console_exporter=False,
                    enable_baggage_processor=False,
                )

        assert result is True
        assert "OTLP exporter for protocol 'grpc' not available" in caplog.text
        assert "opentelemetry-exporter-otlp-proto-grpc" in caplog.text
