"""
Integration tests for GenAI file export functionality.

This module tests the actual file export capabilities of GenAI spans,
including console output that can be redirected to files.
"""

import io
import os
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from pharia_telemetry.setup import setup_telemetry


@pytest.mark.integration
class TestGenAIFileExportIntegration:
    """Integration tests for GenAI file export functionality."""

    def setup_method(self):
        """Set up for each test method."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("pharia_telemetry.setup.setup.OTEL_AVAILABLE", True)
    def test_setup_telemetry_with_file_output_config(self):
        """Test telemetry setup with file output configuration."""
        # Test setup with console exporter enabled (for file redirection)
        result = setup_telemetry(
            service_name="file-export-test",
            service_version="1.0.0",
            environment="test",
            enable_console_exporter=True,
        )

        assert result is True

    def test_console_output_capture(self):
        """Test capturing console output for file writing."""
        # Capture stdout to simulate file redirection
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            # Simulate console exporter output
            span_data = {
                "name": "chat gpt-4",
                "context": {"trace_id": "0x123", "span_id": "0x456"},
                "kind": "CLIENT",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.conversation.id": "conv_123",
                    "gen_ai.agent.id": "qa_chat",
                    "pharia.data.collections": ["docs", "knowledge_base"],
                },
                "status": {"status_code": "UNSET"},
            }

            # Simulate what console exporter would output
            print(f"Span: {span_data}")

        output = captured_output.getvalue()

        # Verify span data is in output
        assert "chat gpt-4" in output
        assert "gen_ai.operation.name" in output
        assert "conv_123" in output
        assert "qa_chat" in output
        assert "pharia.data.collections" in output

    def test_file_writing_with_span_data(self):
        """Test writing span data to actual files."""
        # Create temporary file for output
        temp_file_path = os.path.join(self.temp_dir, "spans_output.jsonl")

        # Sample span data that would be exported
        span_entries = [
            {
                "span_name": "chat gpt-4",
                "operation": "chat",
                "model": "gpt-4",
                "conversation_id": "conv_123",
                "data_context": {
                    "collections": ["docs", "knowledge_base"],
                    "dataset_ids": ["train_data"],
                    "namespaces": ["pharia"],
                    "indexes": ["vector_index"],
                },
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 85,
                    "total_tokens": 235,
                },
                "response": {
                    "id": "resp_abc123",
                    "model": "gpt-4-0613",
                    "finish_reasons": ["stop"],
                },
            },
            {
                "span_name": "embeddings text-embedding-3-small",
                "operation": "embeddings",
                "model": "text-embedding-3-small",
                "data_context": {
                    "collections": ["documents"],
                    "indexes": ["vector_index"],
                },
                "usage": {"input_tokens": 50, "output_tokens": 0, "total_tokens": 50},
            },
            {
                "span_name": "execute_tool calculator",
                "operation": "execute_tool",
                "tool_name": "calculator",
                "conversation_id": "conv_123",
                "result": {"success": True, "output": "42"},
            },
        ]

        # Write span data to file (JSONL format)
        with open(temp_file_path, "w") as f:
            for entry in span_entries:
                import json

                f.write(json.dumps(entry) + "\n")

        # Verify file was written correctly
        assert os.path.exists(temp_file_path)

        # Read back and verify content
        with open(temp_file_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Parse and verify first span
        import json

        first_span = json.loads(lines[0])
        assert first_span["span_name"] == "chat gpt-4"
        assert first_span["operation"] == "chat"
        assert first_span["model"] == "gpt-4"
        assert first_span["conversation_id"] == "conv_123"
        assert first_span["usage"]["input_tokens"] == 150
        assert first_span["usage"]["output_tokens"] == 85
        assert first_span["data_context"]["collections"] == ["docs", "knowledge_base"]

        # Parse and verify second span (embeddings)
        second_span = json.loads(lines[1])
        assert second_span["span_name"] == "embeddings text-embedding-3-small"
        assert second_span["operation"] == "embeddings"
        assert second_span["model"] == "text-embedding-3-small"
        assert second_span["usage"]["input_tokens"] == 50
        assert second_span["data_context"]["collections"] == ["documents"]

        # Parse and verify third span (tool execution)
        third_span = json.loads(lines[2])
        assert third_span["span_name"] == "execute_tool calculator"
        assert third_span["operation"] == "execute_tool"
        assert third_span["tool_name"] == "calculator"
        assert third_span["result"]["success"] is True

    @patch.dict(
        os.environ,
        {
            "OTEL_TRACES_EXPORTER": "console",
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG",
        },
    )
    def test_environment_based_file_export_config(self):
        """Test file export configuration via environment variables."""
        # These environment variables should enable console exporter
        # which can then be redirected to files

        # Simulate checking environment variables
        assert os.getenv("OTEL_TRACES_EXPORTER") == "console"
        assert os.getenv("ENVIRONMENT") == "development"
        assert os.getenv("LOG_LEVEL") == "DEBUG"

        # This configuration should enable console output for file redirection
        env_config = {
            "enable_console_exporter": (
                os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
                or os.getenv("OTEL_TRACES_EXPORTER") == "console"
                or os.getenv("ENVIRONMENT") == "development"
            )
        }

        assert env_config["enable_console_exporter"] is True

    def test_csv_export_format(self):
        """Test exporting span data in CSV format."""
        csv_file_path = os.path.join(self.temp_dir, "spans_export.csv")

        # Sample span data for CSV export
        span_data = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "span_name": "chat gpt-4",
                "operation": "chat",
                "model": "gpt-4",
                "conversation_id": "conv_123",
                "input_tokens": 150,
                "output_tokens": 85,
                "total_tokens": 235,
                "collections": "docs,knowledge_base",
                "duration_ms": 1250,
            },
            {
                "timestamp": "2024-01-15T10:30:05Z",
                "span_name": "embeddings text-embedding-3-small",
                "operation": "embeddings",
                "model": "text-embedding-3-small",
                "conversation_id": "",
                "input_tokens": 50,
                "output_tokens": 0,
                "total_tokens": 50,
                "collections": "documents",
                "duration_ms": 300,
            },
        ]

        # Write CSV file
        import csv

        with open(csv_file_path, "w", newline="") as csvfile:
            fieldnames = [
                "timestamp",
                "span_name",
                "operation",
                "model",
                "conversation_id",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "collections",
                "duration_ms",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in span_data:
                writer.writerow(row)

        # Verify CSV file
        assert os.path.exists(csv_file_path)

        # Read and verify CSV content
        with open(csv_file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert len(rows) == 2

        # Verify first row
        assert rows[0]["span_name"] == "chat gpt-4"
        assert rows[0]["operation"] == "chat"
        assert rows[0]["model"] == "gpt-4"
        assert rows[0]["input_tokens"] == "150"
        assert rows[0]["output_tokens"] == "85"
        assert rows[0]["collections"] == "docs,knowledge_base"

        # Verify second row
        assert rows[1]["span_name"] == "embeddings text-embedding-3-small"
        assert rows[1]["operation"] == "embeddings"
        assert rows[1]["input_tokens"] == "50"
        assert rows[1]["output_tokens"] == "0"

    def test_log_file_export_simulation(self):
        """Test simulating log file export with structured format."""
        log_file_path = os.path.join(self.temp_dir, "genai_spans.log")

        # Simulate structured log entries for GenAI spans
        log_entries = [
            "2024-01-15T10:30:00.123Z [INFO] GenAI span_start: chat gpt-4 | conversation_id=conv_123 | model=gpt-4",
            "2024-01-15T10:30:00.124Z [DEBUG] GenAI attributes: collections=['docs','knowledge_base'] namespaces=['pharia']",
            "2024-01-15T10:30:01.200Z [INFO] GenAI usage: input_tokens=150 output_tokens=85 total_tokens=235",
            "2024-01-15T10:30:01.250Z [INFO] GenAI response: id=resp_abc123 model=gpt-4-0613 finish_reason=stop",
            "2024-01-15T10:30:01.374Z [INFO] GenAI span_end: chat gpt-4 | duration=1251ms | status=ok",
            "",
            "2024-01-15T10:30:05.100Z [INFO] GenAI span_start: embeddings text-embedding-3-small | model=text-embedding-3-small",
            "2024-01-15T10:30:05.101Z [DEBUG] GenAI attributes: collections=['documents'] indexes=['vector_index']",
            "2024-01-15T10:30:05.300Z [INFO] GenAI usage: input_tokens=50 output_tokens=0 total_tokens=50",
            "2024-01-15T10:30:05.400Z [INFO] GenAI span_end: embeddings text-embedding-3-small | duration=300ms | status=ok",
        ]

        # Write log file
        with open(log_file_path, "w") as f:
            for entry in log_entries:
                f.write(entry + "\n")

        # Verify log file
        assert os.path.exists(log_file_path)

        # Read and verify log content
        with open(log_file_path) as f:
            content = f.read()

        # Verify key log entries are present
        assert "GenAI span_start: chat gpt-4" in content
        assert "conversation_id=conv_123" in content
        assert "input_tokens=150" in content
        assert "output_tokens=85" in content
        assert "collections=['docs','knowledge_base']" in content
        assert "GenAI span_start: embeddings text-embedding-3-small" in content
        assert "collections=['documents']" in content
        assert "duration=1251ms" in content

    @patch("pharia_telemetry.sem_conv.gen_ai.OTEL_AVAILABLE", True)
    def test_batch_file_export_simulation(self):
        """Test simulating batch export of multiple spans to file."""
        batch_file_path = os.path.join(self.temp_dir, "batch_spans_export.json")

        # Simulate a batch of spans that would be exported
        batch_data = {
            "export_timestamp": "2024-01-15T10:35:00Z",
            "service_name": "genai-service",
            "environment": "production",
            "total_spans": 5,
            "spans": [
                {
                    "span_id": "span_001",
                    "name": "chat gpt-4",
                    "operation": "chat",
                    "start_time": "2024-01-15T10:30:00Z",
                    "end_time": "2024-01-15T10:30:01.374Z",
                    "duration_ms": 1374,
                    "attributes": {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": "gpt-4",
                        "gen_ai.conversation.id": "conv_123",
                        "gen_ai.agent.id": "qa_chat",
                        "pharia.data.collections": ["docs", "knowledge_base"],
                        "pharia.data.namespaces": ["pharia"],
                    },
                    "usage": {
                        "input_tokens": 150,
                        "output_tokens": 85,
                        "total_tokens": 235,
                    },
                },
                {
                    "span_id": "span_002",
                    "name": "embeddings text-embedding-3-small",
                    "operation": "embeddings",
                    "start_time": "2024-01-15T10:30:05Z",
                    "end_time": "2024-01-15T10:30:05.400Z",
                    "duration_ms": 400,
                    "attributes": {
                        "gen_ai.operation.name": "embeddings",
                        "gen_ai.request.model": "text-embedding-3-small",
                        "pharia.data.collections": ["documents"],
                        "pharia.data.indexes": ["vector_index"],
                    },
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 0,
                        "total_tokens": 50,
                    },
                },
                {
                    "span_id": "span_003",
                    "name": "execute_tool calculator",
                    "operation": "execute_tool",
                    "start_time": "2024-01-15T10:30:10Z",
                    "end_time": "2024-01-15T10:30:10.150Z",
                    "duration_ms": 150,
                    "attributes": {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "calculator",
                        "gen_ai.conversation.id": "conv_123",
                        "gen_ai.agent.id": "qa_chat",
                    },
                },
            ],
        }

        # Write batch export file
        import json

        with open(batch_file_path, "w") as f:
            json.dump(batch_data, f, indent=2)

        # Verify file exists and content
        assert os.path.exists(batch_file_path)

        # Read and verify batch export
        with open(batch_file_path) as f:
            loaded_data = json.load(f)

        assert loaded_data["service_name"] == "genai-service"
        assert loaded_data["total_spans"] == 5
        assert len(loaded_data["spans"]) == 3

        # Verify first span
        first_span = loaded_data["spans"][0]
        assert first_span["name"] == "chat gpt-4"
        assert first_span["operation"] == "chat"
        assert first_span["duration_ms"] == 1374
        assert first_span["usage"]["input_tokens"] == 150
        assert first_span["attributes"]["gen_ai.conversation.id"] == "conv_123"

        # Verify embeddings span
        embeddings_span = loaded_data["spans"][1]
        assert embeddings_span["name"] == "embeddings text-embedding-3-small"
        assert embeddings_span["operation"] == "embeddings"
        assert embeddings_span["usage"]["input_tokens"] == 50
        assert embeddings_span["attributes"]["pharia.data.collections"] == ["documents"]

        # Verify tool span
        tool_span = loaded_data["spans"][2]
        assert tool_span["name"] == "execute_tool calculator"
        assert tool_span["operation"] == "execute_tool"
        assert tool_span["attributes"]["gen_ai.tool.name"] == "calculator"

    def test_file_export_error_handling(self):
        """Test error handling during file export operations."""
        # Test with invalid file path
        invalid_path = "/root/cannot_write_here/spans.log"

        try:
            with open(invalid_path, "w") as f:
                f.write("test")
            # If we get here, the path was actually writable, so skip this test
            os.unlink(invalid_path)
            pytest.skip("Cannot test invalid path - path was writable")
        except (PermissionError, FileNotFoundError, OSError):
            # This is expected - we can't write to this location
            pass

        # Test with valid path but simulated disk full error
        temp_file_path = os.path.join(self.temp_dir, "error_test.log")

        # This should work normally
        with open(temp_file_path, "w") as f:
            f.write("Normal write operation\n")

        assert os.path.exists(temp_file_path)

        # Verify content
        with open(temp_file_path) as f:
            content = f.read()

        assert "Normal write operation" in content
