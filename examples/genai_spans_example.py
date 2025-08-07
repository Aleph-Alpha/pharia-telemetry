#!/usr/bin/env python3
"""
Example script demonstrating GenAI span convenience functions.

This script shows how to use the pharia-telemetry library to create
GenAI spans following OpenTelemetry semantic conventions. The convenience
functions (create_chat_span, create_embeddings_span, create_tool_execution_span)
automatically detect sync/async context and work seamlessly in both environments.
"""

import time
from typing import Any

from pharia_telemetry import (
    create_chat_span,
    create_embeddings_span,
    create_tool_execution_span,
    set_genai_span_response,
    set_genai_span_usage,
    setup_telemetry,
)
from pharia_telemetry.sem_conv.gen_ai import GenAI


def simulate_chat_completion() -> dict[str, Any]:
    """Simulate a chat completion with proper telemetry."""
    print("🤖 Simulating chat completion...")

    # Create a chat completion span
    with create_chat_span(
        agent_name="Pharia QA Assistant",
        model="llama-3.1-8B",
        conversation_id="conv-12345",
        agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
    ) as _span:
        print("   📡 Created GenAI span for chat completion")

        # Simulate API call delay
        time.sleep(0.1)

        # Set token usage
        set_genai_span_usage(input_tokens=150, output_tokens=85)
        print("   📊 Set token usage: 150 input, 85 output")

        # Set response information
        set_genai_span_response(
            response_id="chatcmpl-8abc123",
            finish_reasons=["stop"],
            model="gpt-4-0125-preview",
        )
        print("   ✅ Set response metadata")

        return {
            "response": "Hello! I'm your AI assistant.",
            "usage": {"total_tokens": 235},
        }


def simulate_tool_execution() -> dict[str, Any]:
    """Simulate a tool execution with proper telemetry."""
    print("\n🔧 Simulating tool execution...")

    # Create a tool execution span
    with create_tool_execution_span(
        tool_name="calculate_sum",
        agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
        conversation_id="conv-12345",
        additional_attributes={
            GenAI.TOOL_DESCRIPTION: "Calculate the sum of two numbers",
            GenAI.TOOL_CALL_ID: "call_abc123",
        },
    ) as _span:
        print("   🔧 Created GenAI span for tool execution")

        # Simulate tool execution
        time.sleep(0.05)

        return {"result": 42, "status": "success"}


def simulate_embeddings() -> dict[str, Any]:
    """Simulate embeddings generation with proper telemetry."""
    print("\n📊 Simulating embeddings generation...")

    # Create an embeddings span
    with create_embeddings_span(
        model="llama-3.1-8B-embeddings",
        additional_attributes={
            "gen_ai.request.encoding_format": "float",
            "gen_ai.request.dimensions": 1536,
        },
    ) as _span:
        print("   🧮 Created GenAI span for embeddings")

        # Simulate embeddings API call
        time.sleep(0.08)

        # Set token usage (embeddings typically only have input tokens)
        set_genai_span_usage(input_tokens=75, output_tokens=0)
        print("   📊 Set token usage: 75 input tokens")

        return {"embeddings": [0.1, 0.2, 0.3], "dimensions": 1536}


def main() -> int:
    """Main demonstration function."""
    print("🚀 Starting GenAI spans demonstration")
    print("=" * 50)

    # Setup telemetry (using console exporter for demo)
    setup_telemetry(
        service_name="genai-demo",
        service_version="1.0.0",
        enable_console_exporter=True,  # For demonstration purposes
        environment="development",
    )
    print("📋 Telemetry setup complete")
    print()

    # Demonstrate different GenAI operations
    try:
        # 1. Chat completion
        chat_result = simulate_chat_completion()
        print(f"   💬 Chat result: {chat_result['response'][:30]}...")
        print()

        # 2. Tool execution
        tool_result = simulate_tool_execution()
        print(f"   🔧 Tool result: {tool_result}")
        print()

        # 3. Embeddings generation
        embeddings_result = simulate_embeddings()
        print(f"   🧮 Embeddings dimensions: {embeddings_result['dimensions']}")
        print()

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return 1

    print("✅ GenAI spans demonstration completed successfully!")
    print()
    print("📝 Key features demonstrated:")
    print("   • Smart convenience functions that auto-detect sync/async context")
    print("   • Different GenAI operation types (chat, tools, embeddings)")
    print("   • Proper span attribute setting following OpenTelemetry conventions")
    print("   • Token usage tracking")
    print("   • Response metadata capture")
    print("   • Conversation and agent correlation")

    return 0


if __name__ == "__main__":
    exit(main())
