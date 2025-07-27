#!/usr/bin/env python3
"""
Comprehensive GenAI Span Functions Example

This example demonstrates all the GenAI convenience functions available in the
pharia-telemetry library, showing how to create spans for different AI operations
with proper OpenTelemetry semantic conventions.
"""

from typing import Any

from pharia_telemetry import (
    set_genai_span_response,
    set_genai_span_usage,
    setAgentCreationSpan,
    setAgentInvocationSpan,
    setChatSpan,
    setEmbeddingsSpan,
    setToolExecutionSpan,
    setup_telemetry,
)
from pharia_telemetry.constants.gen_ai import GenAI


def simulate_chat_operation() -> dict[str, Any]:
    """Demonstrate chat span with token usage tracking."""
    print("💬 Simulating chat operation...")

    with setChatSpan(
        model="gpt-4",
        system=GenAI.Values.System.OPENAI,
        conversation_id="conv-chat-123",
        agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
        agent_name="Pharia QA Assistant",
    ):
        # Simulate AI model call
        print("   🤖 Processing user query with GPT-4...")

        # Mock response
        response: dict[str, Any] = {
            "id": "chatcmpl-123",
            "choices": [{"finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
            },
        }

        # Set usage tracking
        set_genai_span_usage(
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        # Set response metadata
        set_genai_span_response(
            response_id=response["id"], finish_reasons=["stop"], model="gpt-4"
        )

        print(f"   📊 Used {response['usage']['total_tokens']} tokens")
        return response


def simulate_tool_execution() -> dict[str, Any]:
    """Demonstrate tool execution span."""
    print("\n🛠️  Simulating tool execution...")

    with setToolExecutionSpan(
        tool_name="web_search",
        tool_description="Search the web for current information",
        tool_call_id="call_tool_456",
        conversation_id="conv-tool-123",
        agent_id=GenAI.Values.PhariaAgentId.QA_CHAT,
    ):
        print("   🔍 Executing web search tool...")

        # Simulate tool execution
        search_results: dict[str, Any] = {
            "query": "Python OpenTelemetry best practices",
            "results_count": 10,
            "execution_time_ms": 250,
        }

        print(
            f"   ✅ Found {search_results['results_count']} results in {search_results['execution_time_ms']}ms"
        )
        return search_results


def simulate_agent_creation() -> dict[str, Any]:
    """Demonstrate agent creation span."""
    print("\n🤖 Simulating agent creation...")

    with setAgentCreationSpan(
        agent_name="Custom Research Assistant",
        agent_id="agent_research_789",
        conversation_id="conv-creation-123",
    ):
        print("   ⚙️  Creating custom research assistant...")

        # Simulate agent configuration
        agent_config: dict[str, Any] = {
            "name": "Custom Research Assistant",
            "capabilities": ["web_search", "document_analysis", "summarization"],
            "model": "gpt-4",
            "system_prompt": "You are a helpful research assistant...",
        }

        print(
            f"   ✅ Agent created with {len(agent_config['capabilities'])} capabilities"
        )
        return agent_config


def simulate_embeddings_operation() -> dict[str, Any]:
    """Demonstrate embeddings span."""
    print("\n📊 Simulating embeddings operation...")

    with setEmbeddingsSpan(
        model="text-embedding-3-small",
        system=GenAI.Values.System.OPENAI,
        encoding_format="float",
        dimensions=1536,
    ):
        print("   🧮 Generating embeddings for text chunks...")

        # Simulate embeddings generation
        embeddings_result: dict[str, Any] = {
            "input_texts": ["Document chunk 1", "Document chunk 2", "Document chunk 3"],
            "embedding_dimensions": 1536,
            "total_tokens": 45,
        }

        # Set usage (embeddings only use input tokens)
        set_genai_span_usage(input_tokens=embeddings_result["total_tokens"])

        print(
            f"   ✅ Generated {len(embeddings_result['input_texts'])} embeddings ({embeddings_result['embedding_dimensions']} dimensions)"
        )
        return embeddings_result


def simulate_agent_invocation() -> dict[str, Any]:
    """Demonstrate agent invocation span."""
    print("\n🚀 Simulating agent invocation...")

    with setAgentInvocationSpan(
        agent_id=GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
        agent_name="Pharia Agentic Assistant",
        model="gpt-4",
        system=GenAI.Values.System.PHARIA_AI,
        conversation_id="conv-agentic-123",
    ):
        print("   🎯 Invoking agentic chat assistant...")

        # Simulate complex agentic workflow
        workflow_steps = [
            "🔍 Analyzing user request",
            "📋 Planning multi-step approach",
            "🛠️  Executing tools in sequence",
            "📝 Synthesizing final response",
        ]

        for i, step in enumerate(workflow_steps, 1):
            print(f"   Step {i}/4: {step}")

        # Mock final result
        result: dict[str, Any] = {
            "steps_executed": len(workflow_steps),
            "tools_used": ["web_search", "document_analyzer", "summarizer"],
            "total_tokens": 850,
            "execution_time_ms": 3200,
        }

        # Set comprehensive usage tracking
        set_genai_span_usage(
            input_tokens=400, output_tokens=450, total_tokens=result["total_tokens"]
        )

        set_genai_span_response(
            response_id="agentic_response_789", finish_reasons=["stop"], model="gpt-4"
        )

        print(
            f"   ✅ Completed {result['steps_executed']}-step workflow using {len(result['tools_used'])} tools"
        )
        return result


def simulate_complex_workflow() -> dict[str, Any]:
    """Demonstrate a complex workflow using multiple GenAI operations."""
    print("\n🌟 Simulating complex multi-operation workflow...")

    # Start with agent invocation that orchestrates other operations
    with setAgentInvocationSpan(
        agent_id=GenAI.Values.PhariaAgentId.AGENTIC_CHAT,
        agent_name="Workflow Orchestrator",
        conversation_id="conv-workflow-123",
    ):
        print("   🎯 Starting workflow orchestration...")

        # Step 1: Create specialized agent
        agent_config = simulate_agent_creation()

        # Step 2: Use tool for information gathering
        search_results = simulate_tool_execution()

        # Step 3: Generate embeddings for knowledge base
        embeddings_result = simulate_embeddings_operation()

        # Step 4: Final chat synthesis
        chat_response = simulate_chat_operation()

        # Calculate total workflow metrics
        total_tokens = (
            chat_response["usage"]["total_tokens"] + embeddings_result["total_tokens"]
        )

        print("\n   📊 Workflow Summary:")
        print(f"      • Agent created: {agent_config['name']}")
        print(f"      • Search results: {search_results['results_count']} items")
        print(
            f"      • Embeddings generated: {len(embeddings_result['input_texts'])} chunks"
        )
        print(f"      • Total tokens used: {total_tokens}")

        return {
            "workflow_completed": True,
            "operations_count": 4,
            "total_tokens": total_tokens,
        }


def main() -> int:
    """Main demonstration function."""
    print("🚀 Starting Comprehensive GenAI Span Functions Demo")
    print("=" * 70)

    # Setup telemetry with console output for demonstration
    setup_success = setup_telemetry(
        service_name="genai-span-demo",
        service_version="1.0.0",
        enable_console_exporter=True,
        environment="development",
    )

    if setup_success:
        print("📋 Telemetry setup complete - spans will be traced")
    else:
        print("⚠️  Telemetry setup failed - running in demo mode")

    try:
        # Demonstrate individual operations
        print("\n" + "🔹" * 50)
        print("INDIVIDUAL OPERATIONS DEMO")
        print("🔹" * 50)

        _chat_result = simulate_chat_operation()
        _tool_result = simulate_tool_execution()
        _agent_result = simulate_agent_creation()
        _embeddings_result = simulate_embeddings_operation()
        _invocation_result = simulate_agent_invocation()

        # Demonstrate complex workflow
        print("\n" + "🔸" * 50)
        print("COMPLEX WORKFLOW DEMO")
        print("🔸" * 50)

        workflow_result = simulate_complex_workflow()

        # Summary
        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("\n📝 Functions demonstrated:")
        print("   • setChatSpan - Chat operations with token tracking")
        print("   • setToolExecutionSpan - Tool execution with metadata")
        print("   • setAgentCreationSpan - Agent creation workflows")
        print("   • setEmbeddingsSpan - Embeddings generation")
        print("   • setAgentInvocationSpan - Complex agentic workflows")
        print("   • set_genai_span_usage - Token usage tracking")
        print("   • set_genai_span_response - Response metadata")

        print("\n🎯 All functions follow OpenTelemetry GenAI semantic conventions")
        print(
            f"📊 Total operations: {workflow_result['operations_count']} + 5 individual demos"
        )

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
