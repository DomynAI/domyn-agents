# Mock configuration: tools referenced by uid and by name across agents and sub-agents.
import os
from unittest.mock import AsyncMock, patch

import pytest

from domyn.agents.agents import Agent
from domyn.agents.loader import load_application_from_config_new

vllm_api_key = os.getenv("VLLM_API_KEY") or os.getenv("VLLM_API_KEY_DEFAULT")
config = {
    "context_managers": [
        {
            "name": "summary",
            "type": "interactions_summary",
            "prompting": {"type": "json_schema"},
            "llm_provider": "qwen3",
            "min_events_for_summary": 5,
        }
    ],
    "planner_strategies": [
        {
            "name": "basic",
            "type": "react",
            "max_iterations": 10,
            "verbose": True,
            "use_stop": False,
            "context_manager": "summary",
        },
        {
            "name": "with_stop",
            "type": "react",
            "max_iterations": 10,
            "verbose": True,
            "use_stop": True,
        },
    ],
    "llm_providers": [
        {
            "name": "qwen3",
            "type": "vllm",
            "model_name": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            "url": "https://llm.example.com/v1/chat/completions",
            "api_key": vllm_api_key,
        },
        {
            "name": "qwen32B",
            "type": "vllm",
            "model_name": "Qwen/Qwen3-32B",
            "url": "https://llm.example.com/v1/chat/completions",
            "api_key": vllm_api_key,
        },
    ],
    "tools": [
        {
            "name": "get_order_info",
            "uid": "id_3",
            "module": "get_order_info",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
        },
        {
            "name": "get_customer_orders",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
        },
        {
            "name": "get_payment_info",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
        },
        {
            "name": "refund_payment",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
        },
        {
            "name": "get_user_id_from_name",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
            "uid": "id_1",
            "config_params": [{"name": "key", "value": "example-key-1"}],
            "pre_hooks": ["print_tool_args"],
            "post_hooks": ["print_tool_result"],
        },
        {
            "name": "get_user_id_from_name",
            "type": "function",
            "path": "tests/loader/customer_service_tools.py",
            "uid": "id_2",
            "config_params": [{"name": "key", "value": "example-key-2"}],
        },
    ],
    "agents": [
        {
            "name": "OrderAgent",
            "description": "Agent able to retrieve information about orders",
            "instruction": "unused prompt",
            "context_visibility": "standard",
            "llm_provider": "qwen32B",
            "tools": ["id_3", "get_customer_orders"],
            "planner_strategy": "basic",
            "pre_hooks": ["print_start_agent_name"],
            "post_hooks": ["print_end_agent_name"],
        },
        {
            "name": "CustomerServiceAgent",
            "description": "Agent for handling customer service inquiries. Has the ability to retrieve the user_id from the User's name.",
            "instruction": "unused prompt",
            "context_visibility": "standard",
            "llm_provider": "qwen32B",
            "tools": ["id_1"],
            "sub_agents": ["OrderAgent", "PaymentAgent"],
            "planner_strategy": "basic",
            "pre_hooks": ["print_start_agent_name"],
            "post_hooks": ["print_end_agent_name"],
        },
        {
            "name": "PaymentAgent",
            "description": "Agent able to retrieve information about payments",
            "instruction": "unused prompt",
            "context_visibility": "standard",
            "llm_provider": "qwen32B",
            "tools": ["get_payment_info", "refund_payment"],
            "planner_strategy": "with_stop",
            "pre_hooks": ["print_start_agent_name"],
            "post_hooks": ["print_end_agent_name"],
        },
    ],
}


@pytest.mark.asyncio
async def test_loading_config_resolves_tools_by_id_and_name():
    agent = await load_application_from_config_new(
        {"version": "1.0", "entrypoint": "CustomerServiceAgent", **config}
    )

    tool_names = [t.name for t in agent.tools]
    assert any(name == "get_user_id_from_name" for name in tool_names)

    assert all(t.name == "get_user_id_from_name" for t in agent.tools), (
        "CustomerServiceAgent must have get_user_id_from_name as its only tool"
    )

    order_agent = next(a for a in agent.sub_agents if a.name == "OrderAgent")
    order_tool_names = [t.name for t in order_agent.tools]
    assert any(name == "get_order_info" for name in order_tool_names)
    assert any(name == "get_customer_orders" for name in order_tool_names)

    payment_agent = next(a for a in agent.sub_agents if a.name == "PaymentAgent")
    payment_tool_names = [t.name for t in payment_agent.tools]
    assert any(name == "get_payment_info" for name in payment_tool_names)
    assert any(name == "refund_payment" for name in payment_tool_names)


@pytest.mark.asyncio
async def test_mcp_tools_can_be_configured_via_name_or_uid_under_subagent():
    """Test that MCP tools can be configured via name or uid under a subagent."""
    # Configuration with MCP tools similar to the provided example
    mcp_config = {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [
            {
                "name": "qwen32B",
                "type": "vllm",
                "model_name": "Qwen/Qwen3-32B",
                "url": "https://llm.example.com/v1/chat/completions",
                "api_key": vllm_api_key,
            }
        ],
        "planner_strategies": [
            {
                "name": "react",
                "type": "react",
                "max_iterations": 10,
                "verbose": True,
                "use_stop": False,
            }
        ],
        "context_managers": [
            {
                "name": "summary",
                "type": "interactions_summary",
                "prompting": {"type": "json_schema"},
                "llm_provider": "qwen32B",
            }
        ],
        "tools": [
            {
                "name": "fetch",
                "uid": "silver_uid_456",
                "type": "mcp",
                "server_url": "https://remote.mcpservers.org/server-a",
            },
            {
                "name": "test_tool_b",
                "uid": "test2",
                "type": "mcp",
                "server_url": "https://remote.mcpservers.org/server-b",
            },
        ],
        "agents": [
            {
                "name": "MarketAnalyst",
                "description": "Specialist in market data analysis",
                "instruction": "You specialize in market data analysis using available tools.",
                "llm_provider": "qwen32B",
                "context_manager": "summary",
                "planner_strategy": "react",
                "context_visibility": "standard",
                "tools": ["silver_uid_456", "test_tool_b"],  # Mixed: uid and name reference
            },
            {
                "name": "Assistant",
                "description": "Expert assistant with MCP tools",
                "instruction": "You are an expert assistant aiding users with their requests. Use the available tools to gather information and provide accurate responses.",
                "llm_provider": "qwen32B",
                "context_manager": "summary",
                "planner_strategy": "react",
                "context_visibility": "global",
                "tools": ["fetch", "test2"],  # Reference by name
                "sub_agents": ["MarketAnalyst"],
            },
        ],
    }

    # Load the configuration
    with patch.object(Agent, "initialize_mcp_tools", new_callable=AsyncMock):
        agent = await load_application_from_config_new(mcp_config)

    # Verify main agent has the tools referenced by name and uid
    main_agent_tool_names = [t.name for t in agent.tools]
    assert "fetch" in main_agent_tool_names, "Main agent should have fetch tool referenced by name"
    assert "test_tool_b" in main_agent_tool_names, (
        "Main agent should have test_tool_b tool referenced by uid"
    )

    # Verify subagent has tools referenced both by uid and name
    market_analyst = next((a for a in agent.sub_agents if a.name == "MarketAnalyst"), None)
    assert market_analyst is not None, "MarketAnalyst subagent should exist"

    market_analyst_tool_names = [t.name for t in market_analyst.tools]
    assert "fetch" in market_analyst_tool_names, "Subagent should have fetch tool referenced by uid"
    assert "test_tool_b" in market_analyst_tool_names, (
        "Subagent should have test_tool_b referenced by name"
    )

    # Verify that tools are properly resolved regardless of reference method
    assert len(market_analyst.tools) == 2, "MarketAnalyst should have exactly 2 tools"
