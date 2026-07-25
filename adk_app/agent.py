"""ADK entrypoint exposing the required ``root_agent`` symbol."""

from banking_agent.adk_adapter import create_adk_root_agent

root_agent = create_adk_root_agent()
