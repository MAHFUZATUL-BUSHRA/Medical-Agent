"""Main agent built with the OpenAI Agents SDK.

The two DB tools (LangChain SQL agents) and the web search tool are exposed to the
router agent as SDK function tools.
"""
from __future__ import annotations

from agents import Agent, Runner, function_tool

from . import config
from .llm import make_sdk_model
from .db_tools import (
    CANCER_TOOL_DESCRIPTION,
    HEART_TOOL_DESCRIPTION,
    make_cancer_tool,
    make_heart_tool,
)
from .prompts import ROUTER_PROMPT
from .web_tool import WEB_TOOL_DESCRIPTION, MedicalWebSearchTool


def build_agent() -> Agent:
    heart, cancer, web = make_heart_tool(), make_cancer_tool(), MedicalWebSearchTool()

    @function_tool(name_override="HeartDiseaseDBTool", description_override=HEART_TOOL_DESCRIPTION)
    def heart_disease_db_tool(question: str) -> str:
        """Args:
            question: The data question, in plain English.
        """
        return heart.run(question)

    @function_tool(name_override="CancerDBTool", description_override=CANCER_TOOL_DESCRIPTION)
    def cancer_db_tool(question: str) -> str:
        """Args:
            question: The data question, in plain English.
        """
        return cancer.run(question)

    @function_tool(name_override="MedicalWebSearchTool", description_override=WEB_TOOL_DESCRIPTION)
    def medical_web_search_tool(query: str) -> str:
        """Args:
            query: A concise web search query.
        """
        return web.run(query)

    return Agent(
        name="MedicalRouterAgent",
        instructions=ROUTER_PROMPT,
        tools=[heart_disease_db_tool, cancer_db_tool, medical_web_search_tool],
        model=make_sdk_model(),
    )


class AgentsSDKChat:
    """Chat session with memory on top of the Agents SDK."""

    def __init__(self) -> None:
        config.require_llm_key()
        self.agent = build_agent()
        self._history: list = []

    def ask(self, question: str) -> tuple[str, list[str]]:
        result = Runner.run_sync(self.agent, self._history + [{"role": "user", "content": question}])
        self._history = result.to_input_list()
        tools_used = [
            getattr(item.raw_item, "name", "?")
            for item in result.new_items
            if item.type == "tool_call_item"
        ]
        return str(result.final_output), tools_used
