"""Main agent built with LangChain's AgentExecutor (alternative backend)."""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from . import config
from .llm import make_chat_model
from .db_tools import (
    CANCER_TOOL_DESCRIPTION,
    HEART_TOOL_DESCRIPTION,
    make_cancer_tool,
    make_heart_tool,
)
from .prompts import ROUTER_PROMPT
from .web_tool import WEB_TOOL_DESCRIPTION, MedicalWebSearchTool


def build_executor() -> AgentExecutor:
    heart, cancer, web = make_heart_tool(), make_cancer_tool(), MedicalWebSearchTool()

    tools = [
        StructuredTool.from_function(
            func=heart.run, name="HeartDiseaseDBTool", description=HEART_TOOL_DESCRIPTION
        ),
        StructuredTool.from_function(
            func=cancer.run, name="CancerDBTool", description=CANCER_TOOL_DESCRIPTION
        ),
        StructuredTool.from_function(
            func=web.run, name="MedicalWebSearchTool", description=WEB_TOOL_DESCRIPTION
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    llm = make_chat_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=config.VERBOSE, return_intermediate_steps=True
    )


class LangChainChat:
    def __init__(self) -> None:
        config.require_llm_key()
        self.executor = build_executor()
        self._history: list = []

    def ask(self, question: str) -> tuple[str, list[str]]:
        out = self.executor.invoke({"input": question, "chat_history": self._history})
        self._history += [HumanMessage(question), AIMessage(out["output"])]
        tools_used = [action.tool for action, _ in out.get("intermediate_steps", [])]
        return out["output"], tools_used
