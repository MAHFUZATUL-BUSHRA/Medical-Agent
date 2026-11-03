"""MedicalWebSearchTool: web search for general medical knowledge."""
from __future__ import annotations

import os

from . import config

WEB_TOOL_DESCRIPTION = (
    "Search the web for GENERAL medical knowledge: definitions, symptoms, causes, risk factors, "
    "treatments, prevention, cures, or explanations of medical terms. "
    "Do NOT use for statistics about the provided datasets. Input: a concise search query."
)


class MedicalWebSearchTool:
    name = "MedicalWebSearchTool"

    def __init__(self, provider: str | None = None, max_results: int = 5):
        self.provider = (provider or config.SEARCH_PROVIDER).lower()
        self.max_results = max_results

    # ---- providers -------------------------------------------------
    def _tavily(self, query: str) -> str:
        from tavily import TavilyClient

        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not set.")
        res = TavilyClient(api_key=key).search(
            query=f"{query} (medical)",
            max_results=self.max_results,
            search_depth="basic",
            include_answer=True,
        )
        lines = []
        if res.get("answer"):
            lines.append(f"Summary: {res['answer']}\n")
        lines.append("Sources:")
        for i, r in enumerate(res.get("results", []), 1):
            snippet = (r.get("content") or "")[:400].replace("\n", " ")
            lines.append(f"{i}. {r.get('title')} - {r.get('url')}\n   {snippet}")
        return "\n".join(lines)

    def _serpapi(self, query: str) -> str:
        import requests

        key = os.getenv("SERPAPI_API_KEY")
        if not key:
            raise RuntimeError("SERPAPI_API_KEY is not set.")
        data = requests.get(
            "https://serpapi.com/search.json",
            params={"q": f"{query} medical", "api_key": key, "num": self.max_results},
            timeout=30,
        ).json()
        lines = []
        box = data.get("answer_box") or {}
        if box.get("answer") or box.get("snippet"):
            lines.append(f"Summary: {box.get('answer') or box.get('snippet')}\n")
        lines.append("Sources:")
        for i, r in enumerate(data.get("organic_results", [])[: self.max_results], 1):
            lines.append(f"{i}. {r.get('title')} - {r.get('link')}\n   {r.get('snippet', '')}")
        return "\n".join(lines)

    # ---- public ----------------------------------------------------
    def run(self, query: str) -> str:
        try:
            if self.provider == "serpapi":
                return self._serpapi(query)
            return self._tavily(query)
        except Exception as exc:
            return f"MedicalWebSearchTool failed: {exc}"
