"""Quick connectivity test for your LLM + search settings.

    python -m scripts.check_llm
"""
import time

from medagent import config
from medagent.llm import make_chat_model

print(f"Provider: {config.LLM_PROVIDER} | model: {config.LLM_MODEL} | key set: {bool(config.LLM_API_KEY)}")

print("\n[1] Plain LLM call...")
t = time.time()
try:
    r = make_chat_model().invoke("Reply with the single word: OK")
    print(f"    OK in {time.time() - t:.1f}s -> {r.content!r}")
except Exception as e:
    print(f"    FAILED after {time.time() - t:.1f}s -> {type(e).__name__}: {e}")

print("\n[2] Web search call...")
from medagent.web_tool import MedicalWebSearchTool

t = time.time()
out = MedicalWebSearchTool().run("what is hypertension")
print(f"    {time.time() - t:.1f}s -> {out[:200]!r}")
