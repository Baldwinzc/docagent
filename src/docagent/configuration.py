"""Configuration for the document knowledge-base agent.

Every value can be overridden with an environment variable of the same name in
upper case (see ``.env.example``). The module-level ``DEFAULT_*`` constants are
used by the ingestion script and the retrieval backend; the ``Configuration``
dataclass exposes the same knobs to the LangGraph runtime via ``configurable``.
"""

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

# --- Module-level defaults (shared by ingest + retrieval) ---
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "docagent")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "4"))
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
# LLM is the only piece that may need an API key; default to OpenAI, but any
# provider supported by ``init_chat_model`` works, e.g. "ollama:llama3.1".
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "openai:gpt-4.1")


@dataclass(kw_only=True)
class Configuration:
    """Runtime-configurable parameters for the agent graph."""

    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chroma_path: str = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration from env vars first, then ``configurable``."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
