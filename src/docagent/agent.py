"""Document knowledge-base agent (agentic RAG over a local Chroma store).

Two-layer LangGraph:
    START -> intent_router --in_scope--> response_agent (subgraph) -> END
                          \--out_of_scope / empty KB--> END

The agent is constructed by ``build_agent(config)`` — **nothing is initialised at
import time** (no LLM, no reranker), so importing this module is cheap and the
model/retriever lifecycles are explicit. ``make_graph`` is the factory used by
``langgraph.json``; ``get_default_agent`` is a lazily-built, cached default for
the CLI and web server.

Every retrieval is recorded in ``state["trace"]`` and the locators it returned
are accumulated in ``state["retrieved_locators"]`` so the final answer's
citations can be *verified* against what was actually retrieved.
"""

import re
from functools import lru_cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from docagent.configuration import Configuration
from docagent.prompts import (
    AGENT_TOOLS_PROMPT,
    agent_system_prompt,
    default_intent_instructions,
    default_kb_description,
    intent_system_prompt,
    intent_user_prompt,
)
from docagent.retriever import get_retriever
from docagent.schemas import IntentSchema, State, StateInput
from docagent.tools import get_tools, get_tools_by_name

TERMINAL_TOOLS = {"Answer", "Question"}
# matches "locator: <loc>  (relevance" in search_docs output
_LOCATOR_RE = re.compile(r"locator:\s*(.+?)\s{2,}\(relevance")


def build_agent(config: Configuration | None = None):
    """Build and compile the agent graph for a given configuration.

    All model/tool wiring lives here (not at module import), captured in node
    closures, so the same module can serve multiple configs and import stays cheap.
    """
    config = config or Configuration.from_runnable_config()

    tools = get_tools()
    tools_by_name = get_tools_by_name(tools)
    llm = init_chat_model(config.llm_model, temperature=0.0)
    llm_router = llm.with_structured_output(IntentSchema)
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")

    system_prompt = agent_system_prompt.format(
        tools_prompt=AGENT_TOOLS_PROMPT, kb_description=default_kb_description
    )

    def llm_call(state: State):
        """LLM decides which retrieval tool to call next."""
        return {
            "messages": [
                llm_with_tools.invoke(
                    [{"role": "system", "content": system_prompt}] + state["messages"]
                )
            ]
        }

    def tool_node(state: State):
        """Run non-terminal tools; record trace + retrieved locators; catch failures."""
        result, trace, locators = [], [], []
        for tool_call in state["messages"][-1].tool_calls:
            name = tool_call["name"]
            try:
                observation = tools_by_name[name].invoke(tool_call["args"])
            except Exception as e:  # noqa: BLE001 — never crash the graph
                observation = f"Tool '{name}' failed: {e}"
            result.append(
                {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
            )
            if name == "search_docs":
                trace.append(
                    {"step": "search_docs", "query": tool_call["args"].get("query", "")}
                )
                locators.extend(_LOCATOR_RE.findall(observation))
            elif name == "list_sources":
                trace.append({"step": "list_sources"})
        return {"messages": result, "trace": trace, "retrieved_locators": locators}

    def should_continue(state: State) -> Literal["environment", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            if any(tc["name"] in TERMINAL_TOOLS for tc in last_message.tool_calls):
                return END
            return "environment"
        return END

    agent_builder = StateGraph(State)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call", should_continue, {"environment": "environment", END: END}
    )
    agent_builder.add_edge("environment", "llm_call")
    response_agent = agent_builder.compile()

    def intent_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
        """Guard an empty KB, then decide if the question is worth retrieving for."""
        question = state["question_input"].get("question", "")

        if get_retriever(config.chroma_path, config.collection_name).is_empty:
            return Command(
                goto=END,
                update={
                    "classification_decision": "out_of_scope",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "The knowledge base is empty. Run "
                            "`python -m docagent.ingest --path <your-docs>` first.",
                        }
                    ],
                },
            )

        result = llm_router.invoke(
            [
                {
                    "role": "system",
                    "content": intent_system_prompt.format(
                        kb_description=default_kb_description,
                        intent_instructions=default_intent_instructions,
                    ),
                },
                {"role": "user", "content": intent_user_prompt.format(question=question)},
            ]
        )

        if result.classification == "in_scope":
            print("🔎 Intent: IN_SCOPE — retrieving from knowledge base")
            return Command(
                goto="response_agent",
                update={
                    "classification_decision": "in_scope",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Answer this question using the knowledge base: {question}",
                        }
                    ],
                },
            )

        print("🚫 Intent: OUT_OF_SCOPE — politely declining")
        return Command(
            goto=END,
            update={
                "classification_decision": "out_of_scope",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "This question is outside the scope of the local "
                        "knowledge base, so I can't answer it from the available documents.",
                    }
                ],
            },
        )

    overall_workflow = (
        StateGraph(State, input_schema=StateInput)
        .add_node("intent_router", intent_router)
        .add_node("response_agent", response_agent)
        .add_edge(START, "intent_router")
    )
    return overall_workflow.compile()


def make_graph():
    """Graph factory for ``langgraph.json`` / LangGraph dev."""
    return build_agent()


@lru_cache(maxsize=1)
def get_default_agent():
    """Lazily-built, cached default agent (used by the CLI and web server)."""
    return build_agent()
