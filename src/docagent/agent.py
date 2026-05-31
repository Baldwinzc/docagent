"""Document knowledge-base agent (agentic RAG over a local Chroma store).

Two-layer LangGraph:
    START -> intent_router --in_scope--> response_agent (subgraph) -> END
                          \--out_of_scope / empty KB--> END

The response_agent is a tool-calling loop: it searches the hybrid retriever,
may re-search with a better query, and finishes by calling `Answer` (which
forces citations). Every retrieval/decision is appended to `state["trace"]`
for observability, and tool failures are caught rather than crashing the graph.
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv

from docagent.tools import get_tools, get_tools_by_name
from docagent.prompts import (
    AGENT_TOOLS_PROMPT,
    agent_system_prompt,
    intent_system_prompt,
    intent_user_prompt,
    default_kb_description,
    default_intent_instructions,
)
from docagent.schemas import State, StateInput, IntentSchema
from docagent.configuration import DEFAULT_LLM_MODEL
from docagent.retriever import get_retriever

load_dotenv(".env")

# Tools
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# One LLM, two roles: structured router + tool-caller
llm = init_chat_model(DEFAULT_LLM_MODEL, temperature=0.0)
llm_router = llm.with_structured_output(IntentSchema)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

# Terminal tools end the loop instead of being executed
TERMINAL_TOOLS = {"Answer", "Question"}


# --- Nodes for the response agent (the RAG loop) ---
def llm_call(state: State):
    """LLM decides which retrieval tool to call next."""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": agent_system_prompt.format(
                            tools_prompt=AGENT_TOOLS_PROMPT,
                            kb_description=default_kb_description,
                        ),
                    }
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: State):
    """Execute non-terminal tool calls, recording a trace and catching failures."""
    result = []
    trace = []
    for tool_call in state["messages"][-1].tool_calls:
        name = tool_call["name"]
        try:
            observation = tools_by_name[name].invoke(tool_call["args"])
        except Exception as e:  # noqa: BLE001 — never let a tool crash the graph
            observation = f"Tool '{name}' failed: {e}"
        result.append(
            {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
        )
        if name == "search_docs":
            trace.append(
                {"step": "search_docs", "query": tool_call["args"].get("query", "")}
            )
        elif name == "list_sources":
            trace.append({"step": "list_sources"})
    return {"messages": result, "trace": trace}


def should_continue(state: State) -> Literal["environment", "__end__"]:
    """End when the agent calls a terminal tool (Answer/Question); else run tools."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        if any(tc["name"] in TERMINAL_TOOLS for tc in last_message.tool_calls):
            return END
        return "environment"
    return END


# Build the response-agent subgraph
agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"environment": "environment", END: END},
)
agent_builder.add_edge("environment", "llm_call")
agent = agent_builder.compile()


# --- Intent router (front-line harness) ---
def intent_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Guard an empty KB, then decide if the question is worth retrieving for."""
    question = state["question_input"].get("question", "")

    # Robustness: don't run the agent against an empty knowledge base.
    if get_retriever().is_empty:
        return Command(
            goto=END,
            update={
                "classification_decision": "out_of_scope",
                "trace": [{"step": "guard", "detail": "empty knowledge base"}],
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


# Build the overall workflow
overall_workflow = (
    StateGraph(State, input_schema=StateInput)
    .add_node("intent_router", intent_router)
    .add_node("response_agent", agent)
    .add_edge(START, "intent_router")
)

docagent = overall_workflow.compile()
