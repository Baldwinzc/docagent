"""Document knowledge-base agent (agentic RAG over a local Chroma store).

Two-layer LangGraph:
    START -> intent_router --in_scope--> response_agent (subgraph) -> END
                          \--out_of_scope--> END (politely declined)

The response_agent is the classic tool-calling loop, but instead of writing
emails it searches the knowledge base, may re-search with a better query, and
finishes by calling `Answer` (which forces citations).
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
    """Execute the (non-terminal) tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
        )
    return {"messages": result}


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


# --- Intent router (the front-line harness) ---
def intent_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Decide whether the question can be answered from the knowledge base."""
    question = state["question_input"].get("question", "")
    system_prompt = intent_system_prompt.format(
        kb_description=default_kb_description,
        intent_instructions=default_intent_instructions,
    )
    user_prompt = intent_user_prompt.format(question=question)

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    if result.classification == "in_scope":
        print("🔎 Intent: IN_SCOPE — retrieving from knowledge base")
        goto = "response_agent"
        update = {
            "classification_decision": result.classification,
            "messages": [
                {
                    "role": "user",
                    "content": f"Answer this question using the knowledge base: {question}",
                }
            ],
        }
    elif result.classification == "out_of_scope":
        print("🚫 Intent: OUT_OF_SCOPE — politely declining")
        goto = END
        update = {
            "classification_decision": result.classification,
            "messages": [
                {
                    "role": "assistant",
                    "content": "This question is outside the scope of the local "
                    "knowledge base, so I can't answer it from the available documents.",
                }
            ],
        }
    else:
        raise ValueError(f"Invalid classification: {result.classification}")

    return Command(goto=goto, update=update)


# Build the overall workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node("intent_router", intent_router)
    .add_node("response_agent", agent)
    .add_edge(START, "intent_router")
)

docagent = overall_workflow.compile()
