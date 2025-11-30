# src/agent.py
import os
import json
from openai import OpenAI

import pandas as pd

# Load your functions and data loader
from src.dataloader import load_data
from src.functions import (
    search_relevant_courses,
    keep_columns,
    filter_courses_by_keyword,
    add_ECTS,
    get_course_ects,
)

from src.planner import (
    plan_msc_program,
    print_plan,
)

# --------------------------------------------------------
# Load API key
# --------------------------------------------------------
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
if not CAMPUSAI_API_KEY:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://chat.campusai.compute.dtu.dk/api/v1"
)

# --------------------------------------------------------
# Load DTU dataset once globally
# --------------------------------------------------------
COURSES_DF = load_data()

# --------------------------------------------------------
# Tool definitions: how OpenAI sees your Python functions
# --------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_relevant_courses",
            "description": "Search for relevant DTU courses using hybrid dense+sparse similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "mode": {"type": "string", "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.2},
                },
                "required": ["query", "top_k"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_courses_by_keyword",
            "description": "Filter courses by a keyword across selected columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["keyword", "columns"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keep_columns",
            "description": "Keep only selected columns from a DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_names": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["column_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_ECTS",
            "description": "Sum total ECTS of selected courses.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_msc_program",
            "description": "Build a 2-year MSc program plan following DTU semester rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ects_target": {"type": "number", "default": 85.0},
                    "top_k": {"type": "integer", "default": 200},
                    "alpha": {"type": "number", "default": 0.2},
                },
                "required": ["query"]
            }
        }
    }
]

# --------------------------------------------------------
# TOOL DISPATCHER — executes Python when LLM calls tool
# --------------------------------------------------------

def execute_tool(tool_name, arguments):
    """
    Dispatch pipe connecting LLM's tool call to actual Python functions.
    """

    if tool_name == "search_relevant_courses":
        df = COURSES_DF.copy()
        result = search_relevant_courses(
            df=df,
            query=arguments["query"],
            top_k=arguments["top_k"],
            mode=arguments.get("mode", "hybrid"),
            alpha=arguments.get("alpha", 0.2),
        )
        return result.to_json(orient="records")

    if tool_name == "filter_courses_by_keyword":
        df = COURSES_DF.copy()
        filtered = filter_courses_by_keyword(
            df=df,
            keyword=arguments["keyword"],
            columns=arguments["columns"],
        )
        return filtered.to_json(orient="records")

    if tool_name == "keep_columns":
        df = COURSES_DF.copy()
        kept = keep_columns(arguments["column_names"], df)
        return kept.to_json(orient="records")

    if tool_name == "add_ECTS":
        df = COURSES_DF.copy()
        ects = add_ECTS(df)
        return json.dumps({"total_ects": ects})

    if tool_name == "plan_msc_program":
        df = COURSES_DF.copy()
        plan = plan_msc_program(
            df=df,
            query=arguments["query"],
            ects_target=arguments.get("ects_target", 85.0),
            top_k=arguments.get("top_k", 200),
            alpha=arguments.get("alpha", 0.2),
        )
        # Convert DataFrames inside return to JSON
        serializable = {
            "total_ects": plan["total_ects"],
            "buffer_courses": plan["buffer_courses"].to_dict(orient="records"),
            "retrieved_candidates": plan["retrieved_candidates"].to_dict(orient="records"),
            "periods": {
                pid: {
                    **{
                        k: v for k, v in info.items()
                        if k not in ["courses_df"]
                    },
                    "courses_df": info["courses_df"].to_dict(orient="records")
                }
                for pid, info in plan["periods"].items()
            }
        }
        return json.dumps(serializable)

    return json.dumps({"error": "unknown tool"})


# --------------------------------------------------------
# SYSTEM PROMPT FOR THE AGENT
# --------------------------------------------------------

SYSTEM_PROMPT = """
You are the DTU MSc Study Plan Agent.

Your job:
- Understand user goals for a DTU MSc study program.
- Use the available tools to search, filter, and adjust courses.
- Use `plan_msc_program` to generate full plans that obey:
    • Autumn ≤ 30 ECTS
    • January = 5 ECTS
    • Spring ≤ 30 ECTS
    • June = 5 ECTS
    • July = 5 ECTS
    • August = 5 ECTS
    • Autumn2 ≤ 30 ECTS
    • January2 = 5 ECTS
- Ensure no schedule overlaps based on DTU time slot structure.
- If the user asks to adjust something, revise the plan using the buffer courses.

You MUST call tools when relevant. 
If the user asks for a plan, ALWAYS call `plan_msc_program`.
"""


# --------------------------------------------------------
# MAIN CALL: send message to the agent
# --------------------------------------------------------

def ask_agent(user_message: str):
    """
    Run a single agent turn.
    Handles tool calls automatically.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # If LLM calls a tool
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"\n[Executing tool: {tool_name}]")
            result = execute_tool(tool_name, arguments)

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
            continue

        # Otherwise return final response
        return msg["content"]


# --------------------------------------------------------
# Manual test
# --------------------------------------------------------
if __name__ == "__main__":
    answer = ask_agent("Create an MSc plan specializing in data science and robotics.")
    print("\nAGENT RESPONSE:\n", answer)
