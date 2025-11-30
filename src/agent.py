# ================================================================
# src/agent.py  —  DTU MSc Course Planner Agent (UPDATED VERSION)
# ================================================================

import os
import json
from openai import OpenAI
import pandas as pd

from src.dataloader import load_data
from src.functions import (
    search_relevant_courses,
    keep_columns,
    filter_courses_by_keyword,
    add_ECTS,
)
from src.planner import plan_msc_program

# --------------------------------------------------------
# Load API key & client
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
# Load courses globally
# --------------------------------------------------------
COURSES_DF = load_data()

# --------------------------------------------------------
# Tools available to the agent
# --------------------------------------------------------
SYSTEM_PROMPT = """
You are the DTU MSc Course Planner Agent.

Your tasks:
1. Select and call the appropriate tools to build or modify a DTU MSc study plan.
2. NEVER output chain-of-thought. Only output final answers.
3. ALWAYS return the result STRICTLY in the format provided by the tool.
4. NEVER invent, regenerate, rewrite, or restructure JSON yourself. 
5. After a tool call returns a result, your ONLY job is to:
   - Read the tool output (JSON)
   - Return it EXACTLY as-is to the user
   - Do NOT add text outside the JSON
   - Do NOT summarize again in your own words
   - Do NOT rename keys or restructure the object
   
STRICT OUTPUT RULES:
- The final assistant message MUST be valid JSON.
- It MUST contain the keys: "summary", "total_ects", "periods", "buffer_courses".
- It MUST NOT contain any additional commentary, greetings, or natural language.
- It MUST NOT wrap the JSON in explanations or markdown.
- It MUST NOT reduce the response to only {"plan": {...}} or omit required fields.

Your mission:
- Ensure a stable, predictable JSON format.
- Reject incomplete formats and ALWAYS return exactly the tool JSON.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_relevant_courses",
            "description": "Search for top matching DTU courses using hybrid similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "mode": {"type": "string", "default": "hybrid"},
                    "alpha": {"type": "number", "default": 0.2},
                },
                "required": ["query", "top_k"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "filter_courses_by_keyword",
            "description": "Filter DTU courses by keyword in selected columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["keyword", "columns"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "keep_columns",
            "description": "Keep only specific columns from a DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["column_names"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "add_ECTS",
            "description": "Sum ECTS of given courses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ---------------- UPDATED TOOL SCHEMA ----------------
    {
        "type": "function",
        "function": {
            "name": "plan_msc_program",
            "description": "Build a 2-year MSc plan matching schedule, ECTS constraints, mandatory and forbidden courses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ects_target": {"type": "number", "default": 85.0},
                    "top_k": {"type": "integer", "default": 80},
                    "alpha": {"type": "number", "default": 0.2},
                    "mandatory_courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "forbidden_courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# --------------------------------------------------------
# Convert sets → lists for JSON serialization
# --------------------------------------------------------
def convert_sets(obj):
    if isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj

def print_agent_response(ans: str):
    """
    Pretty-print the agent response if it's JSON.
    Otherwise print normally.
    """
    try:
        data = json.loads(ans)

        # If the result contains a summary → print the readable summary
        if "summary" in data:
            print("\n==================== STUDY PLAN ====================\n")
            print(data["summary"])
            print("\n====================================================\n")

        # Print full JSON pretty-formatted
        print("\n----------- RAW STRUCTURED JSON (for debugging) -----------")
        print(json.dumps(data, indent=2))
        print("-----------------------------------------------------------\n")

    except Exception:
        # Fallback: just print as-is
        print(ans)
# --------------------------------------------------------
# Tool executor
# --------------------------------------------------------
def execute_tool(tool_name, arguments):
    if tool_name == "search_relevant_courses":
        df = COURSES_DF.copy()
        res = search_relevant_courses(
            df=df,
            query=arguments["query"],
            top_k=arguments["top_k"],
            mode=arguments.get("mode", "hybrid"),
            alpha=arguments.get("alpha", 0.2),
        )
        return res[["course_code"]].to_json(orient="records")

    if tool_name == "filter_courses_by_keyword":
        df = COURSES_DF.copy()
        filt = filter_courses_by_keyword(
            df,
            keyword=arguments["keyword"],
            columns=arguments["columns"],
        )
        return filt[["course_code"]].to_json(orient="records")

    if tool_name == "keep_columns":
        df = COURSES_DF.copy()
        kept = keep_columns(arguments["column_names"], df)
        return kept.to_json(orient="records")

    if tool_name == "add_ECTS":
        df = COURSES_DF.copy()
        ects = add_ECTS(df)
        return json.dumps({"total_ects": ects})

    # ---------------- UPDATED: pass forbidden courses ----------------
    if tool_name == "plan_msc_program":
        df = COURSES_DF.copy()
        plan = plan_msc_program(
            df=df,
            query=arguments["query"],
            ects_target=arguments.get("ects_target", 85.0),
            top_k=arguments.get("top_k", 80),
            alpha=arguments.get("alpha", 0.2),
            mandatory_courses=arguments.get("mandatory_courses", []),
            forbidden_courses=arguments.get("forbidden_courses", []),
        )

        # Slim down the data
        periods_slim = {}
        for pid, info in plan["periods"].items():
            periods_slim[pid] = {
                "season": info["season"],
                "ects_used": info["ects_used"],
                "max_ects": info["max_ects"],
                "courses": [
                    row["course_code"]
                    for row in info["courses_df"].to_dict(orient="records")
                ],
            }

        # Build the human-readable summary
        summary = []
        summary.append(f"Total ECTS planned: {plan['total_ects']:.1f}")

        for pid, pinfo in periods_slim.items():
            summary.append(f"\n{pid} ({pinfo['season']}, max {pinfo['max_ects']} ECTS):")
            if not pinfo["courses"]:
                summary.append("  - No courses")
            else:
                for code in pinfo["courses"]:
                    row = COURSES_DF[COURSES_DF["course_code"] == code].iloc[0]
                    title = row["title"]
                    ects = float(str(row["Point( ECTS )"]).replace(",", "."))
                    summary.append(f"  - {code} {title} ({ects} ECTS)")

        summary_text = "\n".join(summary)

        result = {
            "summary": summary_text,
            "total_ects": plan["total_ects"],
            "periods": periods_slim,
            "buffer_courses": plan["buffer_courses"][["course_code"]].to_dict(
                orient="records"
            ),
        }

        result = convert_sets(result)
        return json.dumps(result)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})

# --------------------------------------------------------
# Main agent
# --------------------------------------------------------
def ask_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Force planner only on the FIRST turn
    force_tool_once = "plan" in user_message.lower() or "msc" in user_message.lower()

    while True:
        tool_choice = (
            {"type": "function", "function": {"name": "plan_msc_program"}}
            if force_tool_once
            else "auto"
        )

        response = client.chat.completions.create(
            model="Granite4",
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            max_tokens=800,
            temperature=0.1,
        )

        msg = response.choices[0].message

        # TOOL CALL?
        if msg.tool_calls:
            print("\n[MODEL DECISION]")
            print("Tool selected:", msg.tool_calls[0].function.name)
            print("Arguments:", msg.tool_calls[0].function.arguments)
            force_tool_once = False  # next turn: allow normal output
            tool_call = msg.tool_calls[0]
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"\n[Executing tool: {name}]")
            print("Model responded with:", msg)

            result = execute_tool(name, args)

            messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
            continue

        return msg.content or ""


# --------------------------------------------------------
# Run test
# --------------------------------------------------------
if __name__ == "__main__":
    ans = ask_agent(
        "Create an MSc plan specializing in robotics and data science. "
        "I do not want to include course 42580 Introduction to Data Science."
    )
    print("\nAGENT RESPONSE:\n")
    print_agent_response(ans)
