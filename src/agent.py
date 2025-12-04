# ================================================================
# run_agent.py — Build initial plan, pass to LLM, get refined plan
# ================================================================

import os
import pandas as pd
from openai import OpenAI

from src.dataloader import load_data
from src.planner import plan_msc_program, print_plan


# -------------------------------------------------------------
# Load API key & client
# -------------------------------------------------------------
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
if not CAMPUSAI_API_KEY:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://chat.campusai.compute.dtu.dk/api/v1",
)


# -------------------------------------------------------------
# Load dataframe once
# -------------------------------------------------------------
COURSES_DF = load_data()


# -------------------------------------------------------------
# SYSTEM PROMPT — buffers already inside the plan
# -------------------------------------------------------------
SYSTEM_PROMPT = """
You are an intelligent MSc course-planning agent for DTU.

You receive:
- A user request describing their preferences.
- A full MSc course plan as a SINGLE string, already containing:
    • All scheduled courses
    • Buffer courses

Your goals:
1. Ensure the selected courses are of HIGH RELEVANCE to the chosen MSc programme and to the student's stated interests.
2. Read and understand the existing plan string.
3. Modify the plan ONLY when necessary, and ONLY by using buffer courses provided in the plan.
4. Enforce strict feasibility rules:
   - A maximum ONE course per time period per semester.
   - No repeated courses.
   - Total ECTS must NOT exceed 90.
   - No overlap between periods.
5. Maintain the **existing formatting exactly** for the plan section.
6. When you modify the plan:
   - At the end of your response, include a short section titled:
     `CHANGES MADE:`  
     where you list what changed (if anything). Include the names and time periodes of the courses added/removed.
   - Example:
     `CHANGES MADE: Added: Course A, Course B; Removed: Course C`
7. If no changes are required, return the plan unchanged and write:
   `CHANGES MADE: None`

- Return always the total ECTS of the final plan.
Make special sure to follow these instructions EXACTLY. Do NOT invent new courses or modify course names/IDs.
Do NOT exceed the ECTS limit. The total planned ECTS must be at most 90.

Output format:
- First output the UPDATED PLAN STRING exactly in its original format.
- Then output the `CHANGES MADE:` section.
- No other commentary. No markdown. No JSON.

Always return the FULL UPDATED PLAN in a format like the received one, even if no changes were made.
"""


# -------------------------------------------------------------
# 1. BUILD INITIAL PLAN (UPDATED FOR YOUR SIGNATURE)
# -------------------------------------------------------------
def build_initial_plan(df, query, ects_target=85.0, forbidden=None):
    if forbidden is None:
        forbidden = []

    plan_dict = plan_msc_program(
        df=df,
        query=query,
        ects_target=ects_target,
        forbidden_courses=forbidden,
    )

    # Convert dict → plan string (this includes buffer courses)
    plan_string = print_plan(plan_dict)
    return plan_string


# -------------------------------------------------------------
# 2. SEND PLAN TO LLM
# -------------------------------------------------------------
def refine_plan_with_llm(user_request, plan_string):
    response = client.chat.completions.create(
        model="DeepSeek-R1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
            {"role": "user", "content": "PLAN:\n" + plan_string},
        ],
    )
    return response.choices[0].message.content



# -------------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------------
if __name__ == "__main__":

    user_prompt = """
    I want more robotics and AI courses.
    Remove less relevant courses if needed and use the buffer courses at the end of the plan.
    """

    # 1. Build plan using your updated plan_msc_program signature
    initial_plan_string = build_initial_plan(
        df=COURSES_DF,
        query="robotics data science",
        ects_target=120,
        forbidden=["42580"],
    )

    # 2. Refine using LLM
    refined_plan = refine_plan_with_llm(
        user_request=user_prompt,
        plan_string=initial_plan_string,
    )

    print("\n----- REFINED PLAN -----\n")
    print(refined_plan)
