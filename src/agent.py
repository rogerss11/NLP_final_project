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

You always receive:
• A user request describing their preferences.
• A COMPLETE MSc course plan represented as ONE raw text string.  
  This plan already contains:
    - All scheduled courses
    - A list of buffer courses

YOUR TASK
You must revise the plan **strictly according to the user’s request**, respecting the constraints below.

RULES YOU MUST FOLLOW
1. **Relevance**  
   Ensure all selected courses are highly relevant to the MSc programme and the user's stated interests.

2. **Use ONLY the provided plan string**  
   - Read and understand the full plan.  
   - You may ONLY introduce courses that appear in the buffer list of the provided plan.  
   - NEVER invent courses, change names, change IDs, or modify periods/ECTS of existing courses.

3. **Modification constraints**  
   You may modify the plan ONLY by:
   - Replacing a scheduled course with a buffer course, OR  
   - Removing or adding buffer courses where allowed.  
   You must NOT create new buffer courses.

4. **Feasibility rules** (strict):
   - Max **one** course per time period per semester.
   - No repeated courses anywhere.
   - No overlapping time periods.
   - Total ECTS ≤ **90** under all circumstances.
   - Always respect course metadata (IDs, names, ECTS, periods) as given.
   - Do NOT add extra semesters or periods beyond those in the original plan.
   - If you remove a course, you must add a buffer course to keep the plan full, if possible.

5. **Formatting rules** (very strict):
   - You must output the UPDATED PLAN STRING **exactly in the same structure and formatting** as the input plan.
   - Do NOT add markdown, explanations, or JSON.
   - The plan always comes first in the output.

6. **Mandatory final section**
   After the updated plan string, append a section titled exactly:

   CHANGES MADE:
   - List all added and removed courses with IDs, names, ECTS, and periods.
   - If nothing changed, write:  
     `CHANGES MADE: No changes.`

   Example format (do NOT use markdown):  
   CHANGES MADE: Added: Course A (5 ECTS, Autumn1), Course B (10 ECTS, Spring1); Removed: Course C (5 ECTS, January)

7. **Total ECTS**  
   You must ALWAYS calculate and return the total ECTS of the final updated plan in the same format as the original plan.

ABSOLUTE REQUIREMENTS
- Follow all instructions EXACTLY.
- Never exceed 90 ECTS.
- Never invent or alter course metadata.
- Always return the FULL updated plan, even if no changes were made.

"""


# -------------------------------------------------------------
# 1. BUILD INITIAL PLAN (UPDATED FOR YOUR SIGNATURE)
# -------------------------------------------------------------
def build_initial_plan(df, query, ects_target=85.0, mandatory=None,forbidden=None):
    if forbidden is None:
        forbidden = []

    plan_dict = plan_msc_program(
        df=df,
        query=query,
        ects_target=ects_target,
        mandatory_courses=mandatory,
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
