import json
from fastapi import FastAPI, Query
from typing import Optional
from src.agent import (
    build_initial_plan,
    refine_plan_with_llm,
    COURSES_DF,
)

app = FastAPI(
    title="DTU MSc Planner API",
    version="1.0.0",
    description="API wrapper around the DTU MSc planning agent"
)

def prettify(plan_text: str):
    # Split into lines for readability
    return plan_text.split("\n")

# ---------------------- ENDPOINTS ----------------------
@app.get("/v1/build_plan")
async def build_plan(
    query: str = Query(..., description="Keywords describing user's interests"),
    user_request: str = Query(..., description="User modification request for the plan, any courses that you dont like, etc."),
    forbidden: Optional[str] = Query(None, description="Comma-separated list of course IDs to exclude"),
    mandatory: Optional[str] = Query(None, description="Comma-separated list of mandatory course IDs"),
):
    """
    Ask an AI agent to build a MSc study plan based on user query and preferences.
    An LLM will then oversee the initial plan and make adjustments based on the user request.
    [Be patient - this may take a bit of time to process :) ]
    """

    # Parse forbidden list
    forbidden_list = []
    if forbidden:
        forbidden_list = [c.strip() for c in forbidden.split(",") if c.strip()]

    # Parse mandatory list
    mandatory_list = []
    if mandatory:
        mandatory_list = [c.strip() for c in mandatory.split(",") if c.strip()]
        
    # 1. Build initial plan
    initial_plan = build_initial_plan(
        df=COURSES_DF,
        query=query,
        ects_target=85,
        mandatory=mandatory_list,
        forbidden=forbidden_list,
    )

    # 2. Refine with LLM
    refined = refine_plan_with_llm(
        user_request=query+user_request,
        plan_string=initial_plan,
    )

    # API returns plain text
    return {
        "refined_plan": prettify(refined),
        "initial_plan": prettify(initial_plan)
    }


# ---------------------- MAIN SERVER ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
