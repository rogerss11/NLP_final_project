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


# ---------------------- ENDPOINTS ----------------------
@app.get("/v1/build_plan")
async def build_plan(
    query: str = Query(..., description="Keywords describing user's interests"),
    user_request: str = Query(..., description="User modification request for the plan, any courses that you dont like, etc."),
    ects_target: float = Query(85.0, description="ECTS target for initial plan"),
    forbidden: Optional[str] = Query(None, description="Comma-separated list of course IDs to exclude"),
):
    """
    Build initial plan → refine with LLM → return final plan.
    """

    # Parse forbidden list
    forbidden_list = []
    if forbidden:
        forbidden_list = [c.strip() for c in forbidden.split(",") if c.strip()]

    # 1. Build initial plan
    initial_plan = build_initial_plan(
        df=COURSES_DF,
        query=query,
        ects_target=ects_target,
        forbidden=forbidden_list,
    )

    # 2. Refine with LLM
    refined = refine_plan_with_llm(
        user_request=query+user_request,
        plan_string=initial_plan,
    )

    # API returns plain text
    return {
        "initial_plan": initial_plan,
        "refined_plan": refined,
    }


# ---------------------- MAIN SERVER ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
