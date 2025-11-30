import pandas as pd
import re
from typing import List, Tuple, Set

from src.dataloader import load_data
from src.functions import (
    search_relevant_courses,
    get_course_ects,
    keep_columns,
    filter_courses_by_keyword,
    add_ECTS)

# ============= CONFIG ==========
PROGRAM_PATTERN = [
    {"id": "Autumn1",  "season": "Autumn",  "max_ects": 30},
    {"id": "January1", "season": "January", "max_ects": 5},
    {"id": "Spring1",  "season": "Spring",  "max_ects": 30},
    {"id": "June",     "season": "June",    "max_ects": 5},
    {"id": "July",     "season": "July",    "max_ects": 5},
    {"id": "August",   "season": "August",  "max_ects": 5},
    {"id": "Autumn2",  "season": "Autumn",  "max_ects": 30},
    {"id": "January2", "season": "January", "max_ects": 5},
]


SEASON_KEYWORDS = ["Autumn", "Spring", "January", "June", "July", "August"]

SLOT_PATTERN = re.compile(r"\b[FE]\d+[A-Z]?\b", re.IGNORECASE)

# ==================================
def normalize_schedule_value(val) -> str:
    """Turn a Schedule cell (str or list) into a single lowercase string."""
    if isinstance(val, list):
        text = " ".join(map(str, val))
    else:
        text = str(val)
    return text.lower()

def extract_seasons_and_slots(schedule) -> Tuple[Set[str], Set[str]]:
    """
    From a Schedule cell, extract:
      - seasons: set like {"autumn", "spring"}
      - slots: set like {"f1a", "f2"}
    """
    text = normalize_schedule_value(schedule)

    seasons = {
        season.lower()
        for season in SEASON_KEYWORDS
        if season.lower() in text
    }

    slots = {m.group(0).lower() for m in SLOT_PATTERN.finditer(text)}

    return seasons, slots

def schedules_overlap(slots_a: Set[str], slots_b: Set[str]) -> bool:
    """Return True if two courses share any time-slot token."""
    return len(slots_a & slots_b) > 0

def print_plan(plan: dict):
    """Utility to print a planned MSc program."""
    print("Total planned ECTS:", plan["total_ects"])

    for pid in PROGRAM_PATTERN:
        pid = pid["id"]
        period = plan["periods"][pid]
        print("\n=== Period:", pid, "| season:", period["season"], "===")
        print("ECTS used:", period["ects_used"], "/", period["max_ects"])

        df_p = period["courses_df"]
        if df_p.empty:
            print("No courses assigned.")
        else:
            print(df_p[["course_code", "title", "Point( ECTS )", "Schedule", "score"]]
                  )
            
# ==================================
def plan_msc_program(
    df: pd.DataFrame,
    query: str,
    ects_target: float = 85.0,
    top_k: int = 200,
    alpha: float = 0.2,
) -> dict:
    """
    Retrieve relevant courses and build a 2-year MSc plan respecting:
        - season schedule
        - ECTS caps per period
        - schedule slot non-overlap
    
    Includes a buffer list of other relevant courses for LLM adjustments.
    """

    # ----------------------------
    # 1) Search relevant courses
    # ----------------------------
    candidates = search_relevant_courses(
        df=df,
        query=query,
        top_k=top_k,
        mode="hybrid",
        alpha=alpha,
    ).copy()

    # ----------------------------
    # Precompute ECTS + schedule parsing
    # ----------------------------
    seasons_list = []
    slots_list = []
    ects_list = []

    for _, row in candidates.iterrows():
        seasons, slots = extract_seasons_and_slots(row.get("Schedule", ""))
        seasons_list.append(seasons)
        slots_list.append(slots)
        ects_list.append(get_course_ects(row))

    candidates["parsed_seasons"] = seasons_list
    candidates["parsed_slots"] = slots_list
    candidates["ects"] = ects_list

    # ----------------------------
    # 2) Prepare periods state
    # ----------------------------
    periods = {}
    for p in PROGRAM_PATTERN:
        periods[p["id"]] = {
            "season": p["season"].lower(),
            "max_ects": float(p["max_ects"]),
            "ects_used": 0.0,
            "courses": [],         # store candidate indices
            "taken_slots": set(),  # union of slot tokens
        }

    total_ects = 0.0

    # ----------------------------
    # 3) Greedy assignment
    # ----------------------------
    for idx, row in candidates.iterrows():
        course_ects = row["ects"]
        if course_ects <= 0:
            continue

        course_seasons = row["parsed_seasons"]
        course_slots = row["parsed_slots"]

        if not course_seasons:
            continue

        assigned = False

        for p in PROGRAM_PATTERN:
            pid = p["id"]
            period = periods[pid]

            # Must match season
            if period["season"] not in course_seasons:
                continue

            # ECTS capacity
            if period["ects_used"] + course_ects > period["max_ects"]:
                continue

            # No schedule overlap
            if schedules_overlap(period["taken_slots"], course_slots):
                continue

            # Assign
            period["courses"].append(idx)
            period["ects_used"] += course_ects
            period["taken_slots"] |= course_slots

            total_ects += course_ects
            assigned = True
            break

        if total_ects >= ects_target:
            break

    # ----------------------------
    # 4) Build DataFrames per period
    # ----------------------------
    selected_indices = set()

    for pid, period in periods.items():
        if period["courses"]:
            selected_indices |= set(period["courses"])
            period["courses_df"] = candidates.loc[period["courses"]].copy()
        else:
            period["courses_df"] = candidates.iloc[0:0].copy()  # empty df

    # ----------------------------
    # 5) Buffer of ALL OTHER relevant courses
    # ----------------------------
    buffer_df = candidates.loc[
        [i for i in candidates.index if i not in selected_indices]
    ].copy()

    # Sort buffer by score (descending)
    buffer_df = buffer_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # ----------------------------
    # Final return
    # ----------------------------
    return {
        "periods": periods,
        "total_ects": total_ects,
        "retrieved_candidates": candidates,   # ALL ranked results
        "buffer_courses": buffer_df,          # UNUSED but relevant
    }

if __name__ == "__main__":
    courses_df = load_data()
    plan = plan_msc_program(
        df=courses_df,
        query="Data Science and Machine Learning",
        ects_target=85.0,
        top_k=300,
        alpha=0.2,
    )

    print("Total planned ECTS:", plan["total_ects"])
    
    print_plan(plan)
    print("\nBuffer of other relevant courses:")
    buffer_df = plan["buffer_courses"]
    print(buffer_df[["course_code", "title", "Point( ECTS )", "Schedule", "score"]])