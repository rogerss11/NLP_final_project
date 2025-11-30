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

MANDATORY_COURSES = ["12105", "42500"]
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
        - mandatory courses (inserted first)
        - season schedule
        - ECTS caps per period
        - schedule slot non-overlap
        - NO duplicates (based on course title)
    """

    # ----------------------------
    # 1) Retrieve relevant courses
    # ----------------------------
    candidates = search_relevant_courses(
        df=df,
        query=query,
        top_k=top_k,
        mode="hybrid",
        alpha=alpha,
    ).copy()

    # ----------------------------
    # Precompute season + slot + ECTS
    # ----------------------------
    seasons_list, slots_list, ects_list = [], [], []
    for _, row in candidates.iterrows():
        seasons, slots = extract_seasons_and_slots(row.get("Schedule", ""))
        seasons_list.append(seasons)
        slots_list.append(slots)
        ects_list.append(get_course_ects(row))

    candidates["parsed_seasons"] = seasons_list
    candidates["parsed_slots"] = slots_list
    candidates["ects"] = ects_list

    # ----------------------------
    # 2) Periods initialization
    # ----------------------------
    periods = {}
    for p in PROGRAM_PATTERN:
        periods[p["id"]] = {
            "season": p["season"].lower(),
            "max_ects": float(p["max_ects"]),
            "ects_used": 0.0,
            "courses": [],         # store DataFrame indices
            "taken_slots": set(),  # union of slot codes
        }

    total_ects = 0.0

    # ----------------------------
    # Tracking for duplicates (by title)
    # ----------------------------
    assigned_titles = set()

    # =========================================================================
    # 3) INSERT MANDATORY COURSES FIRST
    # =========================================================================
    mandatory_df = candidates[candidates["course_code"].isin(MANDATORY_COURSES)]

    for idx, row in mandatory_df.iterrows():

        title_key = row["title"].strip().lower()
        if title_key in assigned_titles:
            continue

        course_ects = row["ects"]
        course_seasons = row["parsed_seasons"]
        course_slots = row["parsed_slots"]

        placed = False

        for p in PROGRAM_PATTERN:
            pid = p["id"]
            period = periods[pid]

            if period["season"] not in course_seasons:
                continue
            if period["ects_used"] + course_ects > period["max_ects"]:
                continue
            if schedules_overlap(period["taken_slots"], course_slots):
                continue

            period["courses"].append(idx)
            period["ects_used"] += course_ects
            period["taken_slots"] |= course_slots

            assigned_titles.add(title_key)
            total_ects += course_ects
            placed = True
            break

        if not placed:
            print(f"⚠️ WARNING: Mandatory course '{row['title']}' could not be placed!")

    # =========================================================================
    # 4) GREEDY FILL WITH REMAINING CANDIDATES
    # =========================================================================
    non_mandatory = candidates[~candidates["course_code"].isin(MANDATORY_COURSES)]

    for idx, row in non_mandatory.iterrows():

        title_key = row["title"].strip().lower()
        if title_key in assigned_titles:
            continue  # skip duplicates by title

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

            if period["season"] not in course_seasons:
                continue
            if period["ects_used"] + course_ects > period["max_ects"]:
                continue
            if schedules_overlap(period["taken_slots"], course_slots):
                continue

            period["courses"].append(idx)
            period["ects_used"] += course_ects
            period["taken_slots"] |= course_slots

            assigned_titles.add(title_key)
            total_ects += course_ects
            assigned = True
            break

        if total_ects >= ects_target:
            break

    # ----------------------------
    # 5) Build DataFrames per period
    # ----------------------------
    selected_indices = set()
    for pid, period in periods.items():
        if period["courses"]:
            selected_indices |= set(period["courses"])
            period["courses_df"] = candidates.loc[period["courses"]].copy()
        else:
            period["courses_df"] = candidates.iloc[0:0].copy()

    # ----------------------------
    # 6) Buffer of all unused but relevant courses
    # ----------------------------
    buffer_df = candidates.loc[
        [i for i in candidates.index if i not in selected_indices]
    ].copy()

    buffer_df = buffer_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    return {
        "periods": periods,
        "total_ects": total_ects,
        "retrieved_candidates": candidates,
        "buffer_courses": buffer_df,
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