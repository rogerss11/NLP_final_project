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

FORBIDDEN_COURSES = []
# ==================================
def normalize_schedule_value(val) -> str:
    if isinstance(val, list):
        text = " ".join(map(str, val))
    else:
        text = str(val)
    return text.lower()

def trim_title(title: str) -> str:
    """Remove the first 6 characters (course code prefix)."""
    return title[6:].strip().lower() if len(title) > 6 else title.lower()

def extract_seasons_and_slots(schedule) -> Tuple[Set[str], Set[str]]:
    text = normalize_schedule_value(schedule)

    seasons = {
        season.lower()
        for season in SEASON_KEYWORDS
        if season.lower() in text
    }

    slots = {m.group(0).lower() for m in SLOT_PATTERN.finditer(text)}

    return seasons, slots

def schedules_overlap(slots_a: Set[str], slots_b: Set[str]) -> bool:
    return len(slots_a & slots_b) > 0

def print_plan(plan: dict):
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
            print(df_p[["course_code", "title", "Point( ECTS )", "Schedule", "score"]])


# ==================================
def title_seen_before(trimmed_title: str, assigned_titles: Set[str], df: pd.DataFrame) -> bool:
    """
    Use your keyword search logic to find duplicates.
    trimmed_title = title without first 6 characters.
    """
    if not trimmed_title:
        return False

    # Search all courses whose title contains this trimmed part
    dup_df = filter_courses_by_keyword(df, trimmed_title, ["title"])

    # Check if any already-assigned course title is present in dup_df
    for t in assigned_titles:
        matches = dup_df[dup_df["trimmed_title"] == t]
        if len(matches) > 0:
            return True

    return False


def plan_msc_program(
    df: pd.DataFrame,
    query: str,
    ects_target: float = 85.0,
    top_k: int = 200,
    alpha: float = 0.2,
    mandatory_courses: List[str] = MANDATORY_COURSES,
    forbidden_courses: List[str] = FORBIDDEN_COURSES,
) -> dict:
    """
    Retrieve relevant courses and build a 2-year MSc plan respecting:
        - mandatory courses (inserted first)
        - season schedule
        - ECTS caps per period
        - schedule slot non-overlap
        - NO duplicates (based on course title)
        - EXCLUSION of forbidden courses (by code or title)
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

    # Remove forbidden courses by code or title substring
    forb_keys = {f.lower() for f in forbidden_courses}

    def is_forbidden(row):
        title_lower = row["title"].lower()
        return (
            row["course_code"] in forbidden_courses
            or any(k in title_lower for k in forb_keys)
        )

    before = len(candidates)
    candidates = candidates[~candidates.apply(is_forbidden, axis=1)].copy()
    after = len(candidates)
    if before != after:
        print(f" - Removed {before - after} forbidden courses from candidates.")

    # Precompute seasons, slots, ECTS, and trimmed titles
    seasons_list, slots_list, ects_list, trimmed_titles = [], [], [], []

    for _, row in candidates.iterrows():
        seasons, slots = extract_seasons_and_slots(row.get("Schedule", ""))
        seasons_list.append(seasons)
        slots_list.append(slots)
        ects_list.append(get_course_ects(row))
        trimmed_titles.append(trim_title(row["title"]))

    candidates["parsed_seasons"] = seasons_list
    candidates["parsed_slots"] = slots_list
    candidates["ects"] = ects_list
    candidates["trimmed_title"] = trimmed_titles

    # ----------------------------
    # 2) Period initialization
    # ----------------------------
    periods = {}
    for p in PROGRAM_PATTERN:
        periods[p["id"]] = {
            "season": p["season"].lower(),
            "max_ects": float(p["max_ects"]),
            "ects_used": 0.0,
            "courses": [],
            "taken_slots": set(),
        }

    total_ects = 0.0
    assigned_titles = set()  # store trimmed titles only

    # =========================================================================
    # 3) Insert Mandatory Courses
    # =========================================================================
    mandatory_df = candidates[candidates["course_code"].isin(mandatory_courses)]

    for idx, row in mandatory_df.iterrows():

        trimmed = row["trimmed_title"]
        if trimmed in assigned_titles:
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

            assigned_titles.add(trimmed)
            total_ects += course_ects
            placed = True
            break

        if not placed:
            print(f"⚠️ WARNING: Mandatory course '{row['title']}' could not be placed!")

    # =========================================================================
    # 4) Greedy Fill with Remaining Courses
    # =========================================================================
    non_mandatory = candidates[~candidates["course_code"].isin(mandatory_courses)]

    for idx, row in non_mandatory.iterrows():

        trimmed = row["trimmed_title"]
        if title_seen_before(trimmed, assigned_titles, candidates):
            continue

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

            assigned_titles.add(trimmed)
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
    # 6) Buffer of unused but relevant courses
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
        query="Create an MSc plan specializing in robotics and data science.",
        ects_target=85.0,
        top_k=300,
        alpha=0.2,
    )

    print("Total planned ECTS:", plan["total_ects"])
    print_plan(plan)
    print("\nBuffer of other relevant courses:")
    buffer_df = plan["buffer_courses"]
    print(buffer_df[["course_code", "title", "Point( ECTS )", "Schedule", "score"]])