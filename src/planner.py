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

def print_plan(plan: dict) -> str:
    lines = []
    lines.append(f"Total planned ECTS: {plan['total_ects']}")

    # ---------- PERIODS ----------
    for p in PROGRAM_PATTERN:
        pid = p["id"]
        period = plan["periods"][pid]

        lines.append(f"\n=== Period: {pid} | season: {period['season']} ===")
        lines.append(f"ECTS used: {period['ects_used']} / {period['max_ects']}")

        df_p = period["courses_df"]

        if df_p.empty:
            lines.append("No courses assigned.")
        else:
            for _, r in df_p.iterrows():
                lines.append(
                    f"  {r.course_code} | {r.title} | {r['Point( ECTS )']} ECTS | "
                    f"{r.Schedule} | score={r.score:.3f}"
                )

    # ---------- BUFFER COURSES ----------
    buffer_df = plan.get("buffer_courses", None)

    lines.append("\n=== Buffer of Other Relevant Courses ===")

    if buffer_df is None or buffer_df.empty:
        lines.append("No buffer courses.")
    else:
        for _, r in buffer_df.iterrows():
            lines.append(
                f"  {r.course_code} | {r.title} | {r['Point( ECTS )']} ECTS | "
                f"{r.Schedule} | score={r.score:.3f}"
            )

    return "\n".join(lines)



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

    The plan is returned as a dictionary with the following structure:

    plan["periods"] = {
        "Autumn1": {
            "season": "autumn",
            "max_ects": 30.0,
            "ects_used": 25.0,
            "courses": [12, 55, 88],     # course code
            "taken_slots": {"f1a", "f2b"},
            "courses_df": <DataFrame with selected courses>
        },
        "January1": {
            ...
        },
        ...
    }

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
    # if before != after:
        # print(f" - Removed {before - after} forbidden courses from candidates.")

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
    buffer_df = buffer_df.head(15)

    return {
        "periods": periods,
        "total_ects": total_ects,
        "retrieved_candidates": candidates,
        "buffer_courses": buffer_df,
    }

# ================== FUNCTIONS FOR MODIFYING AN EXISTING PLAN ==================

def replace_course(course_name_or_id: str, plan: dict, df: pd.DataFrame = None):
    """
    Replace a specified course in the existing plan with an alternative from buffer_courses.
    Searches by course code or title substring.

    Parameters:
        course_name_or_id : str
            Course identifier (exact code) or part of the title.
        plan : dict
            The plan returned from plan_msc_program().
        df  : pd.DataFrame (optional)
            Full DF if needed; if not passed, retrieved_candidates inside the plan is used.

    Returns:
        Updated plan (modified in-place).
    """

    # --- Data Sources ---
    candidates = plan["retrieved_candidates"]
    buffer_df = plan["buffer_courses"]

    # Choose search DF
    source_df = candidates if df is None else df

    # Normalize search key
    key = course_name_or_id.lower()

    # ---------------------------------------------------------------------
    # 1. Locate the course in the plan (search by exact code or title match)
    # ---------------------------------------------------------------------
    target = None
    target_period_id = None

    for pid, period in plan["periods"].items():
        df_period = period["courses_df"]

        if df_period.empty:
            continue

        # Find rows where course_code == key or title contains key
        match_df = df_period[
            (df_period["course_code"].str.lower() == key)
            | (df_period["title"].str.lower().str.contains(key))
        ]

        if len(match_df) > 0:
            target = match_df.iloc[0]  # take the first match
            target_period_id = pid
            break

    if target is None:
        print(f"❌ Course '{course_name_or_id}' not found in the plan.")
        return plan

    # Extract data
    idx = target.name           # DataFrame index in candidates
    ects = target["ects"]
    slots = target["parsed_slots"]

    print(f"🔎 Found course '{target['title']}' in period {target_period_id}. Replacing it...")

    # ---------------------------------------------------------------------
    # 2. Remove the course from its period
    # ---------------------------------------------------------------------
    period = plan["periods"][target_period_id]

    # Remove from course list
    period["courses"].remove(idx)
    period["ects_used"] -= ects
    period["taken_slots"] -= slots  # free occupied slots

    # Rebuild the period DataFrame
    if period["courses"]:
        period["courses_df"] = candidates.loc[period["courses"]].copy()
    else:
        period["courses_df"] = candidates.iloc[0:0].copy()

    # Add course back to buffer
    buffer_df = pd.concat([buffer_df, target.to_frame().T], ignore_index=True)
    buffer_df = buffer_df.sort_values("score", ascending=False).reset_index(drop=True)
    plan["buffer_courses"] = buffer_df

    # ---------------------------------------------------------------------
    # 3. Try inserting a replacement from buffer
    # ---------------------------------------------------------------------
    season = period["season"]
    taken_slots = period["taken_slots"]

    def is_duplicate(trimmed_title):
        # check duplication using candidates trimmed_title
        return trimmed_title in set(
            candidates.loc[period["courses"]]["trimmed_title"].values
        )

    for i, row in buffer_df.iterrows():

        # Season match
        if season not in row["parsed_seasons"]:
            continue

        # ECTS fit
        if period["ects_used"] + row["ects"] > period["max_ects"]:
            continue

        # Slot conflict
        if len(taken_slots & row["parsed_slots"]) > 0:
            continue

        # Duplicate title check
        if is_duplicate(row["trimmed_title"]):
            continue

        # --- SUCCESS: Insert replacement ---
        new_idx = row.name  # index in candidates
        period["courses"].append(new_idx)
        period["ects_used"] += row["ects"]
        period["taken_slots"] |= row["parsed_slots"]
        plan["total_ects"] = plan["total_ects"] - ects + row["ects"]

        period["courses_df"] = candidates.loc[period["courses"]].copy()

        # Remove from buffer
        plan["buffer_courses"] = buffer_df.drop(i).reset_index(drop=True)

        print(f"✅ Replaced with '{row['title']}'")
        return plan

    print("⚠️ No valid replacement found from buffer.")
    return plan

def introduce_course(new_course: str, plan: dict, df: pd.DataFrame = None):
    """
    Force a course into the existing plan. If schedule conflicts or ECTS limits
    prevent placement, conflicting courses are removed and pushed to the buffer
    until insertion becomes feasible.

    Parameters:
        new_course : str
            Course code or part of course title.
        plan : dict
            The MSc plan returned by plan_msc_program().
        df : pd.DataFrame (optional)
            If provided, use this DF; otherwise use plan["retrieved_candidates"].

    Returns:
        Updated plan (modified in-place).
    """
    candidates = plan["retrieved_candidates"]
    source_df = df if df is not None else candidates
    buffer_df = plan["buffer_courses"]

    key = new_course.lower()

    # ------------------------------------------------------
    # 1. Find the course by code or title substring
    # ------------------------------------------------------
    match = source_df[
        (source_df["course_code"].str.lower() == key)
        | (source_df["title"].str.lower().str.contains(key))
    ]

    if match.empty:
        print(f"❌ Course '{new_course}' not found in candidates.")
        return plan

    row = match.iloc[0]
    idx = row.name
    ects = row["ects"]
    trimmed = row["trimmed_title"]
    seasons = row["parsed_seasons"]
    slots = row["parsed_slots"]

    print(f"📌 Forcing introduction of course: {row['title']}")

    # ------------------------------------------------------
    # 2. Try to place in any valid period based on season
    # ------------------------------------------------------
    for pid, period in plan["periods"].items():

        if period["season"] not in seasons:
            continue  # season mismatch

        print(f"➡️ Attempting to place in period: {pid}")

        # --- Step A: Ensure no duplicate titles ---
        existing_titles = set(
            period["courses_df"]["trimmed_title"].values
        )
        if trimmed in existing_titles:
            print(f"⚠️ Title already present in this period. Skipping {pid}.")
            continue

        # --- Step B: Remove conflicting slots ---
        conflicts = []
        for cidx in period["courses"]:
            c = candidates.loc[cidx]
            if len(c["parsed_slots"] & slots) > 0:
                conflicts.append(cidx)

        # Remove slot conflicts
        for cidx in conflicts:
            removed_course = candidates.loc[cidx]
            print(f"🔄 Removing conflicting course: {removed_course['title']}")
            period["courses"].remove(cidx)
            period["ects_used"] -= removed_course["ects"]
            period["taken_slots"] -= removed_course["parsed_slots"]
            buffer_df = pd.concat(
                [buffer_df, removed_course.to_frame().T], ignore_index=True
            )

        # --- Step C: Free ECTS if needed ---
        while period["ects_used"] + ects > period["max_ects"]:
            if not period["courses"]:
                print(f"⚠️ No courses left to remove in {pid}, cannot free enough ECTS.")
                break
            # Remove lowest score item (heuristic best)
            dfp = period["courses_df"].sort_values("score")
            remove_idx = dfp.iloc[0].name
            removed_course = candidates.loc[remove_idx]

            print(
                f"🔻 Removing course to free ECTS: "
                f"{removed_course['title']} ({removed_course['ects']} ECTS)"
            )

            period["courses"].remove(remove_idx)
            period["ects_used"] -= removed_course["ects"]
            period["taken_slots"] -= removed_course["parsed_slots"]

            buffer_df = pd.concat(
                [buffer_df, removed_course.to_frame().T], ignore_index=True
            )

        # Check again after ECTS cleanup
        if period["ects_used"] + ects > period["max_ects"]:
            print(f"❌ Could not free enough ECTS in {pid}. Trying next period.")
            continue

        # ------------------------------------------------------
        # 3. SUCCESS: Insert the new course
        # ------------------------------------------------------
        print(f"✅ Successfully inserted '{row['title']}' into {pid}")

        period["courses"].append(idx)
        period["ects_used"] += ects
        period["taken_slots"] |= slots
        plan["total_ects"] += ects

        # Rebuild DF
        period["courses_df"] = candidates.loc[period["courses"]].copy()

        # Remove from buffer if it was there
        plan["buffer_courses"] = buffer_df[
            buffer_df["course_code"] != row["course_code"]
        ].reset_index(drop=True)

        return plan

    print(f"❌ Could not insert course '{row['title']}' into any valid period.")
    return plan


if __name__ == "__main__":
    courses_df = load_data()
    plan = plan_msc_program(
        df=courses_df,
        query="I do not like robotics",
        ects_target=85.0,
        top_k=300,
        alpha=0.2,
    )

    print_res = print_plan(plan)
    print(print_res)