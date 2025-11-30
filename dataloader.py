import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import jsonlines
import os
import sys

# ====== Configuration ======
embedding_columns = [
    "course_code", 
    "title",
    "learning_objectives", 
    "Academic prerequisites",
    "Responsible",
    "Course co-responsible",
    "Department",
]

courses_path = "dtu_courses.jsonl"
embed_path = "dtu_courses_embed.jsonl"

# ============== FUNCTIONS =============

# ------------- Data Loading and Embedding -------------
# If embeddings jsonl is not found: load jsonl file, compute embeddings, 
#       save to jsonl and return df
# If embeddings jsonl is found: load and return df directly.
# df: : one column per field plus "embedded_course".

def load_courses_jsonl_to_df(path: str) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            # If "fields" exists (original DTU file)
            if "fields" in entry:
                base = {
                    "course_code": entry.get("course_code"),
                    "url": entry.get("url"),
                    "title": entry.get("title"),
                    "academic_year": entry.get("academic_year"),
                    "learning_objectives": entry.get("learning_objectives"),
                }
                fields = entry.get("fields", {})
                flat_entry = {**base, **fields}
            else:
                # Already flattened JSONL (embedded file)
                flat_entry = entry

            rows.append(flat_entry)

    return pd.DataFrame(rows)

def save_df_as_jsonl(df, path: str):
    """
    Save a pandas DataFrame to a .jsonl file.
    Each row in the DataFrame becomes one JSON object per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.dropna().to_dict()  # optionally drop NaN fields
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved {len(df)} rows to {path}")

def embed_courses(df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame:
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def row_to_text(row):
        parts = []
        for col in embedding_columns:
            val = row[col]
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            parts.append(f"{col}: {val}")
        return " | ".join(parts)

    texts = df.apply(row_to_text, axis=1).tolist()

    # Encode one by one so we can track progress
    embeddings = []
    total = len(texts)
    bar_length = 40

    for i, text in enumerate(texts, start=1):
        emb = sentence_model.encode([text], convert_to_numpy=True)[0].tolist()
        embeddings.append(emb)

        # progress update
        progress = i / total
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        sys.stdout.write(f"\rEmbedding [{bar}] {progress*100:5.1f}% ({i}/{total})")
        sys.stdout.flush()

    print()  # newline after bar

    df["embedded_course"] = embeddings
    return df
 
def load_data(courses_path, embed_path: str) -> pd.DataFrame:
    """
    Load the DTU courses and embeddings into a DataFrame.
    If embeddings jsonl is not found: load jsonl file, compute embeddings,
        save to jsonl and return df.
    If embeddings jsonl is found: load and return df directly.
    """
    if os.path.exists(embed_path):
        print(f"Found embeddings file '{embed_path}', loading DataFrame...")
        df = load_courses_jsonl_to_df(embed_path)
        print("Loaded DataFrame from disk.")
    else:
        print("No embeddings file found — loading courses and computing embeddings...")
        df = load_courses_jsonl_to_df(courses_path)
        df = embed_courses(df, embedding_columns)
        save_df_as_jsonl(df, embed_path)
        print(f"Computed embeddings and saved DataFrame to '{embed_path}'.")
    return df
# -----------------------------------------------------

if __name__ == "__main__":
    df = load_data(courses_path, embed_path)

    print(df.columns)
    