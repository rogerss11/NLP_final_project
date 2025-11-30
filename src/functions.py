
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import re
from typing import List, Tuple, Set
import json

from src.dataloader import load_data



def search_relevant_courses(df: pd.DataFrame, query: str, top_k: int,
                            mode: str = "hybrid", alpha: float = 0.2):
    """
    Compute dense, sparse, and hybrid similarities between `query` and all rows in df,
    then return a NEW DataFrame with the top_k rows sorted by score.

    Returns:
        A DataFrame containing all original columns PLUS:
            - "score"
            - "Point( ECTS )"
            - "Schedule"
    """

    # ---- MODE HANDLING ----
    if mode == "dense":
        alpha = 1.0
    elif mode == "sparse":
        alpha = 0.0

    # -------------------------------------------------------------
    # 1) Dense similarities
    # -------------------------------------------------------------
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    q_emb = sentence_model.encode(query.lower())
    q_emb = torch.tensor(q_emb, dtype=torch.float)

    emb_matrix = df["embedded_course"].apply(
        lambda x: torch.tensor(x, dtype=torch.float)
    )

    dense_scores = emb_matrix.apply(
        lambda emb: torch.nn.functional.cosine_similarity(q_emb, emb, dim=0).item()
    )

    # -------------------------------------------------------------
    # 2) Sparse similarities (TF-IDF)
    # -------------------------------------------------------------
    def row_to_text(row):
        data = row.to_dict()
        data.pop("embedded_course", None)  # remove embedding
        return json.dumps(data)

    texts = df.apply(row_to_text, axis=1)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_tfidf = vectorizer.transform([query.lower()])

    sparse_scores = (tfidf_matrix @ query_tfidf.T).toarray().flatten()

    # -------------------------------------------------------------
    # 3) HYBRID score
    # -------------------------------------------------------------
    hybrid_scores = alpha * dense_scores.values + (1 - alpha) * sparse_scores

    # -------------------------------------------------------------
    # 4) Build resulting DataFrame (top-k sorted)
    # -------------------------------------------------------------
    result_df = df.copy()
    result_df["score"] = hybrid_scores

    # Include ECTS and Schedule automatically because we keep all columns
    result_df = result_df.sort_values(by="score", ascending=False).head(top_k)

    return result_df.reset_index(drop=True)

def keep_columns(column_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the specified columns in the DataFrame if they exist.
    """
    existing_columns = [col for col in column_names if col in df.columns]
    return df[existing_columns]

def filter_courses_by_keyword(df: pd.DataFrame, keyword: str, columns: list[str]) -> pd.DataFrame:
    """
    Filter courses that contain the specified keyword in any of the given columns.
    """
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
    return df[mask]


def get_course_ects(row: pd.Series) -> float:
    if "Point( ECTS )" not in row.index:
        return 0.0
    return float(str(row["Point( ECTS )"]).replace(",", "."))

def add_ECTS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum up ECTS for all the courses in the DataFrame.
    return total ECTS value.
    """
    if "Point( ECTS )" in df.columns:
        df["Point( ECTS )"] = df["Point( ECTS )"].astype(str).str.replace(",", ".", regex=False).astype(float)
        total_ects = df["Point( ECTS )"].sum()
        print(f"Total ECTS for filtered courses: {total_ects}")
    else:
        print("Column 'Point( ECTS )' not found in DataFrame.")
        total_ects = 0.0
    return total_ects

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1500)
    pd.set_option('display.max_colwidth', 30)
    
    df = load_data()
    
    schedule_columns = [
        "course_code",
        "title",
        "Location",
        "academic_year",
        "Point( ECTS )",
        "Schedule",
        "Date of examination",
        "Course type"]
        
    print(f"Total courses: {len(df)}")
    top_results = search_relevant_courses(
        df=df,
        query="machine learning",
        top_k=10,
        mode="hybrid",
        alpha=0.2
    )

    print(top_results[["course_code", "title", "Point( ECTS )", "Schedule", "score"]])
    
    keyword = "Lyngby"
    filtered_df = filter_courses_by_keyword(df, keyword, ["Location"])
    print(f"Courses containing '{keyword}': {len(filtered_df)}")
    
    top_results = search_relevant_courses(
    df=filtered_df,
    query="machine learning",
    top_k=30,
    mode="hybrid",
    alpha=0.2
)

    print(top_results[["course_code", "title", "Point( ECTS )", "Schedule", "score"]])
    total_ects = add_ECTS(top_results)