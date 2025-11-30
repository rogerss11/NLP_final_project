from fastapi import FastAPI, Query
from pydantic import BaseModel
import jsonlines
import json
import sys
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

from dataloader import load_data
# ---------------------- INITIALIZATION ----------------------
# Initialize FastAPI app
app = FastAPI()

# Load transformer model (keep separate from any LLM model variable)
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load LLM (don't overwrite `sentence_model`)
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
if not CAMPUSAI_API_KEY:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://campusai.compute.dtu.dk/api"
)

# models = client.models.list()

# for m in models.data:
#     print(m.id)

# ---------------------- DATA LOADING AND EMBEDDING ----------------------
df = load_data()

# Helper function to search courses
def search_relevant_courses(query: str, top_k: int, mode: str, alpha=0.2):
    if mode == "dense":
        alpha = 1.0
    elif mode == "sparse":
        alpha = 0.0
    
    # Compute dense similarities
    # encode query with the sentence transformer and convert to torch tensor
    q_emb = sentence_model.encode(query.lower())
    query_embedding = torch.tensor(q_emb, dtype=torch.float)
    similarities_dense = {}

    for course in courses:
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding,
            courses[course]["embedded_course"],
            dim=0
        ).item()
        similarities_dense[course] = similarity

    sorted_similarities = sorted(similarities_dense.items(), key=lambda x: x[1], reverse=True)
    results = [
        {"course_code": c, "title": courses[c]["title"], "score": s}
        for c, s in sorted_similarities[:top_k]
    ]
    # Compute sparse similarities
    vectorizer = TfidfVectorizer()
    # Exclude non-serializable fields (like tensors) when creating texts for TF-IDF.
    course_texts = []
    for course in courses:
        # make a shallow copy without the embedding tensor
        course_data = {k: v for k, v in courses[course].items() if k != "embedded_course"}
        course_texts.append(json.dumps(course_data))
    tfidf_matrix = vectorizer.fit_transform(course_texts)
    query_tfidf = vectorizer.transform([query.lower()])
    similarities_sparse = (tfidf_matrix @ query_tfidf.T).toarray().flatten()
    course_codes = list(courses.keys())
    sparse_dict = {course_codes[i]: similarities_sparse[i] for i in range(len(course_codes))}
    # Combine similarities
    combined_similarities = {}
    for course in courses:
        combined_similarities[course] = (
            alpha * similarities_dense[course] + (1 - alpha) * sparse_dict[course]
        )
    sorted_combined = sorted(combined_similarities.items(), key=lambda x: x[1], reverse=True)
    # Return the full course information for the top_k courses
    results = [
        {**courses[c], "score": s}
        for c, s in sorted_combined[:top_k]
    ]
    return results

# ---------------------- ROUTES ----------------------

@app.get("/v1/search")
def search_courses(
    query: str = Query(...),
    top_k: int = Query(10),
    mode: str = Query("hybrid", regex="^(dense|sparse|hybrid)$"),
):
    """
    Search for courses based on a free text query.
    Compare query embedding with titles and descriptions.
    Return top_k most similar courses.
    """
    top_courses_ddbb = course_search(query, top_k, mode)
    # Return only the course code title and score 
    results = [
        {"course_code": course["course_code"], "title": course["title"], "score": course["score"]}
        for course in top_courses_ddbb
    ]
    return {"query": query, "mode": mode, "results": results}

@app.get("/v1/ask")
def search_courses(
    query: str = Query(...),
    top_k: int = Query(10),
    mode: str = Query("hybrid", regex="^(dense|sparse|hybrid)$"),
):
    
    history = []
    courses_found = course_search(query, top_k, mode)
    # Remove the embedding tensors from the context
    for course in courses_found:
        course.pop("embedded_course", None)

    PROMPT = f"""You are a helpful assistant for DTU courses.
                Use the context below to answer the user's question.
                If the answer is not present, say you don't know.
                Try to keep the answer concise and to the point.

                Question: {query}

                Context:
                {json.dumps(courses_found, indent=2)}

                Answer:"""

    history.append({"role": "user", "content": PROMPT})
    response = client.chat.completions.create(
        model="Gemma3",
        messages=history,
        stream=False  # explicitly disable streaming
        )

    results = response.choices[0].message.content
    
    courses_found_res = [
        {"course_code": course["course_code"], "title": course["title"], "score": course["score"]}
        for course in courses_found
    ]

    return {
        "query": query,
        "answer": results,
        "retrieved_courses": courses_found_res,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)