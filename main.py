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

from src.dataloader import load_data
# ---------------------- INITIALIZATION ----------------------
# Initialize FastAPI app
app = FastAPI()

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

# ---------------------- ENDPOINTS ----------------------
@app.get("/v1/ask")
def search_courses(
    query: str = Query(...),
    top_k: int = Query(10),
    mode: str = Query("hybrid", regex="^(dense|sparse|hybrid)$"),
):
    
    history = []
    courses_found = (query, top_k, mode)
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
        model="DeepSeek-R1",
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