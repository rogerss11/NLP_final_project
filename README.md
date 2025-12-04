AI Course Planner Companion
==============================
The idea of this project is to expand on the RAG topic covered in the last lecture. As briefly discussed in the weekly meeting, the goal is to integrate an LLM that can search the DTU database in order to tailor a study plan based on a specific query.

The program should take into account the master’s program that the student is going to take, the total ECTS being equal to 90 (without accounting for the thesis), and that each semester no two courses can be in the same time slot. The selected courses must be of high relevance to the studied program and the interests of the student.

Dataset
-------

- `dtu_courses.jsonl` — Original dtu course DDBB structured JSON lines extracted from the pages

Each JSON line corresponds to one course, for example:

```
{
  "course_code": "02195",
  "title": "Quantum Algorithms and Machine Learning",
  "academic_year": "2025/2026",
  "fields": {
    "Responsible": "Sven Karlsson",
    "Point( ECTS )": 5,
    "Language of instruction": "English",
    "Department": "01 Department of Applied Mathematics and Computer Science"
  },
  "learning_objectives": [
    "Explain common quantum algorithms",
    "Use quantum computing frameworks",
    "Develop quantum algorithms"
  ],
  "content": "Use of key quantum concepts in algorithms: Quantum states; Qubits; Entanglement ..."
}
```
- `dtu_courses_embed.jsonl` — structured JSON lines where all the subcategories inside of fields have been extracted and converted to main categories. A new key is given for the embeddings of the courses. This file is loaded and converted to a pandas dataframe for simpler filtering and processing.

The following categories are used for the embeddings:

  - "course_code",
  - "title",
  - "learning_objectives",
  - "Academic prerequisites",
  - "Responsible",
  - "Course co-responsible",
  - "Department",
  - "Department involved",


Inputs
---------------

A user query consists of:

- **query:** Information about what are the user's interests and experience. (e.g.) I am interested in Robotics and Artificial Intelligence
- **user_request:** Additional instructions given to the LLM about modifications to be applied in the plan or preferences about subjects. (e.g.) I don't want to do August courses
- **mandatory:** Course ID's for the subjects that MUST be included in the plan. (e.g.) 02180
- **forbiden:** Course ID's for the subjects that should be excluded from the plan. (e.g.) 02451, 42500


Implementation approach
--------------------------

The process of building the personalised course plan is the following:

1. The input `query` is used to obtain a list of the 200 most relevant courses based on a hybrid search approach combining dense and sparse retrieval.
2. An algorithm builds an initial plan by taking into account the `mandatory`and `forbidden` courses. The algorithm follows a greedy-optimization approach, in which time period slots are filled based on the preferences obtained previously in step 1. A buffer of 15 relevant courses is included that the LLM can choose to replace from.
3. The `query` and `user_request` are passed into the LLM together with the full plan and an initial prompt with guidelines.
4. The LLM reasons about the potential changes and finally returns a modified version of the plan to satisfy the additional constraints.



Running the main code API (in Docker)
---
Build the container
```docker
docker build -t course-planner:latest .
```
Run the container
```docker
docker run --rm -p 8000:8000 --env-file ~/.env course-planner:latest
```
You can then open a browser and go to: http://127.0.0.1:8000/docs to test the course planner.


Running outside of docker (Not recommended)
---
Create a virtual environment and install requirements file
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run the scripts (choose one)
```bash
python -m main # API
```
```bash
python -m src.agent # LLM reasoning on top of course plan
```
```bash
python -m src.planner # Only the greedy planner algorithm
```
When testing `src.agent`and `src.planner`, the queries are hardcoded in the `if __name__ == "__main__":` sections. Other prompts can be tested by manually modifying those.

Testing using CLI
-------

Example queries

```
curl -s \
"http://localhost:8000/v1/build_plan\
?query=robotics+ai\
&user_request=remove+june+courses\
&forbidden=42500,01005\
&mandatory=02456" | jq
```

Repository overview
-------
```
NLP_final_project
  |__ main.py                   -> Entrypoint and fast API wrapper
  |__ dtu_courses.jsonl         -> Original DDBB
  |__ dtu_courses_embed.jsonl   -> Adapted DDBB (explained above)
  |__ src
        |__ dataloader.py       -> Load the correspoing DDBB and/or create the embeddings
        |__ functions.py        -> Useful functions to work with the course DDBB in pandas
        |__ planner.py          -> Create the initial course planner using a greedy approach
        |__ agent.py            -> LLM querying and orchestration
```

Available models
----
Despite its longer inference time, DeepSeek-R1 is the implemented model. Several were tested, but this one was the one that does produce less hallucination with the large prompts containing the entire course plan. Unfortunaly, its reasoning time can sometimes be large.

- Devstral
- Qwen3-VL
- Qwen3
- Qwen3-Coder
- Granite4
- gpt-oss
- **DeepSeek-R1**
- Gemma3