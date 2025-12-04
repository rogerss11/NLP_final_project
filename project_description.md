# AI Course Planner Companion

**Retrieval-augmented generation from the DTU course database**

The idea of this project is to expand on the RAG topic covered in the last lecture. As briefly discussed in the weekly meeting, the goal is to integrate an LLM that can search the DTU database in order to tailor a study plan based on a specific query.

The program should take into account the master’s program that the student is going to take, the total ECTS being equal to 120, and that each semester no two courses can be in the same time slot. The selected courses must be of high relevance to the studied program and the interests of the student.

* *Title of the project:*
  AI Course Planner Companion

* *Description of tools that can be used:*
  DSPY or OpenAI libraries for the LLM. Dense + Sparse search algorithms. Prompt engineering. Potentially some algorithm to handle the scheduling (to be determined).

* *Description of data that can be used:*
  DTU course base provided in the last exercise.

* *Examples of desired input and output:*

**Input:** A query specifying what the student studied in the bachelor’s program and what their areas of interest are, the specific MSc program they are about to study, and the starting term (Spring/Autumn).

**Output:** A list of courses that satisfy the program requirements.

---

**Notes:**

For now, we will limit the scope of the project to the following assumptions and requirements:

* Single query/prompt.
* The courses should add up exactly to 90 ECTS (+30 ECTS for the thesis).
* The courses should be of high relevance to the chosen MSc program and the student’s interests.
* There cannot be more than two courses in the same time slot.
* No course should be repeated.
* The compulsory courses need to be included (e.g., *Innovation in Engineering* and *Quantitative Sustainability*).