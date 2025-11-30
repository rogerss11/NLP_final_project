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

Example of course in the DDBB: 

{"course_code": "01002", 
"url": "https://kurser.dtu.dk/course/01002", 
"title": "01002 Mathematics 1b (Polytechnical foundation)", 
"academic_year": "2025/2026", 
"fields": {
    "Danish title": "Matematik 1b (Polyteknisk grundlag)", 
    "Language of instruction": "Danish", 
    "Point( ECTS )": 10,
    "Course type": "BSc", 
    "Schedule": ["Spring", "Course is given in two different schedules dependent on study programme.", "Scheme A: F1A, F2 Scheme B: F3A, F4"], 
    "Location": "Campus Lyngby", 
    "Scope and form": "Per week: 2 lectures, 5.5h tutorials and 2h group work. Moreover thematic exercises and project work in some weeks.", 
    "Duration of Course": "13 weeks", 
    "Date of examination": "The written exam will be held on a special day: Click \"Date of examination\" to the left to see DTU's examination timetable. The oral exam is during the 13-week period.", 
    "Type of assessment": ["Written examination and exercises", "Written examination and project report 2 parts: 1) Project report (weight 1/5), 2) Written examination (weight 4/5). Precise description: https://01002.compute.dtu.dk"], 
    "Exam duration": ["Written exam: 4 hours", "2+2 hours"],
    "Aid": "All Aid - no access to the internet : 2-hours test with written works (no electronic aid) and a 2-hours test with all aid (no access to the internet).", "Evaluation": "7 step scale , external examiner", 
    "Not applicable together with": "01004/01005/01015/01006", 
    "Academic prerequisites": "01001/01003 , Latest at the same semester", 
    "Responsible": "Ulrik Engelund Pedersen , Lyngby Campus, Building 303B, Ph. (+45) 4525 5203 , uepe@dtu.dk", 
    "Course co-responsible": "Jakob Lemvig , Lyngby Campus, Building 303B, Ph. (+45) 4525 3051 , jakle@dtu.dk", 
    "Department": "01 Department of Applied Mathematics and Computer Science", 
    "Registration Sign up": "At the Studyplanner", 
    "Green challenge participation": "Please contact the teacher for information on whether this course gives the student the opportunity to prepare a project that may participate in DTU´s Study Conference on sustainability, climate technology, and the environment (GRØN DYST). More infor http://www.groendyst.dtu.dk/english"}, 
    "learning_objectives": ["Perform calculations with vec ... cal problems"]
    }

DF Columns:

(['course_code', 'url', 'title', 'academic_year', 'learning_objectives',
  'Danish title', 'Language of instruction', 'Point( ECTS )',
  'Course type', 'Schedule', 'Location', 'Scope and form',
  'Duration of Course', 'Date of examination', 'Type of assessment',
  'Exam duration', 'Aid', 'Evaluation', 'Not applicable together with',
  'Academic prerequisites', 'Responsible', 'Course co-responsible',
  'Department', 'Registration Sign up', 'Green challenge participation',
  'Home page', 'Previous Course', 'Participants restrictions',
  'Mandatory Prerequisites', 'Department involved',
  'External Institution', 'Offered as'],
dtype='object')



Strategy:
- Filter out non-MSc courses --> Can not do this: miss out on many relevant courses (i.e. Intro to ML)
1. Filter out courses not in Location
2. Semesters: 
    Option 1: Autumn (max 30 ECTS) -> January (5 ECTS) -> Spring (max 30 ECTS) -> June (5 ECTS) -> July (5 ECTS) -> August (5 ECTS) -> Autumn (max 30 ECTS) -> January (5 ECTS)
    
    Option 2: Spring (max 30 ECTS) -> June (5 ECTS) -> July (5 ECTS) -> August (5 ECTS) -> Autumn (max 30 ECTS) -> January (5 ECTS) -> Spring (max 30 ECTS)