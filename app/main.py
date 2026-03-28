"""
School ERP Chatbot — Query Planner Service
==========================================
Receives a natural language query from the Frappe Server Script.
Returns a structured query plan (JSON) that Frappe executes against its DB.

No Frappe DB access here — this is pure planning / LLM logic.
"""

import os
import json
import re
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="School ERP Planner", version="1.0.0")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_SECRET    = os.getenv("API_SECRET", "change-me-in-production")
CUR_YEAR      = os.getenv("ACADEMIC_YEAR", "2025-26")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────

class PlanRequest(BaseModel):
    message: str
    session_id: Optional[str] = ""
    history: Optional[list] = []

class PlanResponse(BaseModel):
    type: str          # "quick" | "orm_plan" | "fallback"
    pattern: Optional[str] = None       # for type="quick"
    plan: Optional[dict] = None         # for type="orm_plan"
    message: Optional[str] = None       # for type="fallback"


# ── HEALTH CHECK ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "school-erp-planner"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── ENTITY DETECTION ─────────────────────────────────────────

CLASS_KEYWORDS = {
    "standard", "grade", "class", "lkg", "ukg", "pre kg", "prekg",
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th",
    "10th", "11th", "12th", "students", "student", "staff", "teacher",
    "teachers", "all", "today", "this", "last", "month", "week",
    "year", "the", "a", "an", "me", "my", "their", "our"
}

STOP_WORDS = {
    "pending", "fees", "fee", "absent", "attendance", "contact",
    "details", "email", "phone", "mobile", "class", "standard",
    "grade", "program", "enrolled", "guardian", "timetable",
    "schedule", "teacher", "subject", "period"
}

WHO_TRIGGERS = [
    "who is", "show me everything about", "tell me about",
    "everything about", "details about", "info about", "what about"
]

def has_entity_name(q: str) -> bool:
    ql = q.lower().strip()

    # Possessive
    if "'s " in ql or "\u2019s " in ql:
        return True

    # Who is X / tell me about X patterns
    for trigger in WHO_TRIGGERS:
        if trigger in ql:
            rest = ql.split(trigger)[-1].strip().split()
            if rest:
                candidate = rest[0].rstrip("?.,")
                if (len(candidate) > 2
                        and candidate not in CLASS_KEYWORDS
                        and candidate not in STOP_WORDS):
                    return True

    # "of X" or "for X" where X is not a class word
    words = ql.split()
    for i in range(len(words) - 1):
        if words[i] in ("of", "for", "about"):
            next_w = words[i + 1].lower().rstrip("?.,")
            is_class = any(next_w == kw or next_w.startswith(kw) for kw in CLASS_KEYWORDS)
            is_stop  = next_w in STOP_WORDS
            if not is_class and not is_stop and len(next_w) > 2:
                rest_str = " ".join(words[i+1:])
                has_class_after = any(kw in rest_str for kw in {
                    "standard", "grade", "class", "lkg", "ukg",
                    "1st", "2nd", "3rd", "4th", "5th", "6th",
                    "7th", "8th", "9th", "10th", "11th", "12th"
                })
                if not has_class_after:
                    return True

    return False


# ── QUICK MATCH ───────────────────────────────────────────────

CONTACT_WORDS  = {"contact", "email", "phone", "mobile", "number", "details", "address"}
MULTI_INTENT   = {
    "and their", "along with", "with their", "with guardian",
    "with fees", "with contact", "with attendance", "with class",
    "and fees", "and contact", "and attendance", "and guardian",
    "and teacher", "and timetable", "fee status", "guardian contact",
    "and their class", "and their fee", "also show", "also get",
    "together", "combined"
}
COMPLEX_WORDS  = {
    "guardian", "fees", "fee", "contact", "absent", "attendance",
    "with their", "with guardian", "along with", "and their",
    "and contact", "and fees", "pending", "unpaid"
}
CLASS_KW_LIST  = {
    "standard", "lkg", "ukg", "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th", "11th", "12th"
}
GENDER_CLASS_KW = CLASS_KW_LIST | {
    "class 1","class 2","class 3","class 4","class 5","class 6",
    "class 7","class 8","class 9","class 10","grade 1","grade 2",
    "grade 3","grade 4","grade 5","grade 6","pre kg"
}

def quick_match(q: str) -> str:
    # Entity bypass
    if has_entity_name(q):
        return ""

    # Multi-intent bypass
    if any(w in q for w in MULTI_INTENT):
        return ""

    # Long query bypass
    if len(q.split()) > 8:
        if not any(q.strip().startswith(p) for p in [
            "how many students", "how many teachers", "absent today", "present today"
        ]):
            return ""

    # Student attendance patterns
    if "absent today" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS):
            return "absent_today"
    if "absent last month" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS | {"fee", "fees", "status", "guardian"}):
            return "absent_last_month"
    if "absent this month" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS):
            return "absent_this_month"
    if "absent last week" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS):
            return "absent_last_week"
    if "absent this week" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS):
            return "absent_this_week"
    if "present today" in q and not any(w in q for w in ["staff", "employee"]):
        if not any(w in q for w in CONTACT_WORDS):
            return "present_today"

    # Pending fees
    if any(w in q for w in ["pending fees", "unpaid fees", "outstanding fees", "fee defaulters"]):
        if not any(w in q for w in CONTACT_WORDS):
            if not any(w in q for w in CLASS_KW_LIST):
                return "pending_fees"

    # Simple counts
    if q.strip() in {"how many students", "total students", "count students",
                     "how many students are there", "total number of students"}:
        return "count_students"
    if q.strip() in {"how many teachers", "how many instructors", "how many staff",
                     "total teachers", "total staff"}:
        return "count_instructors"

    # Listings
    if any(w in q for w in ["list all courses", "show all courses", "all courses", "list courses"]):
        return "list_courses"
    if any(w in q for w in ["list all programs", "all programs", "list all standards", "all standards"]):
        return "list_programs"
    if any(w in q for w in ["list all batches", "all batches", "list batches"]):
        return "list_batches"

    # Count/list by program
    count_triggers = [
        "how many students in", "how many students are in",
        "number of students in", "count students in", "total students in"
    ]
    if any(t in q for t in count_triggers):
        if any(w in q for w in CLASS_KW_LIST):
            if not any(w in q for w in COMPLEX_WORDS):
                return "count_by_program"

    list_triggers = ["list students in", "show students in", "list all students in"]
    if any(t in q for t in list_triggers):
        if any(w in q for w in CLASS_KW_LIST):
            if not any(w in q for w in COMPLEX_WORDS):
                return "list_by_program"

    # Gender count
    gender_triggers = [
        "how many boys", "how many girls", "how many male", "how many female",
        "number of boys", "number of girls", "boy students", "girl students",
        "male students", "female students"
    ]
    if any(t in q for t in gender_triggers):
        if any(w in q for w in GENDER_CLASS_KW):
            return "gender_count_by_program"
        return "gender_count_total"

    # Admissions
    if any(t in q for t in [
        "joined this year", "admitted this year", "enrolled this year",
        "new students this year", "joined this month", "admitted this month", "new admissions"
    ]):
        return "students_joined"

    # Staff absent
    if any(w in q for w in ["absent", "absence"]):
        if any(w in q for w in ["staff", "teacher", "teachers", "instructor", "employee", "faculty"]):
            skip = any(w in q for w in [
                "handle", "handles", "which teacher", "whose class",
                "students are", "mostly", "class where", "classes where",
                "class with", "where students"
            ])
            if not skip:
                if "today" in q:       return "staff_absent_today"
                if "this week" in q:   return "staff_absent_week"
                if "last week" in q:   return "staff_absent_last_week"
                if "this month" in q:  return "staff_absent_month"
                if "last month" in q:  return "staff_absent_last_month"
                return "staff_absent_month"

    # Teacher classes
    if any(t in q for t in [
        "which class does", "what class does", "which standard does",
        "which class is", "what does", "classes taught by",
        "teaches which class", "teaches which standard"
    ]):
        if any(w in q for w in ["teach", "teaches", "teaching", "instructor", "teacher"]):
            return "teacher_classes"

    return ""


# ── LLM CALL ─────────────────────────────────────────────────

async def call_llm(messages: list, max_tokens: int = 800) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0
            }
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def clean_json(raw: str) -> str:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start:end+1]
    return "{}"


# ── LLM PLAN ─────────────────────────────────────────────────

def build_prompt(query: str, history: list) -> list:
    # FIX: system prompt sent as proper "system" role, not prepended to user message.
    # This gives the model clearer instruction boundaries and better rule-following.
    system = f"""You are a query planner for a school ERP (Frappe).
Convert the user's question into a JSON query plan.
Current academic year: {CUR_YEAR}

DOCTYPES AND FIELDS:
- Student: name, student_name, first_name, last_name, student_email_id, student_mobile_number, date_of_birth, gender, blood_group, city, state, address_line_1, pincode, joining_date, enabled, custom_father_name, custom_mother_name, custom_guardian_contact_number, custom_guardian_email_id, custom_guardian_occupation, custom_admission_no, custom_student_category, custom_sats_id
- Instructor: name, instructor_name, custom_teacher_email, department, status
- Employee: name, employee_name, department, designation, status
- Guardian: name, guardian_name, mobile_number, email_address, occupation
- Program Enrollment: name, student, student_name, program, academic_year, student_batch_name, enrollment_date
- Student Group: name, student_group_name, program, academic_year
- Student Group Instructor: name, parent, instructor, instructor_name
- Student Attendance: name, student, student_name, student_group, date, status, leave_type
- Attendance: name, employee, employee_name, department, attendance_date, status, leave_type
- Fees: name, student, student_name, program, academic_year, due_date, outstanding_amount, grand_total, status
- Fee Structure: name, program, academic_year, academic_term, total_amount
- Fee Component: name, parent, fees_category, amount, description
- Fee Schedule: name, fee_schedule_name, program, academic_year, based_on, fee_schedule_date
- Course Enrollment: name, student, student_name, course, program, enrollment_date, program_enrollment
- School Timetable: name, title, class, academic_year, school_level
- Timetable Period: name, parent, period_no, weekday, subject, teacher, from_time, to_time

IMPORTANT FIELD NOTES:
- Student.custom_father_name and custom_mother_name hold parent names
- Student.custom_guardian_contact_number holds guardian phone
- Student.custom_admission_no holds the school admission number
- Student Attendance.student_group is the full group name string e.g. "6th Standard Section - {CUR_YEAR}" — do NOT filter with just the program name like "6th Standard"
- Student Attendance has no .program field — always join via Student Group or use student_name LIKE filter for per-student queries

HARD RULES:
1. Never use subqueries or nested filters. Always use pipe between steps.
2. Pipe syntax: pipe_from=save_as_of_prev_step, pipe_field=field_in_prev, pipe_into=field_in_current
3. Name searches use ~ prefix: {{"student_name": "~Navaneeth"}} means LIKE search
4. School Timetable name = program + " Time Table" (e.g. "6th Standard Time Table")
5. Timetable Period always piped from School Timetable: pipe_field=name, pipe_into=parent
6. Fee Component always piped from Fee Structure: pipe_field=name, pipe_into=parent
7. When a period number is mentioned add period_no filter to Timetable Period
8. For class teacher: Student Group (filter by program) → Student Group Instructor (pipe name→parent)
9. NEVER use Timetable for class teacher queries
10. For pending fees of a specific student: Fees with student_name LIKE filter
11. For pending fees by class: Program Enrollment (save student IDs) → pipe student→student into Fees
12. FORBIDDEN fields: Student Attendance.program, Student.student_group, Program Enrollment.student_group, Fees.program (unreliable)
13. No OR filters. No $or. Filter one field at a time.
14. For post_process operations use: set_subtract, group_count, cross_filter, missing_field
15. Always try to answer. Only set fallback=true if completely unrelated to school data.
16. op filter supports: ">", ">=", "<", "<=", "=" — e.g. {{"outstanding_amount": {{"op": ">=", "val": 500}}}}
17. For attendance history of a specific student: filter Student Attendance with student_name LIKE, add date range filters for the period (date >= start, date <= end)

PLAN FORMAT:
{{
  "intent": "short description",
  "steps": [
    {{
      "id": "s1",
      "doctype": "Student",
      "filters": {{"student_name": "~Navaneeth"}},
      "fields": ["student_name", "student_email_id", "student_mobile_number"],
      "limit": 5,
      "order_by": "",
      "save_as": "",
      "pipe_from": "",
      "pipe_field": "",
      "pipe_into": ""
    }}
  ],
  "post_process": [],
  "fallback": false,
  "fallback_msg": ""
}}

EXAMPLES:

Q: fee structure for 6th standard
{{"intent":"fee structure breakdown","steps":[{{"id":"s1","doctype":"Fee Structure","filters":{{"program":"6th Standard","academic_year":"{CUR_YEAR}"}},"fields":["name","program","total_amount"],"limit":1,"order_by":"name desc","save_as":"fs","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Fee Component","filters":{{}},"fields":["fees_category","amount","parent"],"limit":20,"order_by":"","save_as":"","pipe_from":"fs","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: timetable for 6th standard on Thursday
{{"intent":"class timetable by day","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"6th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Thursday"}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":20,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: who teaches 6th standard on thursday 2nd period
{{"intent":"timetable period teacher","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"6th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Thursday","period_no":2}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":5,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: what is the pending fee of Sagar
{{"intent":"student fee lookup","steps":[{{"id":"s1","doctype":"Fees","filters":{{"student_name":"~Sagar","outstanding_amount":{{"op":">","val":0}}}},"fields":["student_name","outstanding_amount","due_date","program"],"limit":5,"order_by":"outstanding_amount desc","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: students in 6th standard with pending fees
{{"intent":"class pending fees","steps":[{{"id":"s1","doctype":"Program Enrollment","filters":{{"program":"6th Standard"}},"fields":["student","student_name","program"],"limit":200,"order_by":"","save_as":"enrolled","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Fees","filters":{{"outstanding_amount":{{"op":">","val":0}}}},"fields":["student","student_name","outstanding_amount","due_date"],"limit":200,"order_by":"outstanding_amount desc","save_as":"","pipe_from":"enrolled","pipe_field":"student","pipe_into":"student"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: who is navaneeth
{{"intent":"student profile","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Navaneeth"}},"fields":["student_name","student_email_id","student_mobile_number","date_of_birth","gender","joining_date","custom_admission_no","custom_father_name","custom_mother_name","city","blood_group"],"limit":3,"order_by":"","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: who is the class teacher of grade 3
{{"intent":"class teacher lookup","steps":[{{"id":"s1","doctype":"Student Group","filters":{{"program":"3rd Standard"}},"fields":["name","student_group_name","program"],"limit":1,"order_by":"","save_as":"grp","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Student Group Instructor","filters":{{}},"fields":["instructor","instructor_name","parent"],"limit":5,"order_by":"","save_as":"","pipe_from":"grp","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: classes without a teacher assigned
{{"intent":"classes without teachers","steps":[{{"id":"s1","doctype":"Student Group","filters":{{}},"fields":["name","student_group_name","program"],"limit":100,"order_by":"","save_as":"all_groups","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Student Group Instructor","filters":{{}},"fields":["parent","instructor_name"],"limit":200,"order_by":"","save_as":"assigned","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"set_subtract","from":"all_groups","exclude":"assigned","match_field":"name","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

Q: which class has the most absences
{{"intent":"class absence ranking","steps":[{{"id":"s1","doctype":"Student Attendance","filters":{{"status":"Absent"}},"fields":["student_group","student_name","date"],"limit":500,"order_by":"","save_as":"absences","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"group_count","from":"absences","group_field":"student_group","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

Q: students without email
{{"intent":"students missing contact","steps":[{{"id":"s1","doctype":"Student","filters":{{"enabled":1}},"fields":["name","student_name","student_email_id","student_mobile_number"],"limit":500,"order_by":"","save_as":"students","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"missing_field","from":"students","field":"student_email_id","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

Q: students without phone number
{{"intent":"students missing mobile","steps":[{{"id":"s1","doctype":"Student","filters":{{"enabled":1}},"fields":["name","student_name","student_email_id","student_mobile_number"],"limit":500,"order_by":"","save_as":"students","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"missing_field","from":"students","field":"student_mobile_number","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

Q: how many times was Navaneeth absent this month
{{"intent":"student attendance history","steps":[{{"id":"s1","doctype":"Student Attendance","filters":{{"student_name":"~Navaneeth","status":"Absent","date":{{"op":">=","val":"MONTH_START"}}}},"fields":["student_name","student_group","date","status"],"limit":100,"order_by":"date desc","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Q: what is navaneeth's father's name
{{"intent":"student parent info","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Navaneeth"}},"fields":["student_name","custom_father_name","custom_mother_name","custom_guardian_contact_number"],"limit":3,"order_by":"","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Return ONLY raw JSON. No markdown, no explanation.

Q: """

    # Build messages array with proper system role separation
    messages = [{"role": "system", "content": system}]

    # Add conversation history (last 6 exchanges)
    for h in history[-6:]:
        messages.append(h)

    # Add current query as user message only (not mixed with system prompt)
    messages.append({"role": "user", "content": query})

    return messages


async def llm_plan(query: str, history: list) -> dict:
    default = {
        "intent": "",
        "steps": [],
        "post_process": [],
        "fallback": True,
        "fallback_msg": "I'm not sure how to answer that. Try asking about students, attendance, fees, timetables, or teachers."
    }

    try:
        messages = build_prompt(query, history)
        raw      = await call_llm(messages, max_tokens=800)
        parsed   = json.loads(clean_json(raw))

        if parsed.get("fallback"):
            return {**default, "fallback_msg": parsed.get("fallback_msg") or default["fallback_msg"]}

        steps = parsed.get("steps") or []
        if not steps:
            return default

        return {
            "intent":       parsed.get("intent") or "",
            "steps":        steps,
            "post_process": parsed.get("post_process") or [],
            "fallback":     False,
            "fallback_msg": ""
        }

    except Exception as e:
        default["fallback_msg"] = f"Planner error: {str(e)[:100]}"
        return default


# ── MAIN ENDPOINT ─────────────────────────────────────────────

@app.post("/plan", response_model=PlanResponse)
async def plan_query(
    req: PlanRequest,
    x_api_key: Optional[str] = Header(None)
):
    # Auth check
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    message = req.message.strip()
    if not message:
        return PlanResponse(type="fallback", message="Empty message.")

    q = message.lower()

    # Quick match — no LLM needed
    qm = quick_match(q)
    if qm:
        return PlanResponse(type="quick", pattern=qm)

    # LLM plan
    plan = await llm_plan(message, req.history or [])

    if plan.get("fallback"):
        return PlanResponse(
            type="fallback",
            message=plan.get("fallback_msg") or "Sorry, I couldn't understand that."
        )

    return PlanResponse(type="orm_plan", plan=plan)
