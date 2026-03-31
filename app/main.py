"""
School ERP Chatbot — Query Planner Service
==========================================
Receives a natural language query from the Frappe Server Script.
Returns a structured query plan (JSON) that Frappe executes against its DB.

No Frappe DB access here — pure planning / pre-processing / LLM logic.
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_SECRET   = os.getenv("API_SECRET", "change-me-in-production")
CUR_YEAR     = os.getenv("ACADEMIC_YEAR", "2025-26")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────

class PlanRequest(BaseModel):
    message: str
    session_id: Optional[str] = ""
    history: Optional[list] = []

class PlanResponse(BaseModel):
    type: str          # "quick" | "orm_plan" | "fallback"
    pattern: Optional[str] = None
    plan: Optional[dict] = None
    message: Optional[str] = None
    ctx: Optional[dict] = None   # resolved pre-processor context for history inheritance


# ── HEALTH CHECK ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "school-erp-planner"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════
# GROUND TRUTH LOOKUP TABLES
# Built from diagnostic data — these are exact values from your DB.
# The pre-processor resolves user input against these before the
# LLM ever sees the query, eliminating a whole class of guessing.
# ═══════════════════════════════════════════════════════════════

# Exact program names as stored in Frappe
PROGRAMS = [
    "Pre KG", "LKG", "UKG",
    "1st Standard", "2nd Standard", "3rd Standard", "4th Standard",
    "5th Standard", "6th Standard", "7th Standard", "8th Standard",
    "9th Standard", "10th Standard"
]

# Alias map: everything a user might type → exact program name
PROGRAM_ALIASES = {
    # Early years
    "pre kg": "Pre KG", "prekg": "Pre KG", "pre-kg": "Pre KG",
    "nursery": "Pre KG",
    "lkg": "LKG", "lower kg": "LKG", "lower kindergarten": "LKG",
    "ukg": "UKG", "upper kg": "UKG", "upper kindergarten": "UKG",
    # Ordinals
    "1st": "1st Standard", "1": "1st Standard", "first": "1st Standard", "class 1": "1st Standard", "grade 1": "1st Standard",
    "2nd": "2nd Standard", "2": "2nd Standard", "second": "2nd Standard", "class 2": "2nd Standard", "grade 2": "2nd Standard",
    "3rd": "3rd Standard", "3": "3rd Standard", "third": "3rd Standard", "class 3": "3rd Standard", "grade 3": "3rd Standard",
    "4th": "4th Standard", "4": "4th Standard", "fourth": "4th Standard", "class 4": "4th Standard", "grade 4": "4th Standard",
    "5th": "5th Standard", "5": "5th Standard", "fifth": "5th Standard", "class 5": "5th Standard", "grade 5": "5th Standard",
    "6th": "6th Standard", "6": "6th Standard", "sixth": "6th Standard", "class 6": "6th Standard", "grade 6": "6th Standard",
    "7th": "7th Standard", "7": "7th Standard", "seventh": "7th Standard", "class 7": "7th Standard", "grade 7": "7th Standard",
    "8th": "8th Standard", "8": "8th Standard", "eighth": "8th Standard", "class 8": "8th Standard", "grade 8": "8th Standard",
    "9th": "9th Standard", "9": "9th Standard", "ninth": "9th Standard", "class 9": "9th Standard", "grade 9": "9th Standard",
    "10th": "10th Standard", "10": "10th Standard", "tenth": "10th Standard", "class 10": "10th Standard", "grade 10": "10th Standard",
}

# Exact timetable names from DB (some deviate from "X Time Table" pattern)
TIMETABLE_NAMES = {
    "Pre KG":       None,                        # no timetable exists
    "LKG":          None,                        # no timetable exists
    "UKG":          "UKG",                       # name is just "UKG"
    "1st Standard": None,                        # no timetable exists
    "2nd Standard": "2nd Standard Time Table",
    "3rd Standard": "3rd Standard Time Table",
    "4th Standard": "4th Standard Time Table",
    "5th Standard": "5th Standard Time Table",
    "6th Standard": "6th Standard Time Table",
    "7th Standard": "7th Standard Time Table",
    "8th Standard": "8th Standard Time Table",
    "9th Standard": "9th standard Time Table",  # lowercase 's' — exact match required
    "10th Standard": "10th Standard Time Table",
}

# Student group names per program + year (all groups from DB)
# Key: (program, academic_year) → list of group names
STUDENT_GROUPS = {
    ("Pre KG",       "2025-26"): ["Pre KG Section - 2025-26", "Topper Pre 2025-26"],
    ("LKG",          "2025-26"): ["LKG Section - 2025-26"],
    ("LKG",          "2026-27"): ["LKG Section - 2026-27"],
    ("UKG",          "2025-26"): ["UKG Section - 2025-26"],
    ("UKG",          "2026-27"): ["UKG Section - 2026-27"],
    ("1st Standard", "2025-26"): ["1st Standard Section - 2025-26"],
    ("2nd Standard", "2025-26"): ["2nd Standard Section - 2025-26"],
    ("3rd Standard", "2025-26"): ["3rd Standard Section - 2025-26"],
    ("3rd Standard", "2026-27"): ["3rd Standard Section - 2026-27"],
    ("4th Standard", "2025-26"): ["4th Standard Section - 2025-26"],
    ("4th Standard", "2026-27"): ["4th Standard Section - 2026-27"],
    ("5th Standard", "2025-26"): ["5th Standard Section - 2025-26"],
    ("5th Standard", "2026-27"): ["5th Standard Section - 2026-27"],
    ("6th Standard", "2025-26"): ["6th Standard Section - 2025-26"],
    ("6th Standard", "2026-27"): ["6th Standard Section - 2026-27"],
    ("7th Standard", "2025-26"): ["7th Standard Section - 2025-26"],
    ("7th Standard", "2026-27"): ["7th Standard Section - 2026-27"],
    ("8th Standard", "2025-26"): ["8th Standard Section - 2025-26"],
    ("9th Standard", "2025-26"): ["9th Standard Section - 2025-26"],
    # 10th has a non-standard group as well
    ("10th Standard","2025-26"): ["10th Standard Section - 2025-26", "10 A section"],
}

# Exact subject names from DB
SUBJECTS = [
    "Activity", "Basic", "Computer", "Drawing", "E.V.S",
    "English", "GK", "Hindi", "Kannada", "Mass PET/YOGA",
    "Maths", "PET", "PET GAME", "PET THEORY", "Science", "Social"
]

# Subject aliases (what users say → exact DB value)
SUBJECT_ALIASES = {
    "maths": "Maths", "math": "Maths", "mathematics": "Maths",
    "english": "English", "eng": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "science": "Science", "sci": "Science",
    "social": "Social", "social science": "Social", "sst": "Social", "evs": "E.V.S",
    "e.v.s": "E.V.S", "environmental": "E.V.S",
    "computer": "Computer", "computers": "Computer", "it": "Computer",
    "drawing": "Drawing", "art": "Drawing",
    "gk": "GK", "general knowledge": "GK",
    "pet": "PET", "physical": "PET", "physical education": "PET",
    "pet game": "PET GAME", "games": "PET GAME",
    "pet theory": "PET THEORY",
    "yoga": "Mass PET/YOGA", "mass pet": "Mass PET/YOGA",
    "activity": "Activity",
    "basic": "Basic",
}

# Exact instructor names from DB
INSTRUCTOR_NAMES = [
    "Ashvini R", "Lakshmana S", "Mahadevamma K M", "Mamatha S R",
    "Manjula N", "Nandini S N", "Reshma R", "S V  Chaithrakumari",
    "Sarvamangala", "SAYAK RAY", "Shuba vidyasagar", "Vidya K M",
    "Yashodha D S", "Yogesh"
]

# Days of week
WEEKDAY_ALIASES = {
    "mon": "Monday", "monday": "Monday",
    "tue": "Tuesday", "tuesday": "Tuesday", "tues": "Tuesday",
    "wed": "Wednesday", "wednesday": "Wednesday", "weds": "Wednesday",
    "thu": "Thursday", "thursday": "Thursday", "thurs": "Thursday", "thur": "Thursday",
    "fri": "Friday", "friday": "Friday",
    "sat": "Saturday", "saturday": "Saturday",
    "sun": "Sunday", "sunday": "Sunday",
}


# ═══════════════════════════════════════════════════════════════
# PRE-PROCESSING LAYER
# Extracts structured facts from the raw query using regex + lookups.
# Returns a context dict injected into the prompt so the LLM
# receives resolved values, not raw user text.
# ═══════════════════════════════════════════════════════════════

def extract_program(q: str) -> Optional[str]:
    """
    Returns exact program name or None.
    Tries longest matches first to avoid '1st' matching inside '10th'.
    """
    ql = q.lower()
    # Try longest keys first
    sorted_keys = sorted(PROGRAM_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_keys:
        # Word-boundary check using regex
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, ql):
            return PROGRAM_ALIASES[alias]
    return None


def extract_weekday(q: str) -> Optional[str]:
    ql = q.lower()
    for alias, day in WEEKDAY_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', ql):
            return day
    return None


def extract_time(q: str) -> Optional[str]:
    """
    Extracts a time mention and converts to HH:MM:SS string.
    Handles: 2:30, 2:30 PM, 14:30, 10 AM, 10:00, half past 2
    Returns HH:MM:SS or None.
    """
    ql = q.lower()

    # Pattern: HH:MM AM/PM or HH:MM
    match = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', ql)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        meridiem = match.group(3)
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        return f"{h:02d}:{m:02d}:00"

    # Pattern: HH AM/PM (no minutes)
    match = re.search(r'\b(\d{1,2})\s*(am|pm)\b', ql)
    if match:
        h = int(match.group(1))
        meridiem = match.group(2)
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        return f"{h:02d}:00:00"

    # Pattern: "half past X" → X:30
    match = re.search(r'half past (\d{1,2})', ql)
    if match:
        h = int(match.group(1))
        return f"{h:02d}:30:00"

    # Pattern: "quarter past X" → X:15
    match = re.search(r'quarter past (\d{1,2})', ql)
    if match:
        h = int(match.group(1))
        return f"{h:02d}:15:00"

    return None


def extract_time_intent(q: str) -> str:
    """
    Distinguishes between:
    - "at 2:30"   → find the period whose window CONTAINS 2:30 (from<=time<=to)
    - "from 2:30" → find periods STARTING AT OR AFTER 2:30 (from_time >= time)
    - "after 2:30"→ same as "from"
    - "before 2:30"→ periods ending before 2:30 (to_time <= time)
    Returns: "contains" | "from" | "after" | "before"
    """
    ql = q.lower()
    if re.search(r'\bfrom\b', ql) or re.search(r'\bstarting\b', ql) or re.search(r'\bonwards\b', ql):
        return "from"
    if re.search(r'\bafter\b', ql):
        return "after"
    if re.search(r'\bbefore\b', ql):
        return "before"
    return "contains"  # default: "at X:XX" means the period containing that time


def extract_subject(q: str) -> Optional[str]:
    ql = q.lower()
    # Try aliases first
    sorted_aliases = sorted(SUBJECT_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if re.search(r'\b' + re.escape(alias) + r'\b', ql):
            return SUBJECT_ALIASES[alias]
    # Try exact subject names
    for subject in SUBJECTS:
        if subject.lower() in ql:
            return subject
    return None


def extract_period_number(q: str) -> Optional[int]:
    """Extracts ordinal or cardinal period number: '2nd period', 'period 3'"""
    match = re.search(r'period\s+(\d+)', q.lower())
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)(st|nd|rd|th)\s+period', q.lower())
    if match:
        return int(match.group(1))
    return None


def extract_instructor_name(q: str) -> Optional[str]:
    """
    Checks if any known instructor name appears in the query.
    Case-insensitive partial match.
    Returns exact DB name or None.
    """
    ql = q.lower()
    for name in INSTRUCTOR_NAMES:
        # Check first name or last name or full name
        parts = name.lower().split()
        for part in parts:
            if len(part) > 2 and re.search(r'\b' + re.escape(part) + r'\b', ql):
                return name
    return None


def extract_role_hint(q: str) -> str:
    """
    Returns "teacher" | "student" | "ambiguous"
    based on context words in the query.
    """
    ql = q.lower()
    teacher_words = {"teacher", "instructor", "sir", "madam", "faculty", "staff", "teaches", "teaching"}
    student_words = {"student", "admission", "roll", "enrolled", "pupil"}
    has_teacher = any(w in ql for w in teacher_words)
    has_student = any(w in ql for w in student_words)
    if has_teacher and not has_student:
        return "teacher"
    if has_student and not has_teacher:
        return "student"
    # Check if query contains a known instructor name
    if extract_instructor_name(q):
        return "teacher"
    return "ambiguous"


def extract_date_period(q: str) -> Optional[str]:
    """
    Returns a named period that the Frappe executor can resolve.
    "today" | "this_week" | "last_week" | "this_month" | "last_month" | None
    """
    ql = q.lower()
    if "today" in ql:
        return "today"
    if "this week" in ql:
        return "this_week"
    if "last week" in ql:
        return "last_week"
    if "this month" in ql:
        return "this_month"
    if "last month" in ql:
        return "last_month"
    if "this year" in ql or "this academic year" in ql:
        return "this_year"
    return None


def get_student_groups_for_program(program: str, year: str) -> list:
    """Returns all known group names for a program+year combination."""
    return STUDENT_GROUPS.get((program, year), [])


def get_timetable_name(program: str) -> Optional[str]:
    """Returns exact timetable name or None if no timetable exists."""
    return TIMETABLE_NAMES.get(program)


def extract_query_context(query: str, history: list = None) -> dict:
    """
    Master pre-processor. Extracts structured facts from the current query,
    then inherits any unresolved fields from the previous turn's context
    (stored as _ctx in history assistant messages).

    Follow-up resolution examples:
      Turn 1: "timetable for 5th standard on monday"
              → resolves program=5th Standard, weekday=Monday
      Turn 2: "what about tuesday?"
              → resolves weekday=Tuesday only,
                inherits program=5th Standard from turn 1

      Turn 1: "who teaches maths in 8th on monday"
              → resolves subject=Maths, program=8th Standard, weekday=Monday
      Turn 2: "and on tuesday?"
              → inherits program + subject, resolves new weekday=Tuesday
    """
    program     = extract_program(query)
    weekday     = extract_weekday(query)
    time_val    = extract_time(query)
    time_intent = extract_time_intent(query) if time_val else None
    subject     = extract_subject(query)
    period_no   = extract_period_number(query)
    role_hint   = extract_role_hint(query)
    date_period = extract_date_period(query)
    instructor  = extract_instructor_name(query)

    # ── History inheritance ───────────────────────────────────
    # Look back through last 3 assistant turns for a saved _ctx block.
    # Inherit only fields the current query left unresolved.
    prev_ctx = {}
    if history:
        checked = 0
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                ctx = turn.get("_ctx") or {}
                if ctx:
                    prev_ctx = ctx
                    break
                checked += 1
                if checked >= 3:
                    break

    if prev_ctx:
        # Program: inherit if not resolved in current query
        if program is None and prev_ctx.get("program"):
            program = prev_ctx["program"]
        # Weekday: only inherit if current query has NO weekday AND
        # has no reference suggesting a new day
        if weekday is None and prev_ctx.get("weekday"):
            weekday = prev_ctx["weekday"]
        # Subject: inherit if not found in current query
        if subject is None and prev_ctx.get("subject"):
            subject = prev_ctx["subject"]
        # Instructor: inherit if not found
        if instructor is None and prev_ctx.get("instructor_name"):
            instructor = prev_ctx["instructor_name"]
        # Role hint: only inherit if current query is ambiguous
        if role_hint == "ambiguous" and prev_ctx.get("role_hint") in ("teacher", "student"):
            role_hint = prev_ctx["role_hint"]
        # NOTE: date_period is intentionally NOT inherited.
        # "today" / "this week" are time-sensitive — must be re-stated each turn.

    # Resolve derived fields after inheritance
    timetable_name = get_timetable_name(program) if program else None
    student_groups = get_student_groups_for_program(program, CUR_YEAR) if program else []

    return {
        "program":                  program,
        "timetable_name":           timetable_name,
        "student_groups":           student_groups,
        "weekday":                  weekday,
        "time_val":                 time_val,
        "time_intent":              time_intent,
        "subject":                  subject,
        "period_no":                period_no,
        "role_hint":                role_hint,
        "date_period":              date_period,
        "instructor_name":          instructor,
        "_inherited_from_history":  bool(prev_ctx),
    }


# ═══════════════════════════════════════════════════════════════
# PLAN VALIDATOR
# Checks logical consistency of the LLM plan before execution.
# Catches the most common structural mistakes the LLM makes.
# ═══════════════════════════════════════════════════════════════

def validate_plan(plan: dict) -> list:
    """
    Returns a list of error strings.
    Empty list means the plan is structurally valid.
    """
    errors = []
    steps  = plan.get("steps") or []
    saved  = {}  # save_as → fields list

    for step in steps:
        sid      = step.get("id") or "?"
        doctype  = step.get("doctype") or ""
        fields   = step.get("fields") or []
        pipe_from  = step.get("pipe_from") or ""
        pipe_field = step.get("pipe_field") or ""
        pipe_into  = step.get("pipe_into") or ""
        save_as    = step.get("save_as") or ""

        # pipe_from references a save_as that doesn't exist yet
        if pipe_from and pipe_from not in saved:
            errors.append(
                f"Step {sid}: pipe_from='{pipe_from}' was never saved by a previous step."
            )

        # pipe_field must exist in the previous step's fields
        if pipe_from and pipe_from in saved:
            prev_fields = saved[pipe_from]
            if pipe_field and pipe_field not in prev_fields:
                errors.append(
                    f"Step {sid}: pipe_field='{pipe_field}' not in previous step fields {prev_fields}. "
                    f"Add '{pipe_field}' to the fields list of the step that saves as '{pipe_from}'."
                )

        # pipe_into must be declared if pipe_from is set
        if pipe_from and not pipe_into:
            errors.append(
                f"Step {sid}: pipe_from is set but pipe_into is empty."
            )

        # A step that is piped into must not have an empty fields list
        if not fields:
            errors.append(f"Step {sid}: fields list is empty.")

        if save_as:
            saved[save_as] = fields

    return errors


# ═══════════════════════════════════════════════════════════════
# ENTITY DETECTION  (unchanged — tested and working)
# ═══════════════════════════════════════════════════════════════

CLASS_KEYWORDS = {
    "standard", "grade", "class", "lkg", "ukg", "pre kg", "prekg",
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th",
    "10th", "students", "student", "staff", "teacher", "teachers",
    "all", "today", "this", "last", "month", "week", "year",
    "the", "a", "an", "me", "my", "their", "our"
}
STOP_WORDS = {
    "pending", "fees", "fee", "absent", "attendance", "contact",
    "details", "email", "phone", "mobile", "class", "standard",
    "grade", "program", "enrolled", "guardian", "timetable",
    "schedule", "teacher", "subject", "period"
}
WHO_TRIGGERS = [
    "who is", "show me everything about", "tell me about",
    "everything about", "details about", "info about", "what about",
    "get the details of", "details of"
]

def has_entity_name(q: str) -> bool:
    ql = q.lower().strip()
    if "'s " in ql or "\u2019s " in ql:
        return True
    for trigger in WHO_TRIGGERS:
        if trigger in ql:
            rest = ql.split(trigger)[-1].strip().split()
            if rest:
                candidate = rest[0].rstrip("?.,")
                if (len(candidate) > 2
                        and candidate not in CLASS_KEYWORDS
                        and candidate not in STOP_WORDS):
                    return True
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


# ═══════════════════════════════════════════════════════════════
# QUICK MATCH
# Fixed bug: gender_count_by_program now requires has_entity_name=False
# AND a program to have been resolved — otherwise falls to LLM
# ═══════════════════════════════════════════════════════════════

# Reference words that signal a follow-up — quick_match must be
# bypassed so history context can be used to resolve the query.
REFERENCE_WORDS = {
    "what about", "how about", "and in", "and for", "same class",
    "that class", "that standard", "that grade", "same standard",
    "those students", "them", "their", "those", "same teacher",
    "that teacher", "the same", "same subject", "also in",
    "in that", "for that", "of that", "same day", "that day",
}

def is_followup_query(q: str, history: list) -> bool:
    """
    Returns True if the query looks like a follow-up that needs
    history context to be resolved correctly.
    Two signals:
      1. Query contains a reference word (that class, what about, etc.)
      2. Query is very short (<=4 words) AND history exists —
         e.g. "what about 6th?" is only 3 words but needs prior context
    """
    if not history:
        return False
    ql = q.lower()
    if any(ref in ql for ref in REFERENCE_WORDS):
        return True
    # Very short query with history → likely a follow-up
    if len(q.split()) <= 4 and len(history) >= 2:
        return True
    return False


CONTACT_WORDS = {"contact", "email", "phone", "mobile", "number", "details", "address"}
MULTI_INTENT  = {
    "and their", "along with", "with their", "with guardian",
    "with fees", "with contact", "with attendance", "with class",
    "and fees", "and contact", "and attendance", "and guardian",
    "and teacher", "and timetable", "fee status", "guardian contact",
    "and their class", "and their fee", "also show", "also get",
    "together", "combined"
}
COMPLEX_WORDS = {
    "guardian", "fees", "fee", "contact", "absent", "attendance",
    "with their", "with guardian", "along with", "and their",
    "and contact", "and fees", "pending", "unpaid"
}
CLASS_KW_LIST = {
    "standard", "lkg", "ukg", "1st", "2nd", "3rd", "4th", "5th",
    "6th", "7th", "8th", "9th", "10th", "11th", "12th"
}
GENDER_CLASS_KW = CLASS_KW_LIST | {
    "class 1","class 2","class 3","class 4","class 5","class 6",
    "class 7","class 8","class 9","class 10","grade 1","grade 2",
    "grade 3","grade 4","grade 5","grade 6","pre kg","pre-kg"
}

def quick_match(q: str) -> str:
    if has_entity_name(q):
        return ""
    if any(w in q for w in MULTI_INTENT):
        return ""
    if len(q.split()) > 8:
        if not any(q.strip().startswith(p) for p in [
            "how many students", "how many teachers", "absent today", "present today"
        ]):
            return ""

    # Student attendance
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

    # Pending fees — only school-wide, no class filter
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
    # BUG FIX: "how many boys and girls in grade 5" was returning school-wide totals.
    # Now checks that a class keyword is present AND routes to "gender_count_total"
    # only when truly no class is mentioned at all.
    gender_triggers = [
        "how many boys", "how many girls", "how many male", "how many female",
        "number of boys", "number of girls", "boy students", "girl students",
        "male students", "female students"
    ]
    if any(t in q for t in gender_triggers):
        # Only use quick_match gender if query is simple (no complex modifiers)
        if not any(w in q for w in COMPLEX_WORDS):
            if any(w in q for w in GENDER_CLASS_KW):
                return "gender_count_by_program"
            # No class mentioned and query is short → school-wide
            if len(q.split()) <= 6:
                return "gender_count_total"
            # Longer query without class → send to LLM to interpret
            return ""

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


# ═══════════════════════════════════════════════════════════════
# LLM CALL
# ═══════════════════════════════════════════════════════════════

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
        return r.json()["choices"][0]["message"]["content"]


def clean_json(raw: str) -> str:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start:end+1]
    return "{}"


# ═══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# The context dict from the pre-processor is injected here so the
# LLM receives resolved, exact values — not raw user text.
# ═══════════════════════════════════════════════════════════════

def build_prompt(query: str, context: dict, history: list) -> list:

    # Format resolved context for injection
    ctx_lines = []
    if context.get("_inherited_from_history"):
        ctx_lines.append("  NOTE: Some fields below were inherited from the previous turn.")
        ctx_lines.append("        If the current query contradicts them, use the current query.")
    if context.get("program"):
        ctx_lines.append(f"  Resolved program:        {context['program']}")
    if context.get("timetable_name"):
        ctx_lines.append(f"  Resolved timetable name: {context['timetable_name']}")
    elif context.get("program") and context.get("timetable_name") is None:
        ctx_lines.append(f"  NOTE: No timetable exists for {context['program']}")
    if context.get("student_groups"):
        ctx_lines.append(f"  Resolved student groups: {context['student_groups']}")
    if context.get("weekday"):
        ctx_lines.append(f"  Resolved weekday:        {context['weekday']}")
    if context.get("time_val"):
        ctx_lines.append(f"  Resolved time:           {context['time_val']}")
        ctx_lines.append(f"  Time intent:             {context['time_intent']}")
    if context.get("subject"):
        ctx_lines.append(f"  Resolved subject:        {context['subject']}")
    if context.get("period_no"):
        ctx_lines.append(f"  Resolved period number:  {context['period_no']}")
    if context.get("role_hint"):
        ctx_lines.append(f"  Role hint:               {context['role_hint']}")
    if context.get("date_period"):
        ctx_lines.append(f"  Date period:             {context['date_period']}")
    if context.get("instructor_name"):
        ctx_lines.append(f"  Matched instructor:      {context['instructor_name']}")

    ctx_block = "\n".join(ctx_lines) if ctx_lines else "  (no structured facts extracted)"

    system = f"""You are a query planner for a school ERP (Frappe).
Convert the user's question into a JSON query plan.
Current academic year: {CUR_YEAR}

════════════════════════════════════════
PRE-RESOLVED CONTEXT  ← USE THESE EXACT VALUES
════════════════════════════════════════
The pre-processor has already extracted these facts from the query.
USE THEM DIRECTLY. Do not re-interpret the user's words for these fields.

{ctx_block}

════════════════════════════════════════
DOCTYPES AND FIELDS
════════════════════════════════════════
Student:            name, student_name, first_name, last_name, student_email_id, student_mobile_number, date_of_birth, gender, blood_group, city, state, joining_date, enabled, custom_father_name, custom_mother_name, custom_guardian_contact_number, custom_guardian_email_id, custom_admission_no, custom_student_category, custom_sats_id
Instructor:         name, instructor_name, custom_teacher_email, department, status
Employee:           name, employee_name, department, designation, status
Guardian:           name, guardian_name, mobile_number, email_address, occupation
Program Enrollment: name, student, student_name, program, academic_year, student_batch_name, enrollment_date
Student Group:      name, student_group_name, program, academic_year
Student Group Instructor: name, parent, instructor, instructor_name
Student Attendance: name, student, student_name, student_group, date, status, leave_type
Attendance:         name, employee, employee_name, department, attendance_date, status, leave_type
Fees:               name, student, student_name, program, academic_year, due_date, outstanding_amount, grand_total, status
Fee Structure:      name, program, academic_year, academic_term, total_amount
Fee Component:      name, parent, fees_category, amount, description
School Timetable:   name, title, class, academic_year
Timetable Period:   name, parent, period_no, weekday, subject, teacher, from_time, to_time

════════════════════════════════════════
HARD RULES
════════════════════════════════════════
1.  Never use subqueries or nested filters. Use pipe for joins.
2.  Pipe syntax: pipe_from=save_as_of_prev, pipe_field=field_in_prev_result, pipe_into=field_in_current_doctype
3.  Always include pipe_field in the fields list of the step that saves it.
4.  Name/LIKE searches: prefix with ~  e.g. {{"student_name": "~Navaneeth"}}
5.  Use pre-resolved timetable_name exactly — never construct it yourself.
6.  Timetable Period MUST be piped from School Timetable (pipe_field=name, pipe_into=parent).
7.  Fee Component MUST be piped from Fee Structure (pipe_field=name, pipe_into=parent).
8.  TIME FILTERS — use pre-resolved time_val and apply based on time_intent:
      contains → from_time: {{op:"<=", val:time_val}}, to_time: {{op:">=", val:time_val}}
      from/after → from_time: {{op:">=", val:time_val}}
      before    → to_time: {{op:"<=", val:time_val}}
9.  Subject filter: use pre-resolved subject value exactly.
10. Period number: add period_no filter if pre-resolved period_no is set.
11. For class teacher: Student Group (filter program) → Student Group Instructor (pipe name→parent). NEVER use Timetable.
12. Pending fees of a named student → Fees with student_name LIKE filter.
13. Pending fees by class → Program Enrollment (save enrolled) → pipe student→student → Fees.
14. WHICH CLASS IS [student] FROM:
      Step 1: Student doctype with student_name LIKE — save as "stu" — fields must include "name"
      Step 2: Program Enrollment — pipe_from=stu, pipe_field=name, pipe_into=student — fields: student_name, program, academic_year
15. FORBIDDEN: Student Attendance.program, Student.student_group, Program Enrollment.student_group, Fees.program
16. No OR filters. No $or.
17. post_process: set_subtract, group_count, cross_filter, missing_field
18. DISAMBIGUATION — use role_hint from context:
      teacher   → query Instructor only, all fields
      student   → query Student only, all fields
      ambiguous → query Student (save_as=student_match) AND Instructor (save_as=instructor_match) in parallel steps
19. Always attempt an answer. fallback=true only if completely unrelated to school data.

════════════════════════════════════════
PLAN FORMAT
════════════════════════════════════════
{{
  "intent": "one-line description",
  "steps": [
    {{
      "id": "s1",
      "doctype": "Student",
      "filters": {{}},
      "fields": ["name", "student_name"],
      "limit": 50,
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

════════════════════════════════════════
EXAMPLES
════════════════════════════════════════

// Fee structure: pipe Fee Structure name → Fee Component parent
Q: fee structure for 6th standard
{{"intent":"fee structure breakdown","steps":[{{"id":"s1","doctype":"Fee Structure","filters":{{"program":"6th Standard","academic_year":"{CUR_YEAR}"}},"fields":["name","program","total_amount"],"limit":1,"order_by":"name desc","save_as":"fs","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Fee Component","filters":{{}},"fields":["fees_category","amount","parent"],"limit":20,"order_by":"","save_as":"","pipe_from":"fs","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Timetable by day only — no time filter, all periods for that day
Q: timetable for 6th standard on Thursday
{{"intent":"class timetable by day","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"6th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Thursday"}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":20,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Period number given — add period_no filter
Q: who teaches 6th standard on thursday 2nd period
{{"intent":"timetable period teacher","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"6th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Thursday","period_no":2}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":5,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Time intent=contains — find the period WHOSE WINDOW contains 2:30 PM
Q: who is teaching in class 4 at 2:30 on friday
{{"intent":"timetable period by time","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"4th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Friday","from_time":{{"op":"<=","val":"14:30:00"}},"to_time":{{"op":">=","val":"14:30:00"}}}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":5,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Time intent=from — periods STARTING FROM 2:30 onwards (user said "from 2:30")
Q: which class is there for 6th standard from 2:30
{{"intent":"timetable periods from time","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"6th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"from_time":{{"op":">=","val":"14:30:00"}}}},"fields":["period_no","weekday","subject","teacher","from_time","to_time"],"limit":20,"order_by":"from_time asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Subject filter — who teaches Maths in 7th on Monday
Q: who teaches maths in 7th standard on monday
{{"intent":"subject teacher lookup","steps":[{{"id":"s1","doctype":"School Timetable","filters":{{"name":"7th Standard Time Table"}},"fields":["name","title","class"],"limit":1,"order_by":"","save_as":"tt","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Timetable Period","filters":{{"weekday":"Monday","subject":"Maths"}},"fields":["period_no","subject","teacher","from_time","to_time"],"limit":10,"order_by":"period_no asc","save_as":"","pipe_from":"tt","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Student fee lookup
Q: what is the pending fee of Sagar
{{"intent":"student fee lookup","steps":[{{"id":"s1","doctype":"Fees","filters":{{"student_name":"~Sagar","outstanding_amount":{{"op":">","val":0}}}},"fields":["student_name","outstanding_amount","due_date","program"],"limit":5,"order_by":"outstanding_amount desc","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Class pending fees — join via Program Enrollment (Fees.program is unreliable)
Q: students in 6th standard with pending fees
{{"intent":"class pending fees","steps":[{{"id":"s1","doctype":"Program Enrollment","filters":{{"program":"6th Standard","academic_year":"{CUR_YEAR}"}},"fields":["student","student_name","program"],"limit":200,"order_by":"","save_as":"enrolled","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Fees","filters":{{"outstanding_amount":{{"op":">","val":0}}}},"fields":["student","student_name","outstanding_amount","due_date"],"limit":200,"order_by":"outstanding_amount desc","save_as":"","pipe_from":"enrolled","pipe_field":"student","pipe_into":"student"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// WHICH CLASS IS [student] FROM — Step 1 saves Student.name, Step 2 pipes name→student into Program Enrollment
Q: which class is bindu from
{{"intent":"student class lookup","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Bindu"}},"fields":["name","student_name"],"limit":3,"order_by":"","save_as":"stu","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Program Enrollment","filters":{{"academic_year":"{CUR_YEAR}"}},"fields":["student_name","program","academic_year","student_batch_name"],"limit":5,"order_by":"","save_as":"","pipe_from":"stu","pipe_field":"name","pipe_into":"student"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Student profile — role_hint=student, request ALL personal fields
Q: who is navaneeth
{{"intent":"student profile","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Navaneeth"}},"fields":["student_name","student_email_id","student_mobile_number","date_of_birth","gender","joining_date","custom_admission_no","custom_father_name","custom_mother_name","custom_guardian_contact_number","city","blood_group"],"limit":3,"order_by":"","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Teacher profile — role_hint=teacher, query Instructor with all fields
Q: who is teacher Ramesh
{{"intent":"instructor profile","steps":[{{"id":"s1","doctype":"Instructor","filters":{{"instructor_name":"~Ramesh"}},"fields":["name","instructor_name","custom_teacher_email","department","status"],"limit":3,"order_by":"","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Ambiguous name — role_hint=ambiguous, query both in parallel steps
Q: get the details of nandini
{{"intent":"person profile lookup","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Nandini"}},"fields":["student_name","student_email_id","student_mobile_number","date_of_birth","gender","joining_date","custom_admission_no","custom_father_name","custom_mother_name","city","blood_group"],"limit":3,"order_by":"","save_as":"student_match","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Instructor","filters":{{"instructor_name":"~Nandini"}},"fields":["name","instructor_name","custom_teacher_email","department","status"],"limit":3,"order_by":"","save_as":"instructor_match","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Class teacher — Student Group → Student Group Instructor, NEVER use Timetable
Q: who is the class teacher of grade 3
{{"intent":"class teacher lookup","steps":[{{"id":"s1","doctype":"Student Group","filters":{{"program":"3rd Standard","academic_year":"{CUR_YEAR}"}},"fields":["name","student_group_name","program"],"limit":1,"order_by":"","save_as":"grp","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Student Group Instructor","filters":{{}},"fields":["instructor","instructor_name","parent"],"limit":5,"order_by":"","save_as":"","pipe_from":"grp","pipe_field":"name","pipe_into":"parent"}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// set_subtract — classes with no teacher assigned
Q: classes without a teacher assigned
{{"intent":"classes without teachers","steps":[{{"id":"s1","doctype":"Student Group","filters":{{}},"fields":["name","student_group_name","program"],"limit":100,"order_by":"","save_as":"all_groups","pipe_from":"","pipe_field":"","pipe_into":""}},{{"id":"s2","doctype":"Student Group Instructor","filters":{{}},"fields":["parent","instructor_name"],"limit":200,"order_by":"","save_as":"assigned","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"set_subtract","from":"all_groups","exclude":"assigned","match_field":"name","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

// group_count — rank classes by absence count
Q: which class has the most absences
{{"intent":"class absence ranking","steps":[{{"id":"s1","doctype":"Student Attendance","filters":{{"status":"Absent"}},"fields":["student_group","student_name","date"],"limit":500,"order_by":"","save_as":"absences","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"group_count","from":"absences","group_field":"student_group","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

// missing_field — students with no email
Q: students without email
{{"intent":"students missing email","steps":[{{"id":"s1","doctype":"Student","filters":{{"enabled":1}},"fields":["name","student_name","student_email_id","student_mobile_number"],"limit":500,"order_by":"","save_as":"students","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[{{"type":"missing_field","from":"students","field":"student_email_id","save_as":"result"}}],"fallback":false,"fallback_msg":""}}

// Per-student attendance — filter by student_name LIKE, not student_group
Q: how many times was Navaneeth absent this month
{{"intent":"student absence count","steps":[{{"id":"s1","doctype":"Student Attendance","filters":{{"student_name":"~Navaneeth","status":"Absent"}},"fields":["student_name","student_group","date","status"],"limit":100,"order_by":"date desc","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

// Parent info — student context, use custom_father_name/custom_mother_name
Q: what is navaneeth's father's name
{{"intent":"student parent info","steps":[{{"id":"s1","doctype":"Student","filters":{{"student_name":"~Navaneeth"}},"fields":["student_name","custom_father_name","custom_mother_name","custom_guardian_contact_number"],"limit":3,"order_by":"","save_as":"","pipe_from":"","pipe_field":"","pipe_into":""}}],"post_process":[],"fallback":false,"fallback_msg":""}}

Return ONLY raw JSON. No markdown. No explanation.

Q: """

    messages = [{"role": "system", "content": system}]
    # Only inject last 3 turns (6 messages: 3 user + 3 assistant).
    # _ctx keys are internal metadata — strip them before sending to LLM.
    clean_history = []
    for h in history[-6:]:
        role    = h.get("role") or ""
        content = h.get("content") or ""
        if role and content:
            clean_history.append({"role": role, "content": content})
    for h in clean_history:
        messages.append(h)
    messages.append({"role": "user", "content": query})
    return messages


# ═══════════════════════════════════════════════════════════════
# LLM PLAN ORCHESTRATOR
# Pre-process → LLM → Validate → (Retry if errors) → Return
# ═══════════════════════════════════════════════════════════════

async def llm_plan(query: str, history: list) -> dict:
    default = {
        "intent": "",
        "steps": [],
        "post_process": [],
        "fallback": True,
        "fallback_msg": "I'm not sure how to answer that. Try asking about students, attendance, fees, timetables, or teachers."
    }

    try:
        # Stage 1: Pre-process — extract structured facts, inheriting from history
        context  = extract_query_context(query, history)

        # Stage 2: LLM call with resolved context injected
        messages = build_prompt(query, context, history)
        raw      = await call_llm(messages, max_tokens=900)
        parsed   = json.loads(clean_json(raw))

        if parsed.get("fallback"):
            return {**default, "fallback_msg": parsed.get("fallback_msg") or default["fallback_msg"]}

        steps = parsed.get("steps") or []
        if not steps:
            return default

        # Stage 3: Validate plan structure
        errors = validate_plan(parsed)

        if errors:
            # Stage 3b: Retry with errors fed back to LLM
            error_note = "CORRECTION NEEDED — your previous plan had these structural errors:\n"
            error_note += "\n".join(f"  - {e}" for e in errors)
            error_note += "\nFix these issues and return a corrected plan."

            retry_messages = build_prompt(query, context, history)
            # Inject error note as an assistant turn + user correction request
            retry_messages.append({"role": "assistant", "content": raw})
            retry_messages.append({"role": "user", "content": error_note})

            raw2    = await call_llm(retry_messages, max_tokens=900)
            parsed2 = json.loads(clean_json(raw2))

            if parsed2.get("steps"):
                parsed = parsed2

        return {
            "intent":       parsed.get("intent") or "",
            "steps":        parsed.get("steps") or [],
            "post_process": parsed.get("post_process") or [],
            "fallback":     False,
            "fallback_msg": "",
            "_ctx":         context,   # carry resolved context for Frappe to store in history
        }

    except Exception as e:
        default["fallback_msg"] = f"Planner error: {str(e)[:100]}"
        return default


# ═══════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════

@app.post("/plan", response_model=PlanResponse)
async def plan_query(
    req: PlanRequest,
    x_api_key: Optional[str] = Header(None)
):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    message = req.message.strip()
    if not message:
        return PlanResponse(type="fallback", message="Empty message.")

    q = message.lower()

    history = req.history or []

    # If this looks like a follow-up query, skip quick_match entirely
    # so the LLM can use history context to resolve it correctly.
    # e.g. "what about 6th?" after a 5th standard query.
    if not is_followup_query(q, history):
        qm = quick_match(q)
        if qm:
            return PlanResponse(type="quick", pattern=qm)

    # LLM path — pre-process (with history) + plan + validate
    plan = await llm_plan(message, history)

    if plan.get("fallback"):
        return PlanResponse(
            type="fallback",
            message=plan.get("fallback_msg") or "Sorry, I couldn't understand that."
        )

    # Forward the resolved context so Frappe can store it in history
    # for follow-up query inheritance
    ctx = plan.pop("_ctx", None)
    return PlanResponse(type="orm_plan", plan=plan, ctx=ctx)
