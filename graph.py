import os
import json
import re
import sys
import tempfile
from typing import TypedDict

import pymupdf
import serpapi
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


class State(TypedDict):
    resume: str
    profile: dict
    preferences: str
    search_query: str
    search_location: str
    jobs: list
    ranked_jobs: list
    selected_job: dict
    tailored_resume: str
    feedback: str
    cover_letter: str
    approval: str
    application_status: str   


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

APPROVE_WORDS = {"approve", "approved", "yes", "y", "ok", "looks good"}


def is_approved(value) -> bool:
    """Single place that decides what counts as approval (graph + API share it)."""
    return str(value or "").strip().lower() in APPROVE_WORDS


def dumps(obj) -> str:
    
    return json.dumps(obj, indent=2, default=str)


def text_from_response(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def extract_json(text):
    """Extract a JSON object OR array from an LLM response.
 
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append((start, end))

    if not candidates:
        raise ValueError("No JSON found in model response")

    start, end = min(candidates)  # whichever opens first is the outer structure
    return json.loads(text[start:end + 1])


def parse_review(result):
    """Normalise a human review answer to (approval, feedback).
 
    """
    if isinstance(result, dict):
        approval = str(result.get("approval", ""))
        feedback = str(result.get("feedback", ""))
    else:
        approval = str(result)
        feedback = ""

    if is_approved(approval):
        return "approve", ""

    if not feedback and approval.strip().lower() not in {"", "revise", "reject", "no", "n"}:
        feedback = approval.strip()   

    return "revise", feedback


def parse_preferences(answer):
    """Preferences may arrive as a dict or as a string (plain text or JSON)."""
    if isinstance(answer, dict):
        return answer
    text = str(answer).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    return {"role": text, "location": ""}


def extract_resume(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def resume_agent(state):
    resume = state["resume"][:15000]

    prompt = f"""
You are a Resume Intelligence Agent.

Analyze the master resume and extract only information explicitly present
in the resume.

Return ONLY valid JSON in exactly this structure:

{{
    "name": "",
    "phone": "",
    "headline": "",
    "bio": "",
    "location": {{
        "city": "",
        "state": "",
        "country": ""
    }},
    "social_links": {{
        "github": "",
        "linkedin": ""
    }},
    "skills": [],
    "experience": [],
    "education": [],
    "projects": [],
    "certifications": [],
    "suitable_roles": []
}}

Rules:
- Do not invent information.
- If information is not present, use an empty string or empty array.
- Skills must come from the resume.
- Experience must contain only actual experience from the resume.
- Suitable roles should be inferred only from the candidate's demonstrated skills
  and experience.

MASTER RESUME:

{resume}
"""

    response = llm.invoke(prompt)
    text = text_from_response(response)
    profile = extract_json(text)
    return {"profile": profile}


def preference_agent(state):
    profile = state["profile"]

    answer = interrupt({
        "type": "preferences",
        "message": "Enter your job preferences",
        "profile": profile
    })

    
    prefs = parse_preferences(answer)

    return {
        "preferences": json.dumps(prefs, default=str),
        "search_query": prefs.get("role", ""),
        "search_location": prefs.get("location", "")
    }


def search_planner_agent(state):
    profile = state["profile"]
    preferences = state["preferences"]

    prompt = f"""
You are a job search query planner.

Candidate profile:
{dumps(profile)}

Candidate preferences:
{preferences}

Create the best Google Jobs search query and location.

Return ONLY valid JSON:
{{
    "query": "",
    "location": ""
}}
"""

    response = llm.invoke(prompt)
    data = extract_json(text_from_response(response))

    return {
        "search_query": data.get("query", ""),
        "search_location": data.get("location", "")
    }


def search_jobs(query, location):
    api_key = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
    client = serpapi.Client(api_key=api_key)

    results = client.search(
        engine="google_jobs",
        q=query,
        location=location
    )
    jobs = results.get("jobs_results", [])

    cleaned = []

    for job in jobs:
        apply_options = job.get("apply_options", [])
        direct_link = apply_options[0].get("link", "") if apply_options else ""
        cleaned.append({
            "title": job.get("title", ""),
            "company": job.get("company_name", ""),
            "location": job.get("location", ""),
            "description": job.get("description", ""),
            "via": job.get("via", ""),
            "link": direct_link or job.get("share_link", ""),
            "share_link": job.get("share_link", ""),
            "apply_options": apply_options,
            "extensions": job.get("detected_extensions", {})
        })

    return cleaned


def job_search_agent(state):
    jobs = search_jobs(
        state["search_query"],
        state["search_location"]
    )

    return {"jobs": jobs}


def ranking_agent(state):
    profile = state["profile"]
    preferences = state["preferences"]
    jobs = state["jobs"]

    prompt = f"""
You are a job ranking agent.

Rank the jobs based on how well they match the candidate.

Candidate profile:
{dumps(profile)}

Candidate preferences:
{preferences}

Jobs:
{dumps(jobs)}

Return ONLY a JSON array.

Each item must contain:
{{
    "rank": 1,
    "score": 0,
    "reason": "",
    "job": {{}}
}}
"""

    response = llm.invoke(prompt)
    ranked_jobs = extract_json(text_from_response(response))

    
    if isinstance(ranked_jobs, dict):
        ranked_jobs = next(
            (v for v in ranked_jobs.values() if isinstance(v, list)), []
        )

    return {"ranked_jobs": ranked_jobs}


def show_jobs(state):
    ranked_jobs = state["ranked_jobs"]

    if not ranked_jobs:
        raise ValueError("No jobs matched this search. Try different preferences.")

    selected = interrupt({
        "type": "job_selection",
        "jobs": ranked_jobs,
        "message": "Select a job"
    })

     
    raw = selected.get("index", 1) if isinstance(selected, dict) else selected
    try:
        index = int(raw) - 1
    except (TypeError, ValueError):
        index = 0

    if index < 0 or index >= len(ranked_jobs):
        index = 0

    return {"selected_job": ranked_jobs[index].get("job", {})}


def tailor_resume_agent(state):
    resume = state["resume"]
    profile = state["profile"]
    selected_job = state["selected_job"]
    feedback = state.get("feedback", "")

    prompt = f"""
You are a professional resume tailoring agent.

Master resume:
{resume}

Candidate profile:
{dumps(profile)}

Selected job:
{dumps(selected_job)}

Previous feedback:
{feedback}

Create an ATS-friendly tailored resume for this job.

Do not invent experience, skills, education, projects, certifications,
achievements, or other facts.

Only reorganize, rewrite, and emphasize information supported by the
master resume and candidate profile.
"""

    response = llm.invoke(prompt)

    return {"tailored_resume": text_from_response(response)}


def review_resume(state):
    result = interrupt({
        "type": "resume_review",
        "resume": state["tailored_resume"],
        "message": "Approve the resume or request changes"
    })

    approval, feedback = parse_review(result)

    return {
        "approval": approval,
        "feedback": feedback
    }


def resume_review_route(state):
    if is_approved(state.get("approval", "")):
        return "cover_letter"

    return "tailor_resume"


def cover_letter_agent(state):
    profile = state["profile"]
    selected_job = state["selected_job"]
    tailored_resume = state["tailored_resume"]

    prompt = f"""
You are a professional cover letter writer.

Candidate profile:
{dumps(profile)}

Tailored resume:
{tailored_resume}

Selected job:
{dumps(selected_job)}

Write a concise, professional cover letter.

Do not invent any facts.
"""

    response = llm.invoke(prompt)

    return {"cover_letter": text_from_response(response)}


def final_review(state):
    result = interrupt({
        "type": "final_review",
        "resume": state["tailored_resume"],
        "cover_letter": state["cover_letter"],
        "message": "Approve the application or request changes"
    })

     
    approval, feedback = parse_review(result)

    return {
        "approval": approval,
        "feedback": feedback
    }


def final_review_route(state):
    if is_approved(state.get("approval", "")):
        return "application"

    return "tailor_resume"


def text_to_pdf(text: str, output_path: str):
    """Convert raw text or markdown into a paginated PDF for ATS upload."""
    doc = pymupdf.open()
    margin = 40
    page_width, page_height = 595, 842  # A4
    page = doc.new_page(width=page_width, height=page_height)
    y = margin + 10
    line_height = 14

    for raw_line in (text or "").split("\n"):
        line = raw_line.rstrip()
        max_chars = 95
        chunks = [line[i:i + max_chars] for i in range(0, max(len(line), 1), max_chars)]
        for chunk in chunks:
            if y + line_height > page_height - margin:
                page = doc.new_page(width=page_width, height=page_height)
                y = margin + 10
            page.insert_text((margin, y), chunk, fontsize=10, fontname="helv")
            y += line_height

    doc.save(output_path)
    doc.close()


def launch_playwright_browser(playwright_ctx, headless: bool = False):
    """Launch system Chrome or Edge for maximum bot detection resistance."""
    for channel in ["chrome", "msedge", None]:
        try:
            kwargs = {"headless": headless}
            if channel:
                kwargs["channel"] = channel
            return playwright_ctx.chromium.launch(**kwargs)
        except Exception:
            continue
    return playwright_ctx.chromium.launch(headless=headless)


def autofill_job_application(page, profile: dict, resume_pdf_path: str, cover_letter: str = ""):
    """Intelligently detects and pre-fills common applicant form fields."""
    filled = []

    name = profile.get("name", "")
    email = profile.get("email", "")
    phone = profile.get("phone", "")

    location = profile.get("location", {})
    if isinstance(location, dict):
        city = location.get("city", "")
        state_loc = location.get("state", "")
        loc_str = f"{city}, {state_loc}".strip(", ")
    else:
        loc_str = str(location or "")

    social = profile.get("social_links", {})
    linkedin = social.get("linkedin", "") if isinstance(social, dict) else ""
    github = social.get("github", "") if isinstance(social, dict) else ""

    parts = name.split()
    first_name = parts[0] if parts else ""
    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

    mappings = [
        ("input[name*='name' i]:not([name*='first' i]):not([name*='last' i]), input[id*='name' i]:not([id*='first' i]):not([id*='last' i]), input[autocomplete='name']", name, "Full Name"),
        ("input[name*='first' i], input[id*='first' i], input[autocomplete='given-name']", first_name, "First Name"),
        ("input[name*='last' i], input[id*='last' i], input[autocomplete='family-name']", last_name, "Last Name"),
        ("input[type='email'], input[name*='email' i], input[id*='email' i], input[autocomplete='email']", email, "Email"),
        ("input[type='tel'], input[name*='phone' i], input[id*='phone' i], input[name*='mobile' i]", phone, "Phone"),
        ("input[name*='city' i], input[name*='location' i], input[id*='location' i]", loc_str, "Location"),
        ("input[name*='linkedin' i], input[id*='linkedin' i]", linkedin, "LinkedIn"),
        ("input[name*='github' i], input[name*='website' i], input[id*='github' i], input[name*='portfolio' i]", github, "GitHub/Website"),
    ]

    for selector, val, label in mappings:
        if not val:
            continue
        try:
            loc = page.locator(selector).first
            if loc.count() > 0 and loc.is_visible(timeout=1000):
                if not loc.input_value():
                    loc.fill(val)
                    filled.append(label)
        except Exception:
            pass

    # Cover letter
    if cover_letter:
        try:
            cl_loc = page.locator("textarea[name*='cover' i], textarea[id*='cover' i], textarea[placeholder*='cover' i]").first
            if cl_loc.count() > 0 and cl_loc.is_visible(timeout=1000):
                if not cl_loc.input_value():
                    cl_loc.fill(cover_letter)
                    filled.append("Cover Letter")
        except Exception:
            pass

    # Tailored resume upload
    if resume_pdf_path and os.path.exists(resume_pdf_path):
        try:
            file_loc = page.locator("input[type='file']").first
            if file_loc.count() > 0:
                file_loc.set_input_files(resume_pdf_path)
                filled.append("Tailored Resume PDF")
        except Exception:
            pass

    return filled


def application_agent(state):
    """Browser agent that visits the job link, pre-fills the form, and pauses for review."""
    selected_job = state.get("selected_job", {})
    profile = state.get("profile", {})
    tailored_resume = state.get("tailored_resume", "")
    cover_letter = state.get("cover_letter", "")

    
    job_url = selected_job.get("link") or selected_job.get("share_link")
    if not job_url and selected_job.get("apply_options"):
        job_url = selected_job["apply_options"][0].get("link")

    if not job_url:
        return {"application_status": "skipped (no link provided)"}

     
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf_path = temp_pdf.name
    temp_pdf.close()

    status = "ready"
    browser = None
    playwright_ctx = None

    try:
        from playwright.sync_api import sync_playwright

        text_to_pdf(tailored_resume, temp_pdf_path)

        playwright_ctx = sync_playwright().start()
         
        headless = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() in {"true", "1", "yes"}
        browser = launch_playwright_browser(playwright_ctx, headless=headless)
        context = browser.new_context()
        page = context.new_page()

        print(f"\n[Application Agent] Opening job application: {job_url}")
        page.goto(job_url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)

        filled_fields = autofill_job_application(page, profile, temp_pdf_path, cover_letter)

         
        result = interrupt({
            "type": "submit_confirmation",
            "message": f"Pre-filled application for {selected_job.get('title', 'Role')} at {selected_job.get('company', 'Company')}. Please review the open browser window and approve to submit.",
            "job_url": job_url,
            "fields_filled": filled_fields
        })

        if is_approved(result):
            
            submit_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "button:has-text('Submit Application')",
                "button:has-text('Submit')",
                "button:has-text('Apply')",
            ]
            submitted = False
            for btn_sel in submit_selectors:
                try:
                    btn = page.locator(btn_sel).first
                    if btn.count() > 0 and btn.is_visible(timeout=1000):
                        btn.click()
                        submitted = True
                        page.wait_for_timeout(3000)
                        break
                except Exception:
                    pass

            status = "submitted" if submitted else "ready_for_manual_submit"
        else:
            status = "saved_as_draft"

    except ImportError:
        status = "ready (playwright not installed)"
    except Exception as e:
        status = f"error: {str(e)}"
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright_ctx:
            try:
                playwright_ctx.stop()
            except Exception:
                pass
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass

    return {"application_status": status}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build the workflow.

    Pass a persistent checkpointer (e.g. MongoDBSaver) in production so
    paused workflows survive a server restart. Falls back to MemorySaver.
    """
    workflow = StateGraph(State)

    workflow.add_node("preference_agent", preference_agent)
    workflow.add_node("search_planner", search_planner_agent)
    workflow.add_node("job_search", job_search_agent)
    workflow.add_node("ranking", ranking_agent)
    workflow.add_node("show_jobs", show_jobs)
    workflow.add_node("tailor_resume", tailor_resume_agent)
    workflow.add_node("review_resume", review_resume)
    workflow.add_node("cover_letter", cover_letter_agent)
    workflow.add_node("final_review", final_review)
    workflow.add_node("application", application_agent)

    workflow.add_edge(START, "preference_agent")
    workflow.add_edge("preference_agent", "search_planner")
    workflow.add_edge("search_planner", "job_search")
    workflow.add_edge("job_search", "ranking")
    workflow.add_edge("ranking", "show_jobs")
    workflow.add_edge("show_jobs", "tailor_resume")
    workflow.add_edge("tailor_resume", "review_resume")

    workflow.add_conditional_edges(
        "review_resume",
        resume_review_route,
        {
            "cover_letter": "cover_letter",
            "tailor_resume": "tailor_resume"
        }
    )

    workflow.add_edge("cover_letter", "final_review")

    workflow.add_conditional_edges(
        "final_review",
        final_review_route,
        {
            "application": "application",
            "tailor_resume": "tailor_resume"
        }
    )

    workflow.add_edge("application", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


if __name__ == "__main__":
     
    resume_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RESUME_PATH")
    if not resume_path:
        raise SystemExit("Usage: python graph.py path/to/resume.pdf")

    resume_text = extract_resume(resume_path)

    graph = build_graph()

    initial_state = {
        "resume": resume_text,
        "profile": {},
        "preferences": "",
        "search_query": "",
        "search_location": "",
        "jobs": [],
        "ranked_jobs": [],
        "selected_job": {},
        "tailored_resume": "",
        "feedback": "",
        "cover_letter": "",
        "approval": "",
        "application_status": ""
    }

    config = {"configurable": {"thread_id": "user-1"}}

     
    result = graph.invoke(initial_state, config=config)
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print("\n---", payload.get("message", "Input needed"), "---")
        if payload.get("type") == "job_selection":
            for i, item in enumerate(payload["jobs"], 1):
                job = item.get("job", {})
                print(f"{i}. {job.get('title', '')} | {job.get('company', '')} | "
                      f"Score: {item.get('score', '')}")
        elif payload.get("type") == "submit_confirmation":
            print(f"Target URL: {payload.get('job_url')}")
            fields = payload.get("fields_filled", [])
            print(f"Fields filled: {', '.join(fields) if fields else 'None detected'}")
            print("Please review the open browser window. Type 'approve' to submit, or 'cancel' to save draft.")
        elif payload.get("resume"):
            print(payload["resume"])
            if payload.get("cover_letter"):
                print("\n", payload["cover_letter"])
        answer = input("> ")
        result = graph.invoke(Command(resume=answer), config=config)

    print(result)
