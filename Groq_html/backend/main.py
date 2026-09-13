import io
import os
import re
import json
import uuid
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Header
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
import pdfplumber
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="TalentFilter Pro Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL = "openai/gpt-oss-120b"

# In-memory candidate storage
CANDIDATE_STORE: Dict[str, dict] = {}
VALID_TOKENS = {"recruiter-session-token-123"}

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication token required.")
    token = authorization.replace("Bearer ", "").strip()
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid or expired token.")
    return token

class AuthRequest(BaseModel):
    username: str
    password: str

class MetricScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=100.0)
    reasoning: str

class CandidateEvaluation(BaseModel):
    candidate_id: str
    candidate_name: str
    metrics: List[MetricScore]
    strengths: List[str]
    red_flags: List[str]

class ChatMessage(BaseModel):
    candidate_id: str
    message: str

class TailorResumeRequest(BaseModel):
    candidate_id: str

def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    try:
        if name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        elif name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif name.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Extraction error ({filename}): {e}")
    return ""

@app.post("/api/auth/login")
def login(auth: AuthRequest):
    if auth.username == "admin" and auth.password == "recruit123":
        return {"token": "recruiter-session-token-123", "user": "Admin Recruiter"}
    raise HTTPException(status_code=401, detail="Invalid username or password.")

@app.post("/api/shortlist")
async def shortlist_candidates(
    jd_file: UploadFile = File(...),
    resume_files: List[UploadFile] = File(...),
    token: str = Depends(verify_token)
):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    jd_text = extract_text(await jd_file.read(), jd_file.filename)
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Could not parse text from Job Description.")

    evaluations = []

    for resume in resume_files:
        cand_id = str(uuid.uuid4())[:8]
        resume_bytes = await resume.read()
        resume_text = extract_text(resume_bytes, resume.filename)
        if not resume_text.strip():
            continue

        prompt = f"""
        You are an expert HR talent evaluation engine. Score this resume against the Job Description across 10 metrics (0-100 each):
        1. Hard Skill Alignment
        2. Experience Relevance
        3. Seniority & Scope
        4. Educational Qualification
        5. Soft Skills & Leadership
        6. Quantifiable Impact
        7. Tool & Platform Stack
        8. Career Continuity
        9. Domain Experience
        10. Communication Quality

        Respond ONLY with a JSON object strictly matching this schema:
        {{
          "candidate_name": "Full Name",
          "metrics": [
            {{"name": "Hard Skill Alignment", "score": 85.0, "reasoning": "..."}},
            {{"name": "Experience Relevance", "score": 80.0, "reasoning": "..."}},
            {{"name": "Seniority & Scope", "score": 75.0, "reasoning": "..."}},
            {{"name": "Educational Qualification", "score": 90.0, "reasoning": "..."}},
            {{"name": "Soft Skills & Leadership", "score": 80.0, "reasoning": "..."}},
            {{"name": "Quantifiable Impact", "score": 70.0, "reasoning": "..."}},
            {{"name": "Tool & Platform Stack", "score": 85.0, "reasoning": "..."}},
            {{"name": "Career Continuity", "score": 90.0, "reasoning": "..."}},
            {{"name": "Domain Experience", "score": 80.0, "reasoning": "..."}},
            {{"name": "Communication Quality", "score": 85.0, "reasoning": "..."}}
          ],
          "strengths": ["...", "..."],
          "red_flags": ["..."]
        }}

        [JOB DESCRIPTION]
        {jd_text[:6000]}

        [RESUME TEXT]
        {resume_text[:6000]}
        """

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a talent evaluation engine. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            data["candidate_id"] = cand_id

            CANDIDATE_STORE[cand_id] = {
                "name": data.get("candidate_name", resume.filename),
                "resume_text": resume_text,
                "jd_text": jd_text,
                "history": []
            }

            evaluations.append(data)
        except Exception as e:
            print(f"Error evaluating {resume.filename}: {e}")

    return {"candidates": evaluations}

@app.post("/api/candidate/chat")
async def chat_with_candidate(chat_req: ChatMessage, token: str = Depends(verify_token)):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    cand = CANDIDATE_STORE.get(chat_req.candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate session not found.")

    system_prompt = f"""
    You are an AI recruiting co-pilot interrogating a specific candidate's resume for an HR manager.
    Answer questions truthfully based ONLY on the Candidate Resume and Job Description.
    If the resume does not mention something, explicitly state that it is not documented.

    IMPORTANT FORMATTING RULES:
    - Use clean bullet points and concise paragraphs.
    - DO NOT format answers as Markdown tables or ASCII grids.
    - Keep output easy to read in a narrow mobile chat drawer.

    [JOB DESCRIPTION]
    {cand['jd_text'][:4000]}

    [CANDIDATE RESUME: {cand['name']}]
    {cand['resume_text'][:7000]}
    """

    cand["history"].append({"role": "user", "content": chat_req.message})
    messages = [{"role": "system", "content": system_prompt}] + cand["history"]

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=600
    )

    reply = completion.choices[0].message.content
    cand["history"].append({"role": "assistant", "content": reply})
    return {"reply": reply}

# --- NEW FEATURE: Generate Standard Downloadable Tailored Resume ---
@app.post("/api/candidate/tailor-resume")
async def generate_tailored_resume(req: TailorResumeRequest, token: str = Depends(verify_token)):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    cand = CANDIDATE_STORE.get(req.candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate session not found.")

    prompt = f"""
    You are a professional executive resume writer and ATS specialist.
    Rewrite and format this candidate's raw resume so that it is strategically aligned and optimized for the target Job Description.
    
    GUIDELINES:
    1. Standard, ATS-compliant layout: Header, Professional Summary, Core Competencies, Professional Experience, Education & Certifications.
    2. Emphasize matching skills, technologies, and achievements relevant to the JD without fabricating untrue qualifications.
    3. Use quantifiable metrics where present in the original resume.
    4. Output clean, complete, standalone HTML enclosed inside ```html ... ``` with standard CSS styling for direct downloading and printing (A4 standard format).

    [TARGET JOB DESCRIPTION]
    {cand['jd_text'][:4000]}

    [ORIGINAL CANDIDATE RESUME]
    {cand['resume_text'][:8000]}
    """

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional resume architect. Output a beautifully styled HTML resume."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2200
        )
        content = completion.choices[0].message.content.strip()
        
        # Extract HTML code if fenced in markdown
        html_match = re.search(r"```(?:html)?\s*(<!DOCTYPE html>.*?</html>)\s*```", content, re.DOTALL | re.IGNORECASE)
        if html_match:
            clean_html = html_match.group(1)
        else:
            clean_html = re.sub(r"^```(?:html)?\s*|```$", "", content, flags=re.MULTILINE).strip()

        return {
            "candidate_id": req.candidate_id,
            "candidate_name": cand["name"],
            "html_resume": clean_html
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate tailored resume: {str(e)}")
