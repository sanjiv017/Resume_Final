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
    allow_credentials=False,
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

import io
import re
import unicodedata
from docx import Document
import pdfplumber

def clean_text_for_llm(raw_text: str) -> str:
    """
    Standardizes and cleans text extracted from PDF, DOCX, or TXT
    to eliminate garbage tokens and optimize token efficiency for LLMs.
    """
    if not raw_text:
        return ""

    # 1. Normalize Unicode (resolves ligatures like 'ﬁ' -> 'fi', curly quotes, non-breaking spaces)
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Remove CID and glyph error encodings common in PDF extraction (e.g., (cid:120))
    text = re.sub(r"\(cid:\d+\)", " ", text)

    # 3. Fix hyphenated line breaks (e.g., "trans- \n formation" -> "transformation")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 4. Standardize diverse bullet point characters to a uniform symbol (•)
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u25CB\u25CF\u25AA\u25AB\u25C6\u25C7\u25A0\u25A1▪►*–—]\s*", "• ", text)

    # 5. Remove non-printable / control characters (keep standard newlines and tabs)
    text = re.sub(r"[^\x20-\x7E\n\t•]", " ", text)

    # 6. Normalize multiple horizontal spaces and tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 7. Clean up whitespace per line
    lines = [line.strip() for line in text.split("\n")]

    # 8. Collapse 3+ consecutive newlines down to 2 (preserves paragraph hierarchy without wasting tokens)
    clean_text = "\n".join(lines)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return clean_text.strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    raw_text = ""

    try:
        if name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            # Extract paragraphs and table contents (often missed in DOCX resumes)
            para_texts = [p.text for p in doc.paragraphs if p.text]
            table_texts = [
                " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                for table in doc.tables
                for row in table.rows
            ]
            raw_text = "\n".join(para_texts + table_texts)

        elif name.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_texts = []
                for page in pdf.pages:
                    # layout=False avoids spacing artifacts from two-column layouts
                    page_str = page.extract_text(layout=False, x_tolerance=2, y_tolerance=2) or ""
                    page_texts.append(page_str)
                raw_text = "\n".join(page_texts)

        elif name.endswith(".txt"):
            raw_text = file_bytes.decode("utf-8", errors="ignore")

    except Exception as e:
        print(f"Extraction error ({filename}): {e}")
        return ""

    # Run the raw text through the cleaner
    return clean_text_for_llm(raw_text)

@app.get("/")
def health_check():
    return {"status": "backend operational", "model": GROQ_MODEL}

@app.post("/api/auth/login")
def login(auth: AuthRequest):
    if auth.username == "admin" and auth.password == "recruit123":
        return {"token": "recruiter-session-token-123", "user": "Admin Recruiter"}
    raise HTTPException(status_code=401, detail="Invalid username or password.")

@app.post("/api/shortlist")
@app.post("/shortlist")
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
        1. Hard Skill Alignment : Direct match and semantic overlap of core programming languages, tools, or domain-specific certifications.
        2. Experience Relevance : Depth of hands-on experience in identical or closely adjacent job roles and industry verticals.
        3. Seniority & Scope    : Alignment of past titles, team sizes managed, or technical scope compared to JD seniority levels.
        4. Educational Qualification : Degree level, specialization field, and accreditation against minimum JD requirements or check if college is IIT or NIT or IIM.
        5. Soft Skills & Leadership : Evidence of cross-functional communication, mentoring, problem-solving, and ownership.
        6. Quantifiable Impact      : Presence of metric-backed achievements (e.g., "$1.2M revenue generated," "reduced latency by 35%").
        7. Tool & Platform Stack    : Match on secondary infrastructure, libraries, APIs, or vendor tooling specified in the JD.
        8. Career Continuity        : Reasonable role progression, clear promotion trajectory, and stability.
        9. Domain Experience        : Prior direct context in the company's specific vertical (e.g., FinTech, SaaS, BioTech).
        10. Communication Quality   : Clarity, formatting rigor, structural conciseness, and absence of typographical errors.

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
    You are an objective, precise technical talent assessor assisting an HR recruiter.
    Answer questions strictly, truthfully, and objectively based ONLY on the Candidate Resume and Job Description below.

    CRITICAL RULES (ANTI-EXAGGERATION & STRUCTURE):
    1. Don't Exaggeration & Strict Grounding:
       - State only factual details in short by understanding context. No need to write exactly as in the docs.
       - NEVER inflate seniority, skill depth, or years of experience.
       - If a skill, tool, company, or domain is not explicitly documented, clearly state: "Not mentioned in the resume."
    2. Structured Format:
       - Start immediately with a 1-sentence direct answer.
       - Use clean, concise bullet points (•) for itemized details.
       - Use inline bold text for key terms, technologies, and dates.
    3. FORBIDDEN FORMATS:
       - DO NOT output Markdown tables, pipes ('|'), dashes ('---'), or ASCII grid matrices.
       - DO NOT use conversational filler like "Sure!", "Here is a breakdown", or "Based on my analysis".
    4. Keep the total response concise, scannable, and under 150 words.
    5. Follow this rule strictly and on first priority; Keep each answer 1 or 2 line max until user asks answers in detail 
       For example: Query: Years of experience Answer: 4 years , Query: Domain knowledge Answer: FinTech, Pharma etc. You have to
       answer like this.

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
        temperature=0.15,
        max_tokens=450
    )

    reply = completion.choices[0].message.content
    cand["history"].append({"role": "assistant", "content": reply})
    return {"reply": reply}

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
