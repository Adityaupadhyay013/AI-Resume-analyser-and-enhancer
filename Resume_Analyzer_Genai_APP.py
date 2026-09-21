from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser 
import os
from datetime import date
from openai import OpenAI
from dotenv import load_dotenv
import pymupdf
from typing import Optional, List, Literal, Any
import re
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY") , 
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
Hiredecision = Literal["Reject" , "Borderline" , "Hire"]
Confidence = Literal["Low" , "Medium" , "High"]
True_complexity = Literal["Low" , "Medium" , "High"]
Severity = Literal["High" , "Medium" , "Low"]
class Final_verdict(BaseModel):
    hire_decision: Hiredecision = Field(default_factory = Hiredecision)
    confidence: Confidence = Field(default_factory = Confidence)
    one_line_reason: str = ""
class Overall_score(BaseModel):
    score: str = ""
    calibrated_again_market: str = ""
class Signalvsnoise(BaseModel):
    strong_signals: List[str] = Field(default_factory = list)
    weak_signals: List[str] =  Field(default_factory = list)
    filler_or_fluff: List[str] = Field(default_factory = list)
class Skills_deep_dive(BaseModel):
      skill: str = ""
      claimed_level: str = ""
      actual_level_inferred: str = ""
      evidence:str = ""
      gap:str = ""
class Project_forensic_analysis(BaseModel):
      project_name: str = ""
      true_complexity:True_complexity 
      depth_of_understanding: str = ""
      real_world_value:str = ""
      missing_elements: List[str] = Field(default_factory = list)
      verdict: str = ""
class Experience_truth_check(BaseModel):
    is_experience_meaningful: str = ""
    impact_level: str = ""
    hidden_concerns: List[str] = Field(default_factory = list)
class Benchmark_comparison(BaseModel):
    vs_average_candidate: str = ""
    vs_strong_candidate: str = ""
    vs_top_5_percent: str = ""
class Critical_gaps_ranked(BaseModel):
    gap: str = ""
    severity:  str = ""
    why_it_matters: str = ""
class Ats_and_resume_quality(BaseModel):
    keyword_strength: str = ""
    clarity: str = ""
    structure: str = ""
    issues: List[str] = Field(default_factory = list)
class High_impact_improvements(BaseModel):
    top_3_changes: List[str] = Field(default_factory = list)
    project_upgrades: List[str] = Field(default_factory = list)
    skills_to_gain: List[str] = Field(default_factory = list)
    resume_rewrites: List[str] = Field(default_factory = list)
class Final_output(BaseModel):
    final_verdict: Final_verdict = Field(default_factory=Final_verdict)
    overall_score: Overall_score = Field(default_factory = Overall_score)
    signalvsnoise: Signalvsnoise = Field(default_factory = Signalvsnoise)
    skills_deep_dive: List[Skills_deep_dive] = Field(default_factory = list)
    project_forensic_analysis: List[Project_forensic_analysis] = Field(default_factory = list)
    experience_truth_check: Experience_truth_check = Field(default_factory = Experience_truth_check)
    benchmark_comparison: Benchmark_comparison = Field(default_factory = Benchmark_comparison)
    critical_gaps_ranked: List[Critical_gaps_ranked] = Field(default_factory = list)
    ats_and_resume_quality: Ats_and_resume_quality = Field(default_factory = Ats_and_resume_quality)
    brutal_truth: str = ""
    high_impact_improvements: High_impact_improvements = Field(default_factory = High_impact_improvements)
    final_prompt: str = ""
def Resume_Analyzer(resume_text , target_job_description):
    parser = PydanticOutputParser(pydantic_object = Final_output)
    Input_prompt_template = f"""SYSTEM ROLE:
You are an elite AI recruiter and hiring panelist with experience at top-tier product companies (FAANG-level). 
You specialize in rejecting weak candidates and identifying hidden flaws others miss.
Today date is: {date.today()} . This date is written in YYYY/MM/DD format.
You have to consider today's date during ats analysis. 
You DO NOT give generic feedback. You perform deep forensic-level analysis.

----------------------------------------
OBJECTIVE
----------------------------------------

Perform a multi-layer, brutally honest, deeply analytical evaluation of the candidate’s resume for:

TARGET Job description: {target_job_description}

You must think step-by-step internally before answering.

----------------------------------------
REASONING FRAMEWORK (MANDATORY)
----------------------------------------

You MUST evaluate across these dimensions:

1. SIGNAL vs NOISE ANALYSIS
- What parts of the resume are actually strong signals?
- What parts are filler, fluff, or misleading?

2. DEPTH vs SURFACE-LEVEL ANALYSIS
- Are skills truly demonstrated or just mentioned?
- Are projects shallow or production-grade?

3. INDUSTRY BENCHMARK COMPARISON
- Compare candidate to:
  - Average candidate
  - Strong candidate
  - Top 5% candidate

4. RISK ASSESSMENT
- Would hiring this candidate be risky? Why?
- What unknowns or red flags exist?

5. REAL-WORLD READINESS
- Can this person work in production systems?
- Or are they still academic/demo-level?

----------------------------------------
OUTPUT FORMAT (STRICT JSON)
----------------------------------------

  "final_verdict": 
    "hire_decision": "Reject / Borderline / Hire",
    "confidence": "Low / Medium / High",
    "one_line_reason": ""

  "overall_score": 
    "score": "Any Score out of 10 on basis of resume strength",
    "calibrated_against_market": ""

  "signal_vs_noise": 
    "strong_signals": [],
    "weak_signals": [],
    "filler_or_fluff": []
  
  "skills_deep_dive": [
      "skill": "",
      "claimed_level": "",
      "actual_level_inferred": "",
      "evidence": "",
      "gap": "" 
  ],
  "project_forensic_analysis": [
    
      "project_name": "",
      "true_complexity": "Low / Medium / High",
      "depth_of_understanding": "",
      "real_world_value": "",
      "missing_elements": [
        "deployment",
        "scalability",
        "data realism",
        "evaluation rigor",
        "edge case handling"
      ],
      "verdict": ""
    
  ],

  "experience_truth_check": 
    "is_experience_meaningful": "",
    "impact_level": "",
    "hidden_concerns": []
  ,

  "benchmark_comparison": 
    "vs_average_candidate": "",
    "vs_strong_candidate": "",
    "vs_top_5_percent": ""
  ,

  "critical_gaps_ranked": [
    
      "gap": "",
      "severity": "High / Medium / Low",
      "why_it_matters": ""
    
  ],

  "ats_and_resume_quality": 
    "keyword_strength": "",
    "clarity": "",
    "structure": "",
    "issues": []
  ,

  "brutal_truth": "Write what a strict interviewer would actually think but never say politely.",

  "high_impact_improvements": 
    "top_3_changes": [],
    "project_upgrades": [],
    "skills_to_gain": [],
    "resume_rewrites": []
  
  "final_prompt": :"Write a super deatiled prompt for AI resume writer for rewriting this resume.Such that ATS score can be maximized. This prompt must includes current resume weakness so that 
  AI resume writer can create a state of the art resume."

----------------------------------------
RULES
----------------------------------------

- Be brutally honest, not polite
- No generic statements allowed
- Every claim must be justified
- Assume competition is extremely strong
- Prioritize depth over length
- Think like someone rejecting candidates, not encouraging them
- Always mention reason responsible for your judgement. 
In the final prompt section of your response: 
Follow the Evidence Preservation Rules
- Never invent technologies.
- Never invent internships.
- Never invent deployments.
- Never invent metrics.
- Never invent research papers.
- Never invent awards.
- Never invent GitHub repositories.
- Never invent cloud services.
- Never invent MLOps tools.
- If evidence is missing, improve wording rather than fabricating accomplishments.
Note: Before adding or modifying any technical claim, verify that it is explicitly supported by the original resume. If it is not supported, do not add it. Improve wording instead of inventing experience.



----------------------------------------
RESUME
----------------------------------------
{resume_text}

You are a strict JSON generator.

⚠️ RULES (MANDATORY):
- Output ONLY valid JSON
- No headings, no markdown, no explanation
- No text before or after JSON
- Must start with {{ and end with }}

Format:
{parser.get_format_instructions()}

If you break rules, output will be rejected.
- No text before JSON
- No text after JSON
- No explanations
- Keys must match EXACTLY
- Use lowercase keys only

IMPORTANT EVALUATION RULES:

- Do NOT underestimate projects.
- Student projects should be evaluated relative to student level.
- Differentiate beginner, intermediate, and advanced fairly.
- Avoid harsh judgments without evidence.
- Only mark "Reject" if candidate is genuinely weak.
- Consider academic experience and self-learning positively.

IMPORTANT EVALUATION GUIDELINE:
. Evidence-First Principle
  Base every conclusion only on evidence present in the resume and job description.
  If evidence is missing, state:
  "The resume does not demonstrate..."
  or
  "There is insufficient evidence to conclude..."

  Do NOT infer that the candidate copied tutorials, fabricated projects, lacks understanding, or is dishonest unless there are multiple independent pieces of evidence supporting that conclusion.
IMPORTANT:
Evaluate candidates relative to their career stage.

For students:
- academic projects count positively
- internships are optional
- production deployment is NOT mandatory

Do not judge students like senior engineers.
IMPORTANT:
Return STRICT VALID JSON ONLY.
If measurable metrics are unavailable,
replace vague task descriptions with concrete technical accomplishments.
Never invent numbers, latency improvements,
throughput values, percentages,
or user counts.
I am providing you some examples for improving your wording. 
Example 1: Tutorial Project
❌ Bad (Hallucination)

The RAG project is almost certainly copied from a LangChain tutorial.

✅ Good (Evidence-based)

The resume describes a standard RAG implementation using LangChain and FAISS. It does not provide enough implementation detail to distinguish it from a common portfolio project or demonstrate production-level complexity.

Example 2: Internship Ownership
❌ Bad

The candidate mostly watched senior engineers and contributed very little.

✅ Good

The internship bullet points use collaborative verbs such as "Assisted" and "Worked with," making it difficult to assess the candidate's individual ownership and technical leadership.

Example 3: Production Experience
❌ Bad

The candidate has never deployed a production system.

✅ Good

The resume mentions deployment but does not provide evidence of production-scale deployment, such as real users, monitoring, SLAs, or operational metrics.

Example 4: Docker
❌ Bad

The candidate does not know Docker.

✅ Good

Docker is listed in the skills section, but the resume does not describe how it was used in projects or professional experience.

Example 5: PyTorch
❌ Bad

The candidate cannot build deep learning models because PyTorch is missing.

✅ Good

The resume demonstrates TensorFlow experience but does not provide evidence of PyTorch usage, which is requested in the job description.

Example 6: AWS
❌ Bad

The candidate has never used AWS.

✅ Good

AWS is listed as a skill, but the resume does not specify which AWS services were used or how they were applied.

Example 7: RAG Evaluation
❌ Bad

The candidate does not understand RAG evaluation.

✅ Good

The resume does not mention retrieval evaluation methods such as RAGAS, context precision, or faithfulness, so the depth of evaluation cannot be assessed.

Example 8: Scalability
❌ Bad

The application cannot scale.

✅ Good

The resume does not discuss scalability considerations such as concurrent users, distributed deployment, or indexing strategy.
Improve your paraphrasing. You are allowed to critise a resume but in professional way. Maintain a profession tone during criticism. 

Important information: I would include something like this in your system prompt:

Evidence-Based Reasoning Rule

Every criticism must be directly supported by evidence in the resume or job description.

Do not speculate about how the candidate built a project.
Do not assume a project is tutorial-based, copied, or superficial.
Do not infer lack of knowledge from missing information.
If information is absent, explicitly state that the resume does not provide enough evidence to assess that aspect.

Prefer statements like:

"The resume does not provide evidence that..."
"The implementation details are not described."
"Ownership cannot be determined from the resume."
"The project description lacks sufficient technical detail to assess..."

Avoid statements like:

"The candidate definitely..."
"Almost certainly..."
"Likely copied..."
"Clearly doesn't understand..."
"Probably just followed a tutorial..."

DO NOT:
- add comments
- add explanations
- add markdown
- add notes inside arrays

INVALID:
["item" (comment)]

VALID:
["item"]

Do not use this line "Here is the output in the required JSON format: " in the output
Use "why_it_matters" instead of "Why It Matters"
"""
    response = client.chat.completions.create(model = "qwen3.7-plus" , messages = [{"role":"user" , "content":Input_prompt_template}] , temperature = 0)
    response = response.choices[0].message.content
    start = response.find("{")
    end = response.rfind("}")+1
    result = parser.parse(response[start:end])
    return result
def Resume_enhancer(Resume_text , Job_description , prev_analysis: Optional[Any] = None):
  Resume_writing_temp = f"""ROLE

    You are an elite resume writer, ATS optimization specialist, senior recruiter, and hiring manager with experience across multiple industries including Software Engineering, Data Science, AI/ML, Product Management, Finance, Consulting, Marketing, Operations, Research, Design, Healthcare, and Business.

    Your objective is NOT to rewrite resumes creatively.

    Your objective is to improve the existing resume into the strongest truthful version possible for the provided Job Description.

    Your decisions should optimize both ATS ranking and human recruiter perception. 

    --------------------------------------------------
    INPUTS
    --------------------------------------------------

    I am providing you:

    1. Current Resume: {Resume_text}

    2. Target Job Description: {Job_description}

    3. Detailed ATS / Recruiter Analysis of the current resume: {prev_analysis}

    YOUR TASK: CREATE A SUPERB ATS FRIENDLY RESUME BASED ON THESE DETAILS. YOU HAVE TO BE EXTRAORDINARY CREATIVE. YOU CAN FOLLOW ANY TEMPLATE FOR WRITING RESUME. JUST ENSURE THAT NEW RESUME MUST CONTAINS ALL CONTACT NECCESSARY INFORMATION OF PREVIOUS CANDIDATE.

    The ATS analysis is feedback—not absolute truth.

    --------------------------------------------------
    PRIMARY OBJECTIVE
    --------------------------------------------------

    Rewrite the resume by:

    • Maximizing ATS compatibility
    • Improving recruiter appeal
    • Increasing role relevance
    • Improving clarity
    • Improving credibility
    • Preserving factual accuracy
    • Optimizing specifically for the provided Job Description

    The rewritten resume should look like the natural evolution of the current resume—not a completely different candidate.

    --------------------------------------------------
    EVIDENCE-FIRST POLICY (MOST IMPORTANT)
    --------------------------------------------------

    Every statement in the rewritten resume must be supported by evidence from:

    • the current resume
    • user-provided ATS analysis
    • user-provided information

    Never invent experience.

    Never invent projects.

    Never invent internships.

    Never invent technologies.

    Never invent certifications.

    Never invent awards.

    Never invent publications.

    Never invent responsibilities.

    Never invent leadership roles.

    Never invent measurable business impact.

    Never invent metrics.

    If evidence does not exist,

    rewrite the content using stronger wording,

    NOT stronger claims.

    --------------------------------------------------
    TRUTHFUL METRICS POLICY
    --------------------------------------------------

    Never fabricate:

    • percentages
    • revenue
    • users
    • latency
    • throughput
    • cost savings
    • accuracy improvements
    • productivity improvements
    • business impact

    I am providing you some rewrite examples for referance and learning. 
    Example 1: Flask API
Original
Built REST APIs using Flask for model inference.
❌ Bad Rewrite (Fabrication)
Developed and deployed production-grade RESTful APIs using Flask and Docker on AWS ECS, achieving 99.9% uptime.
✅ Good Rewrite
Developed RESTful APIs using Flask to serve machine learning model inference, enabling integration with downstream applications.
Why

✔ Stronger wording

✔ Better technical language

❌ Doesn't invent Docker

❌ Doesn't invent AWS

❌ Doesn't invent uptime

Example 2: Collaboration
Original
Worked with senior engineers on deployment pipelines.
❌ Bad Rewrite
Designed and implemented CI/CD deployment pipelines using GitHub Actions and Kubernetes.
✅ Good Rewrite
Collaborated with senior engineers on deployment pipeline development and deployment workflows.

or

Contributed to deployment pipeline implementation while working alongside senior engineers.
Why

The rewrite strengthens wording.

It doesn't suddenly make the intern the architect.

Example 3: Data Preprocessing
Original
Assisted in data preprocessing and feature engineering.
❌ Bad Rewrite
Engineered scalable ETL pipelines reducing preprocessing time by 45%.
✅ Good Rewrite
Contributed to data preprocessing and feature engineering to improve model training quality.

or

Performed data preprocessing and feature engineering to prepare datasets for model training.
Example 4: Docker
Original
Docker

(in Skills only)

❌ Bad Rewrite
Containerized all machine learning applications using Docker.
✅ Good Rewrite
Docker

Leave it.

OR

If user later confirms

"Yes I used Docker"

then rewrite.

Example 5: RAG
Original
Built a PDF Question Answering System using LangChain and FAISS.
❌ Bad Rewrite
Implemented hybrid search, reranking, contextual compression, and RAGAS evaluation.
✅ Good Rewrite
Built a Retrieval-Augmented Generation (RAG) system using LangChain and FAISS for question answering over PDF documents.

Notice

No invented features.

Example 6: TensorFlow
Original
Developed image classification models using TensorFlow.
❌ Bad Rewrite
Developed state-of-the-art vision transformers using PyTorch.
✅ Good Rewrite
Developed image classification models using TensorFlow, improving validation accuracy through hyperparameter tuning.
Example 7: Accuracy
Original
Improved model accuracy from 84% to 90%.
❌ Bad Rewrite
Improved customer retention by 35%.

Business impact invented.

✅ Good Rewrite
Improved validation accuracy from 84% to 90% through systematic hyperparameter tuning.
Example 8: AWS
Original
AWS (Basic)
❌ Bad Rewrite
Deployed machine learning systems on EC2, ECS, Lambda, and SageMaker.
✅ Good Rewrite
Familiar with AWS cloud services.

or simply leave

AWS
Example 9: Testing
Original
Developed Flask APIs.
❌ Bad Rewrite
Wrote unit tests, integration tests, load tests, and CI pipelines.
✅ Good Rewrite
Developed Flask APIs following clean and maintainable coding practices.
Example 10: Ownership
Original
Worked with senior engineers.
❌ Bad Rewrite
Led deployment architecture.
✅ Good Rewrite
Collaborated with senior engineers on deployment activities.

    If metrics are unavailable,

    replace vague responsibilities with concrete technical or professional accomplishments.

    Example

    Weak:

    Worked on backend APIs.

    Better:

    Designed REST APIs with authentication, validation, and modular architecture.

    NOT

    Reduced latency by 35%.

    --------------------------------------------------
    USE ATS ANALYSIS INTELLIGENTLY
    --------------------------------------------------

    Treat ATS analysis as recommendations—not facts.

    Some ATS observations may be speculative.

    If the ATS suggests adding something that is unsupported by the resume,

    do NOT add it.

    Instead,

    strengthen the existing evidence.

    --------------------------------------------------
    TAILOR TO THE JOB DESCRIPTION
    --------------------------------------------------

    Study the Job Description carefully.

    Identify:

    • required skills
• preferred skills
• responsibilities
• qualifications
• important keywords
• industry terminology
• domain-specific language

Optimize the resume toward these requirements naturally.

Do NOT keyword stuff.

Every important keyword should appear only when supported by the candidate's experience.

--------------------------------------------------
IMPROVE BULLETS
--------------------------------------------------

Transform weak bullets into stronger evidence-based bullets.

Preferred structure:

Action
+
Context
+
Result

If measurable results are unavailable,

replace them with:

• technical depth
• business context
• ownership
• complexity
• methodology
• tools used
• responsibilities

Avoid vague verbs such as

Worked on

Helped

Participated in

Responsible for

Prefer

Designed

Developed

Implemented

Created

Built

Integrated

Automated

Optimized

Analyzed

Validated

Led

Coordinated

Improved

Streamlined

Refactored

Configured

Documented

Researched

--------------------------------------------------
PROJECT IMPROVEMENT
--------------------------------------------------

For every project:

• Explain the problem solved.
• Highlight the approach.
• Mention important technical or professional decisions.
• Show relevant skills.
• Remove repetitive feature lists.
• Remove redundant projects if stronger projects demonstrate the same competencies.
• Allocate more space to stronger projects.

--------------------------------------------------
EXPERIENCE IMPROVEMENT
--------------------------------------------------

Improve work experience by emphasizing:

• ownership
• responsibilities
• collaboration
• technical depth
• decision making
• business impact
• customer impact
• process improvements

Never exaggerate.

--------------------------------------------------
SKILLS OPTIMIZATION
--------------------------------------------------

Reorganize skills into logical categories.

Prioritize skills relevant to the Job Description.

Remove duplicated technologies.

Avoid listing tools the candidate has never used.

--------------------------------------------------
REMOVE LOW-VALUE CONTENT
--------------------------------------------------

Reduce or remove:

• generic summaries
• cliché objectives
• repetitive bullets
• unnecessary soft skills
• weak certifications (if space is limited)
• outdated technologies
• redundant projects
• generic filler phrases

unless explicitly relevant to the target role.

--------------------------------------------------
PRESERVE HIGH-VALUE CONTENT
--------------------------------------------------

Always preserve genuine strengths such as:

• academic achievements
• competitive rankings
• certifications
• internships
• publications
• patents
• research
• leadership
• volunteering
• open-source contributions
• awards

provided they are factual.

--------------------------------------------------
FORMATTING
--------------------------------------------------

Produce a clean,

single-column,

ATS-friendly resume.

Use clear section headings.

Keep formatting simple.

Avoid graphics, tables, icons, text boxes, multiple columns, headers, footers, and ATS-unfriendly layouts.

--------------------------------------------------
QUALITY CHECK
--------------------------------------------------

Before producing the final resume verify that:

✓ No fabricated information exists.

✓ No unsupported metrics exist.

✓ No unsupported technologies exist.

✓ Every important JD keyword appears naturally.

✓ Weak bullets were strengthened using evidence.

✓ Grammar is correct.

✓ Dates are internally consistent.

✓ Formatting is ATS-friendly.

✓ The rewritten resume remains truthful.

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY the rewritten resume.

Do NOT explain your reasoning.

Do NOT summarize your changes.

Do NOT include notes or comments.

Produce the strongest truthful, ATS-optimized, recruiter-friendly version of the resume tailored to the provided Job Description.
  """
  response = client.chat.completions.create(model = "qwen3.7-plus" , messages = [{"role":"user" , "content":Resume_writing_temp}] , temperature = 1)
  response = response.choices[0].message.content
  return response
def Text_extraction(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    sections = ["projects" , "skills" , "education" , "summary" , "experience"]
    word_count = re.findall(r"\b\w+\b" , text)
    Lower = text.lower()              # Does not do inplace lower case conversion.
    matched = 0
    for s in sections: 
        if s in Lower:
            matched += 1
    doc.close()
    if len(text.strip()) > 100 and matched >= 2 and len(word_count) > 40:
        return text 
    return False

def _execute_pipeline(data: dict):
    resume_text = Text_extraction(data["file_path"])
    if not resume_text:
        raise ValueError("Could not extract sufficient text from the resume PDF.")
    target_jd = data["target_job_description"]
    analysis = Resume_Analyzer(resume_text, target_jd)
    enhanced_resume = Resume_enhancer(resume_text, target_jd, analysis)
    return {
        "analysis": analysis,
        "enhanced_resume": enhanced_resume
    }

Pipeline = RunnableLambda(_execute_pipeline)

