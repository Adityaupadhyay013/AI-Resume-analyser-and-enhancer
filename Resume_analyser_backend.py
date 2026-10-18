from Resume_Analyzer_Genai_APP import Pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
import pymupdf
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows requests from your Lovable frontend and localhost
    allow_credentials=False,   # Keep False when allow_origins is "*"
    allow_methods=["*"],       # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # Allows all headers (Content-Type, etc.)
)
@app.post("/uploadResume")
async def uploadResume(
    file: UploadFile = File(...),
    target_job_description: str = Form(...)
):
    pdf_bytes = await file.read()
    try: 
            result = Pipeline.invoke({"pdf_bytes": pdf_bytes, "target_job_description": target_job_description})
            return {
                "filename": file.filename, "result": result 
            }
    except (ValueError , pymupdf.FileDataError) as E:
        raise HTTPException(422, str(E))
        
