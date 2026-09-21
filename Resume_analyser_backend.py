from Resume_Analyzer_Genai_APP import Pipeline
from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import os 
app = FastAPI()
@app.post("/uploadResume")
async def uploadResume(
    file: UploadFile = File(...),
    target_job_description: str = Form(...)
):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(
        delete = False , suffix = suffix 
    ) as temp_file:
        content = await file.read()
        temp_file.write(content)
        file_path = temp_file.name 
    try: 
            result = Pipeline.invoke({"file_path": file_path, "target_job_description": target_job_description})
            return {
                "filename": file.filename, "result": result 
            }
    finally: 
            os.remove(file_path)
