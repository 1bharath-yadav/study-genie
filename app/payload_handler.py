from llm.langchain import get_llm_response  # replace with the actual import
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# Import your existing LLM handler


@app.post("/process")
async def handle_payload_and_response(
    userprompt: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Create a temporary directory for this request
    tmp_dir = tempfile.mkdtemp(prefix="session_")
    uploaded_files_paths = []

    try:
        # Save uploaded files to the temporary directory
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400, detail="Uploaded file missing filename.")
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            uploaded_files_paths.append(file_path)

        # Call your LLM function
        final_answer = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=userprompt,
            temp_dir=str(tmp_dir)
        )

        return JSONResponse(content={"answer": final_answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files to avoid storage bloat
        shutil.rmtree(tmp_dir, ignore_errors=True)
