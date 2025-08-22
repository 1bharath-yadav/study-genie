import os
import shutil
import tempfile
from typing import List
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.llm.langchain import get_llm_response

router = APIRouter()

@router.post("/generate-study-material")
async def generate_study_material(
    files: List[UploadFile] = File(...),
    user_prompt: str = Form(...),
):
    """
    Receives files and a user prompt, processes them using the RAG pipeline,
    and returns structured learning content.
    """
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)

        # Call the RAG pipeline
        response = await get_llm_response(
            uploaded_files_paths=[Path(fp) for fp in file_paths],
            userprompt=user_prompt,
            temp_dir=temp_dir,
        )

        if response.get("status") == "error":
            raise HTTPException(status_code=500, detail=response.get("error"))

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
