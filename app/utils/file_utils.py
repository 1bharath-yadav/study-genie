import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

async def save_uploaded_files_to_temp(files: List[UploadFile]) -> (str, List[Path]):
    """
    Saves uploaded files to a temporary directory and returns the temporary directory path
    and a list of paths to the saved files.
    """
    tmp_dir = None
    uploaded_files_paths = []

    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")

        tmp_dir = tempfile.mkdtemp(prefix="study_session_")
        logger.info(f"Created temporary directory: {tmp_dir}")

        for i, file in enumerate(files):
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1} is missing a filename"
                )

            allowed_extensions = {'.pdf', '.txt', '.doc', '.docx', '.jpg', '.jpeg', '.png'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
                )

            file_path = Path(tmp_dir) / file.filename
            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    if not content:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File {file.filename} is empty"
                        )
                    buffer.write(content)
                uploaded_files_paths.append(file_path)
                logger.info(f"Saved file: {file.filename} ({len(content)} bytes)")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error saving file {file.filename}: {str(e)}"
                )
        return tmp_dir, uploaded_files_paths
    except HTTPException:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"Unexpected error in save_uploaded_files_to_temp: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during file upload: {str(e)}")

def cleanup_temp_dir(tmp_dir: str):
    """Cleans up the temporary directory."""
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {tmp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")
