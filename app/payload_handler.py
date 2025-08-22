import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from app.file_manager import create_temp_dir, save_multiple_files, cleanup_temp_dir
from app.llm.langchain import get_llm_response


async def handle_payload_and_response(payload: Dict[str, Any]) -> Any:

    # Accept either `userprompt` or `user_prompt` (route historically used user_prompt)
    userprompt = payload.get("userprompt") or payload.get("user_prompt")
    if not userprompt:
        return {"status": "error", "error": "Missing required field: 'userprompt' or 'user_prompt'"}

    attachments = payload.get("attachments", [])

    # Centralized temp directory creation - single source of truth
    tmp_dir = await create_temp_dir(userprompt)

    # Step 1: Handle file uploads if any
    uploaded_files_paths: List[Path] = []
    if attachments:
        uploaded_files_paths = await save_multiple_files(attachments, tmp_dir)
    else:
        pass

    # If running tests locally or user wants to skip external LLM calls, return a mock response
    if os.getenv("SKIP_LLM") or os.getenv("TEST_MODE"):
        # Build a lightweight mock structured response so the API flow can be tested offline
        files_info = [str(p.name) for p in uploaded_files_paths]
        mock_response = {
            "status": "ok",
            "source": "mock",
            "userprompt": userprompt,
            "files": files_info,
            "flashcards": {
                "card1": {
                    "question": "What is a perceptron?",
                    "answer": "A basic unit of a neural network that computes a weighted sum and applies an activation.",
                    "difficulty": "Easy"
                }
            },
            "quiz": {
                "Q1": {
                    "question": "Perceptron is used for which type of task?",
                    "options": ["Classification", "Sorting", "Regression"],
                    "correct_answer": "Classification",
                    "explanation": "Perceptrons are binary classifiers that separate data with a linear boundary."
                }
            },
            "summary": "Mock summary generated in TEST_MODE. Replace with LLM output when SKIP_LLM is unset.",
            "learning_objectives": [
                "Understand the perceptron model",
                "Explain how a perceptron makes decisions"
            ]
        }

        return mock_response

    final_answer = await get_llm_response(
        uploaded_files_paths=uploaded_files_paths,
        userprompt=userprompt,
        temp_dir=str(tmp_dir)
    )

    # await cleanup_temp_dir(tmp_dir)

    return final_answer
