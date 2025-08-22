from fastapi import APIRouter, HTTPException, Request, Response
from app.payload_handler import handle_payload_and_response


router = APIRouter()


@router.post("/analyze_data", tags=["Analysis"])
async def analyze_data(request: Request):
    form = await request.form()

    questions_file = None
    attachments = []

    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
            if field_name == "questions.txt" or getattr(field_value, 'filename', '') == "questions.txt":
                questions_file = field_value
            else:
                attachments.append(field_value)

    try:
        content = await questions_file.read()
        if isinstance(content, bytes):
            input_text = content.decode("utf-8")
        else:
            input_text = str(content)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Invalid questions.txt file encoding")

    payload = {"user_prompt": input_text, "attachments": attachments}

    result = await handle_payload_and_response(payload)

    return result
