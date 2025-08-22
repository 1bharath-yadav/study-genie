"""Problem Statement : StudyGenie - Personalized Study Guide Generator

Description:
Turn raw learning input (books, PDFs, handwritten notes) into quizzes, flashcards, and

interactive study material. Use RAG/GenAl to personalize learning, add multilingual support, and

track student progress.
Expected Outcomes (Detailed):
* Upload PDF/notes — App extracts text (OCR if needed) - Summaries + quizzes auto-
generated.
* Personalized study flow based on weals/strong subjects logged over time.
*  Flashcard system with active recall and spaced repetition built-in.

* Interactive RAG-powered Al tutor that answers student questions directly from provided

materials.

* Multilingual support (English, Hindi, Marathi, regional). Example: Convert a math
summary into Marathi for regional students.

+ Dashboard with progress bars, study strealts, knowledge heatmaps.

* Example Demo Flow: Student uploads Physics Chapter PDF -» app outputs 15 flashcards +
10 MCQs — student takes quiz - dashboard updates progress — Al tutor explains wrong

answers in simple words.

Technology: LangChain, OpenAl API, Hugging Face Transformers, Tesseract OCR / Google Vision
AP, React, Nodes/Python, Pinecone/FAISS, Firebase/PostgresQL, D3.s/Chartjs
"""


from app.core import app, get_db_manager
from app.api.v1.routes import router

# Re-export app router
app.include_router(router, prefix="/api")

# Health endpoint preserved at top-level for compatibility


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Progress tracker API is running"}

# All API route handlers are now defined in modular files under app/api/v1
# The central router is included above and serves those routes.

# All routes are modularized under app/api/v1 and mounted above.
