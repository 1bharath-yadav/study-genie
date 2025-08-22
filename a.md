ok you understand my project very well again we dont store user uploaded documents in database (we only use numpyz for storing uploaded files of studets and after its usse we delete them).now give me the backend structure creatable linux mkdir ..commands from rooot dir and explain what is the functioonality of each file and folder.make sure wwe are building asynchronousable and cacheble and all modern efficent rag and data management project named as study genie.you got me wrong, dont mess with llm things like data embedding storing and chunksstoring, we never store user doccuemts permenently.in lanchain orchastrater we just take user question +uploaded file then chunking and embedding storingin numpyz(as user docs is not very big)and retriving and give chunks to llm and get structured respone.  this is not problem at all the prblem how we store student each subject and how we update and add new subjects autometically,how to track progress(on which based and what to store)how to store user weakness and that concept and overall personalized suggestions....and lot more tracking for student effective learning.i have one confusion about how to store user data like data schema which track student learning and progress and all user data. for example if user uploads chemistry pdf and workout this subject whole data,next day he may ask about any subject,so my idea is when giving top chinks to llm while extracting structured format we willask for subject name in single word ,if that word is not in our database we will create new column if already exist we add the details to that column.  i am confusing about all these database schema automation and storing stdent  analytics for student progress.give me effective plan,process and which modern database is best suitable.
I am building rag model in enhances self and personalized learning,i have to take student notes in pdfs,images,and  common study data format (extract the data if text is not selextable by any other functions) .2.we will store whole data and perform chunking and then embedding (by  numpysavez) and store embeddings in numpyz.  if user is asked extraquestions we also take account that. at first we will generate embeddings for user prompt and then grep the chunks by hybrid fetch(keyword+vector based)  then give those fetched chunks and around chunks to llm. and by llm extractionfor structured output and json schema(example   {{flashcards:{flashcardname:fashcard data,flashcardname:flashcard data...........................................},quiz{Q1:  question1,   options,correct answer, Q2:question2,options, correct answer...}..other structred output schema}} .now i will givve that structured output to frontend dev to showcas as flashcards....etc.   now tell me which techstack is best for my project and give me effective suggestions in functionalitiess,components.........etc.




Key Components for a StudyGenie Frontend
1. File Upload & Extraction

    Upload section (PDF, image, text file, handwritten image)

    Progress indicator/loader for extraction/OCR

    Editable text preview after extraction

2. Personalized Dashboard

    Welcome widget (student greeting, brief stats)

    Progress bars for each subject/topic

    Study streak tracker (calendar/timeline visualization)

    Knowledge “heatmap” showing strong vs. weak areas (D3.js or Chart.js)

3. Personalized Study Plan

    Interactive roadmap/flowchart: shows suggested study modules based on performance

    “Next up” widget for recommended activities (flashcards, quizzes, review material)

    Goal setting and reminders

4. Study Material Generator

    Dynamic section to display:

        Summaries

        Flashcards (flip card UI)

        Quizzes (MCQs, True/False, Fill-in-the-blank)

    Option to switch language (dropdown for English, Hindi, Marathi, etc.)

5. Active Recall & Spaced Repetition System

    Flashcard viewer (flip, mark as “easy,” “hard,” “needs review”)

    Scheduler to recommend when to review each flashcard set

6. Interactive Tutor (Chatbot/QA Panel)

    Chat interface for RAG-powered AI tutor

        Direct questions about uploaded materials

    Context window to show “where in notes” the answer comes from

7. Error Review/Explanation Module

    “Wrong answer review” pop-up or modal after quizzes

    Simple AI explanation section for each incorrect answer

8. Multi-Language Switcher

    Language selector (toggle, dropdown, or icon-based), affecting the entire interface and generated content

9. User Profile and Settings

    Profile page (avatar, basic info, language/goal settings)

    Option to export progress, sync with cloud, etc.

10. Notifications & Reminders

    Banner or bell icon for progress, review, and encouragement notifications

Bonus UX Ideas

    Accessibility tools: Font size adjuster, high-contrast mode, TTS (text-to-speech)

    Mobile responsive layout: Works on phone, tablet, desktop

    Gamification: XP bars, badges, leaderboards (optional)

Example Layout Flow

    Sidebar/menu: Navigation (Dashboard, Study Plan, Flashcards, Quizzes, Tutor, Settings)

    Main panel: Currently active module (dashboard, quiz, flashcards, chat, etc.)

    user profile

You can use React with component libraries like Material UI, tailwind css, or custom styled-components for a modern and scalable implementation. Focusing on these components ensures your frontend is user-friendly, informative, and highly engaging for personalized study workflows.