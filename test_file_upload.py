#!/usr/bin/env python3
"""
Test script to verify file upload and LLM integration
"""

from app.api.v1.files import extract_text_from_file
from app.llm.langchain import get_llm_response
import asyncio
import tempfile
import os
from pathlib import Path

# Set up path
import sys
sys.path.append('.')


async def test_file_processing():
    """Test the complete file upload and LLM processing workflow"""

    # Create a test text file
    test_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn and make decisions from data without being explicitly programmed.
    
    Key concepts include:
    - Supervised Learning: Learning with labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data  
    - Reinforcement Learning: Learning through interaction and rewards
    
    Common algorithms:
    1. Linear Regression
    2. Decision Trees
    3. Neural Networks
    4. Support Vector Machines
    """

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = Path(f.name)

    try:
        print("🔍 Testing text extraction...")
        extracted_text = await extract_text_from_file(temp_file_path)
        print(f"✅ Extracted {len(extracted_text)} characters")
        print(f"Preview: {extracted_text[:100]}...")

        print("\n🧠 Testing LLM response generation...")
        temp_dir = str(temp_file_path.parent)

        # Test LLM response
        llm_response = await get_llm_response(
            uploaded_files_paths=[temp_file_path],
            userprompt="Create study materials for this machine learning content",
            temp_dir=temp_dir
        )

        print("✅ LLM Response generated successfully!")

        # Check if response has expected structure
        if llm_response and isinstance(llm_response, dict):
            print(f"📊 Response keys: {list(llm_response.keys())}")

            if 'flashcards' in llm_response:
                flashcard_count = len(llm_response['flashcards'])
                print(f"📚 Generated {flashcard_count} flashcards")

            if 'quiz' in llm_response:
                quiz_count = len(llm_response['quiz'])
                print(f"❓ Generated {quiz_count} quiz questions")

            if 'summary' in llm_response:
                summary_length = len(llm_response['summary'])
                print(f"📝 Generated summary ({summary_length} characters)")

        print("\n🎉 File upload and LLM integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if temp_file_path.exists():
            os.unlink(temp_file_path)

if __name__ == "__main__":
    print("🚀 Starting file upload and LLM integration test...\n")
    success = asyncio.run(test_file_processing())

    if success:
        print("\n✅ All tests passed! The file upload → LLM response pipeline is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
