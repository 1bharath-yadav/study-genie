#!/usr/bin/env python3
"""
Comprehensive endpoint testing script for StudyGenie API
Tests all endpoints with the perceptron.pdf file
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"
PDF_FILE_PATH = "/home/archer/projects/study-genie/tests/perceptron.pdf"


class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_health_check(self):
        """Test the health check endpoint"""
        print("\n🔍 Testing Health Check...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print(f"✅ Health Check: {response.status} - {data}")
                return response.status == 200
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return False

    async def test_supported_file_types(self):
        """Test supported file types endpoint"""
        print("\n🔍 Testing Supported File Types...")
        try:
            async with self.session.get(f"{self.base_url}/api/files/supported-types") as response:
                data = await response.json()
                print(f"✅ Supported File Types: {response.status}")
                print(json.dumps(data, indent=2))
                return response.status == 200
        except Exception as e:
            print(f"❌ Supported File Types Failed: {e}")
            return False

    async def test_extract_text_only(self):
        """Test text extraction endpoint without LLM processing"""
        print("\n🔍 Testing Text Extraction Only...")
        try:
            if not Path(PDF_FILE_PATH).exists():
                print(f"❌ PDF file not found: {PDF_FILE_PATH}")
                return False

            with open(PDF_FILE_PATH, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='perceptron.pdf',
                               content_type='application/pdf')

                async with self.session.post(f"{self.base_url}/api/files/extract-text", data=data) as response:
                    result = await response.json()
                    print(f"✅ Text Extraction: {response.status}")
                    print(f"📄 Filename: {result.get('filename')}")
                    print(
                        f"📝 Text Length: {result.get('text_length')} characters")
                    print(f"📊 Word Count: {result.get('word_count')} words")
                    if result.get('extracted_text'):
                        preview = result['extracted_text'][:200] + "..." if len(
                            result['extracted_text']) > 200 else result['extracted_text']
                        print(f"📖 Text Preview: {preview}")
                    return response.status == 200
        except Exception as e:
            print(f"❌ Text Extraction Failed: {e}")
            return False

    async def test_create_student(self):
        """Test student creation endpoint"""
        print("\n🔍 Testing Student Creation...")
        try:
            student_data = {
                "username": "test_student",
                "email": "test@example.com",
                "full_name": "Test Student"
            }

            async with self.session.post(f"{self.base_url}/api/students", json=student_data) as response:
                result = await response.json()
                print(f"✅ Student Creation: {response.status}")
                print(json.dumps(result, indent=2))
                if response.status == 200:
                    return result.get('student_id')
                return None
        except Exception as e:
            print(f"❌ Student Creation Failed: {e}")
            return None

    async def test_file_upload_with_processing(self, student_id: int):
        """Test file upload with LLM processing"""
        print("\n🔍 Testing File Upload with Processing...")
        try:
            if not Path(PDF_FILE_PATH).exists():
                print(f"❌ PDF file not found: {PDF_FILE_PATH}")
                return False

            with open(PDF_FILE_PATH, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='perceptron.pdf',
                               content_type='application/pdf')
                data.add_field('student_id', str(student_id))
                data.add_field('subject_name', 'Machine Learning')
                data.add_field('chapter_name', 'Perceptron Algorithm')

                async with self.session.post(f"{self.base_url}/api/files/upload", data=data) as response:
                    result = await response.json()
                    print(f"✅ File Upload with Processing: {response.status}")
                    print(f"📄 Filename: {result.get('filename')}")
                    print(
                        f"📝 Text Length: {result.get('text_length')} characters")
                    print(f"📊 Word Count: {result.get('word_count')} words")
                    if result.get('processed_response'):
                        print("🧠 LLM Processing: Successfully processed")
                        print(
                            f"📊 Tracking Metadata: {result['processed_response'].get('tracking_metadata')}")
                    return response.status == 200
        except Exception as e:
            print(f"❌ File Upload with Processing Failed: {e}")
            return False

    async def test_student_progress(self, student_id: int):
        """Test getting student progress"""
        print("\n🔍 Testing Student Progress...")
        try:
            async with self.session.get(f"{self.base_url}/api/students/{student_id}/progress") as response:
                result = await response.json()
                print(f"✅ Student Progress: {response.status}")
                print(json.dumps(result, indent=2))
                return response.status == 200
        except Exception as e:
            print(f"❌ Student Progress Failed: {e}")
            return False

    async def test_llm_processing(self, student_id: int):
        """Test direct LLM processing endpoint"""
        print("\n🔍 Testing LLM Processing...")
        try:
            llm_request = {
                "student_id": student_id,
                "subject_name": "Machine Learning",
                "chapter_name": "Neural Networks",
                "concept_name": "Perceptron",
                "llm_response": {
                    "summary": "The perceptron is a fundamental building block of neural networks.",
                    "learning_objectives": ["Understand perceptron algorithm", "Learn weight updates"]
                },
                "user_query": "Explain the perceptron algorithm"
            }

            async with self.session.post(f"{self.base_url}/api/process-llm-response", json=llm_request) as response:
                result = await response.json()
                print(f"✅ LLM Processing: {response.status}")
                print(json.dumps(result, indent=2))
                return response.status == 200
        except Exception as e:
            print(f"❌ LLM Processing Failed: {e}")
            return False

    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting comprehensive API testing with perceptron.pdf...")

        results = []

        # Test basic health and file types
        results.append(("Health Check", await self.test_health_check()))
        results.append(("Supported File Types", await self.test_supported_file_types()))
        results.append(("Text Extraction Only", await self.test_extract_text_only()))

        # Test student creation
        student_id = await self.test_create_student()
        if student_id:
            results.append(("Student Creation", True))

            # Test file upload with processing
            results.append(("File Upload with Processing", await self.test_file_upload_with_processing(student_id)))

            # Test student progress
            results.append(("Student Progress", await self.test_student_progress(student_id)))

            # Test LLM processing
            results.append(("LLM Processing", await self.test_llm_processing(student_id)))
        else:
            results.append(("Student Creation", False))

        # Print summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)

        passed = 0
        total = len(results)

        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1

        print("="*60)
        print(
            f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 All tests passed! API is working correctly!")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")

        return passed == total


async def main():
    """Main test runner"""
    # Check if PDF file exists
    if not Path(PDF_FILE_PATH).exists():
        print(f"❌ Error: PDF file not found at {PDF_FILE_PATH}")
        print("Please make sure the perceptron.pdf file exists in the tests directory.")
        sys.exit(1)

    async with APITester(BASE_URL) as tester:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
