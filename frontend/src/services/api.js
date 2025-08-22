import axios from 'axios';

// Configure axios with your backend base URL
const API_BASE_URL = 'http://127.0.0.1:8000'; // Use IPv4 explicitly instead of localhost

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000, // 30 seconds timeout
    // Force IPv4
    family: 4,
});

// Request interceptor for logging
api.interceptors.request.use(
    (config) => {
        console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
        return config;
    },
    (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => {
        console.log('API Response successful:', response.status, response.config.url);
        return response;
    },
    (error) => {
        console.error('=== API INTERCEPTOR ERROR ===');
        console.error('Error:', error);
        console.error('Error code:', error.code);
        console.error('Error message:', error.message);
        console.error('Error response:', error.response);
        console.error('Error request:', error.request);
        console.error('Error config:', error.config);
        console.error('=============================');

        // Improve error object for better debugging
        if (error.response?.data) {
            console.error('Response data:', JSON.stringify(error.response.data, null, 2));
        }

        return Promise.reject(error);
    }
);

export const apiService = {
    // Health Check
    checkHealth: async () => {
        try {
            const response = await api.get('/health');
            return response.data;
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    },

    // Student Management
    createStudent: async (studentData) => {
        try {
            console.log('Creating student with data:', studentData);
            console.log('API Base URL:', API_BASE_URL);
            console.log('Full URL will be:', `${API_BASE_URL}/api/students`);

            const response = await api.post('/api/students', studentData);
            console.log('Student creation response:', response.data);
            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);
            return response.data;
        } catch (error) {
            console.error('=== STUDENT CREATION ERROR ===');
            console.error('Error object:', error);
            console.error('Error message:', error.message);
            console.error('Error response:', error.response);
            console.error('Error response data:', error.response?.data);
            console.error('Error response status:', error.response?.status);
            console.error('Error response headers:', error.response?.headers);
            console.error('===============================');

            // Better error message parsing
            let errorMessage = 'Failed to create student';

            if (error.response) {
                // Server responded with error status
                const data = error.response.data;
                if (typeof data === 'string') {
                    errorMessage += `: ${data}`;
                } else if (data && typeof data === 'object') {
                    if (data.detail) {
                        errorMessage += `: ${data.detail}`;
                    } else if (data.message) {
                        errorMessage += `: ${data.message}`;
                    } else {
                        errorMessage += `: ${JSON.stringify(data)}`;
                    }
                }
            } else if (error.request) {
                // Request was made but no response received
                errorMessage += ': No response from server. Check if backend is running.';
            } else {
                // Something else happened
                errorMessage += `: ${error.message}`;
            }

            throw new Error(errorMessage);
        }
    },

    updateStudentProgress: async (studentId, progressData) => {
        try {
            const response = await api.post(`/api/students/${studentId}/progress`, progressData);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update progress: ${error.response?.data?.detail || error.message}`);
        }
    },

    getStudentProgress: async (studentId, subjectId = null) => {
        try {
            const params = subjectId ? { subject_id: subjectId } : {};
            const response = await api.get(`/api/students/${studentId}/progress`, { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get student progress: ${error.response?.data?.detail || error.message}`);
        }
    },

    // LLM Processing
    processLLMResponse: async (requestData) => {
        try {
            console.log('Sending LLM request:', requestData);
            const response = await api.post('/api/process-llm-response', requestData);
            return response.data;
        } catch (error) {
            console.error('LLM processing error:', error.response?.data);
            throw new Error(`Failed to process LLM response: ${error.response?.data?.detail || error.message}`);
        }
    },

    // File Upload Mock (since your backend doesn't have file upload endpoint yet)
    processFileContent: async (fileContent, studentId, metadata) => {
        // This would typically be a file upload endpoint
        // For now, we'll simulate processing the extracted text
        const mockLLMResponse = {
            flashcards: {
                card1: {
                    question: "What is the main concept discussed in the uploaded content?",
                    answer: "Based on the uploaded content analysis",
                    key_concepts: "Core concept identification",
                    key_concepts_data: "Detailed analysis of the main topics",
                    difficulty: "Medium"
                },
                // Add more cards based on content...
            },
            quiz: {
                Q1: {
                    question: "Which statement best describes the content?",
                    options: ["Option A", "Option B", "Option C", "Option D"],
                    correct_answer: "Option A",
                    explanation: "This is correct because..."
                },
                // Add more questions...
            },
            match_the_following: {
                columnA: ["Term 1", "Term 2", "Term 3"],
                columnB: ["Definition 1", "Definition 2", "Definition 3"],
                mappings: [
                    { A: "Term 1", B: "Definition 1" },
                    { A: "Term 2", B: "Definition 2" },
                    { A: "Term 3", B: "Definition 3" }
                ]
            },
            summary: "This is a comprehensive summary of the uploaded content covering key concepts and learning objectives.",
            learning_objectives: [
                "Understand the main concepts presented",
                "Apply knowledge in practical scenarios",
                "Analyze and synthesize information"
            ]
        };

        const requestData = {
            student_id: studentId,
            subject_name: metadata.subject || "General Studies",
            chapter_name: metadata.chapter || "Chapter 1",
            concept_name: metadata.concept || "Core Concepts",
            llm_response: mockLLMResponse,
            user_query: `Process uploaded content: ${fileContent.substring(0, 100)}...`,
            difficulty_level: metadata.difficulty || "Medium"
        };

        return await this.processLLMResponse(requestData);
    },

    // Utility function to extract text from different file types
    extractTextFromFile: async (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                const content = event.target.result;

                if (file.type === 'text/plain') {
                    resolve(content);
                } else if (file.type === 'application/pdf') {
                    // For PDF, you'd typically use a library like pdf.js
                    // For now, return a placeholder
                    resolve("PDF content extraction would require additional libraries like pdf.js");
                } else if (file.type.startsWith('image/')) {
                    // For images, you'd typically use OCR
                    // For now, return a placeholder
                    resolve("Image OCR processing would require integration with your backend OCR service");
                } else {
                    resolve(content);
                }
            };

            reader.onerror = () => reject(new Error('Failed to read file'));

            if (file.type.startsWith('image/')) {
                reader.readAsDataURL(file);
            } else {
                reader.readAsText(file);
            }
        });
    }
};

export default apiService;
