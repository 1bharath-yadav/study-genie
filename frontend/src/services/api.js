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

    generateStudyMaterial: async (files, userPrompt) => {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('user_prompt', userPrompt);

        try {
            const response = await api.post('/api/generate-study-material', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to generate study material: ${error.response?.data?.detail || error.message}`);
        }
    },
};

export default apiService;
