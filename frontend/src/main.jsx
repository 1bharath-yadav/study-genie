import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Upload,
    User,
    BookOpen,
    Brain,
    Target,
    Home,
    BarChart3,
    Menu,
    LogOut,
    Bell
} from 'lucide-react';

// Import components
import LoginForm from './components/LoginForm';
import FileUpload from './components/FileUpload';
import FlashCardDeck from './components/FlashCardDeck';
import Quiz from './components/Quiz';
import MatchTheFollowing from './components/MatchTheFollowing';
import ProgressDashboard from './components/ProgressDashboard';
import Loading from './components/Loading';
import ChatBox from './components/ChatBox';
import StudyContentSections from './components/StudyContentSections';

// Import services and utilities
import { apiService } from './services/api';
import { storage } from './utils';

// Import CSS
import './index.css';

const StudyGenie = () => {
    // App state
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('main');
    const [sidebarOpen, setSidebarOpen] = useState(false);

    // Study content state
    const [studyContent, setStudyContent] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progressData, setProgressData] = useState(null);

    // Chat and content state
    const [userPrompt, setUserPrompt] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [expandedSection, setExpandedSection] = useState(null);
    const [showDashboard, setShowDashboard] = useState(false);

    // Check for existing user session on load
    useEffect(() => {
        const checkSession = async () => {
            try {
                const savedUser = storage.get('currentUser');
                if (savedUser) {
                    setCurrentUser(savedUser);
                    setIsAuthenticated(true);
                    await loadUserProgress(savedUser.student_id);
                }
            } catch (error) {
                console.error('Session check error:', error);
            } finally {
                setIsLoading(false);
            }
        };

        checkSession();
    }, []);

    // API health check
    useEffect(() => {
        const healthCheck = async () => {
            try {
                await apiService.checkHealth();
                console.log('Backend API is healthy');
            } catch (error) {
                console.warn('Backend API is not available:', error.message);
                toast.error('Backend API is not available. Using demo mode.');
            }
        };

        if (isAuthenticated) {
            healthCheck();
        }
    }, [isAuthenticated]);

    const handleLogin = async (formData) => {
        try {
            setIsLoading(true);
            const response = await apiService.createStudent(formData);

            const user = {
                student_id: response.student_id,
                username: response.username,
                email: response.email,
                full_name: response.full_name
            };

            setCurrentUser(user);
            setIsAuthenticated(true);
            storage.set('currentUser', user);

            // Load initial progress data
            await loadUserProgress(user.student_id);

            return response;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogout = () => {
        setIsAuthenticated(false);
        setCurrentUser(null);
        setStudyContent(null);
        setProgressData(null);
        storage.remove('currentUser');
        toast.success('Logged out successfully');
    };

    const loadUserProgress = async (studentId) => {
        try {
            const progress = await apiService.getStudentProgress(studentId);
            setProgressData(progress);
        } catch (error) {
            console.error('Failed to load progress:', error);
            // Set mock progress data for demo
            setProgressData({
                student_id: studentId,
                total_concepts: 25,
                mastered_concepts: 8,
                weak_concepts: 3,
                concept_progress: [
                    {
                        concept_id: 1,
                        concept_name: 'Basic Algebra',
                        status: 'mastered',
                        mastery_score: 95,
                        attempts_count: 5,
                        correct_answers: 19,
                        total_questions: 20
                    },
                    {
                        concept_id: 2,
                        concept_name: 'Quadratic Equations',
                        status: 'in_progress',
                        mastery_score: 75,
                        attempts_count: 3,
                        correct_answers: 15,
                        total_questions: 20
                    }
                ],
                subject_progress: [
                    { subject_name: 'Mathematics', progress_percentage: 75 },
                    { subject_name: 'Physics', progress_percentage: 60 },
                    { subject_name: 'Chemistry', progress_percentage: 85 }
                ],
                recent_activity: [
                    { activity_type: 'quiz_attempt', concept_name: 'Algebra Basics', timestamp: new Date(), score: 85 }
                ]
            });
        }
    };

    const handleFileProcessed = async (fileData) => {
        try {
            setIsProcessing(true);

            // Add to chat history
            setChatHistory(prev => [...prev, {
                type: 'file',
                content: `File uploaded: ${fileData.fileName}`,
                timestamp: new Date()
            }]);

            // Extract metadata from filename or use defaults
            const metadata = {
                subject: 'General Studies',
                chapter: 'Chapter 1',
                concept: fileData.fileName.replace(/\.[^/.]+$/, ''), // Remove extension
                difficulty: 'Medium'
            };

            // Process file content with backend
            const response = await apiService.processFileContent(
                fileData.content,
                currentUser.student_id,
                metadata
            );

            // Create study content from LLM response
            const content = response.enhanced_response || response.llm_response;

            // Convert backend response format to frontend format
            const studyContent = {
                flashcards: Object.entries(content.flashcards || {}).map(([key, card]) => ({
                    id: key,
                    question: card.question,
                    answer: card.answer,
                    key_concepts: card.key_concepts,
                    key_concepts_data: card.key_concepts_data,
                    difficulty: card.difficulty
                })),
                quiz: Object.entries(content.quiz || {}).map(([key, question]) => ({
                    id: key,
                    question: question.question,
                    options: question.options,
                    correct_answer: question.correct_answer,
                    explanation: question.explanation
                })),
                matchTheFollowing: content.match_the_following,
                summary: content.summary,
                learningObjectives: content.learning_objectives || []
            };

            setStudyContent(studyContent);

            // Add success message to chat
            setChatHistory(prev => [...prev, {
                type: 'success',
                content: 'Study content generated successfully! Check the sections on the right.',
                timestamp: new Date()
            }]);

            // Update progress
            await loadUserProgress(currentUser.student_id);

            toast.success('Study content generated successfully!');
        } catch (error) {
            console.error('File processing error:', error);
            setChatHistory(prev => [...prev, {
                type: 'error',
                content: 'Failed to process file. Please try again.',
                timestamp: new Date()
            }]);
            toast.error('Failed to process file. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    const handlePromptSubmit = async () => {
        if (!userPrompt.trim()) return;

        try {
            setIsProcessing(true);

            // Add user message to chat
            setChatHistory(prev => [...prev, {
                type: 'user',
                content: userPrompt,
                timestamp: new Date()
            }]);

            // Create mock LLM response based on prompt
            const mockLLMResponse = {
                flashcards: {
                    card1: {
                        question: `What is the main concept about: ${userPrompt}?`,
                        answer: "Based on your query, here's a comprehensive answer.",
                        key_concepts: "Core concept identification",
                        key_concepts_data: "Detailed analysis of the main topics",
                        difficulty: "Medium"
                    },
                    // Add more cards...
                },
                quiz: {
                    Q1: {
                        question: `Which statement best describes: ${userPrompt}?`,
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
                summary: `This is a comprehensive summary based on your query: ${userPrompt}`,
                learning_objectives: [
                    "Understand the main concepts presented",
                    "Apply knowledge in practical scenarios",
                    "Analyze and synthesize information"
                ]
            };

            const requestData = {
                student_id: currentUser.student_id,
                subject_name: "General Studies",
                chapter_name: "Chapter 1",
                concept_name: "User Query",
                llm_response: mockLLMResponse,
                user_query: userPrompt,
                difficulty_level: "Medium"
            };

            const response = await apiService.processLLMResponse(requestData);
            const content = response.enhanced_response || response.llm_response;

            // Convert backend response format to frontend format
            const studyContent = {
                flashcards: Object.entries(content.flashcards || {}).map(([key, card]) => ({
                    id: key,
                    question: card.question,
                    answer: card.answer,
                    key_concepts: card.key_concepts,
                    key_concepts_data: card.key_concepts_data,
                    difficulty: card.difficulty
                })),
                quiz: Object.entries(content.quiz || {}).map(([key, question]) => ({
                    id: key,
                    question: question.question,
                    options: question.options,
                    correct_answer: question.correct_answer,
                    explanation: question.explanation
                })),
                matchTheFollowing: content.match_the_following,
                summary: content.summary,
                learningObjectives: content.learning_objectives || []
            };

            setStudyContent(studyContent);

            // Add AI response to chat
            setChatHistory(prev => [...prev, {
                type: 'ai',
                content: 'Study content generated based on your query! Check the sections on the right.',
                timestamp: new Date()
            }]);

            setUserPrompt('');

            // Update progress
            await loadUserProgress(currentUser.student_id);

            toast.success('Study content generated successfully!');
        } catch (error) {
            console.error('Prompt processing error:', error);
            setChatHistory(prev => [...prev, {
                type: 'error',
                content: 'Failed to process your query. Please try again.',
                timestamp: new Date()
            }]);
            toast.error('Failed to process your query. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleActivityComplete = async (activityType, results) => {
        try {
            // Update progress in backend
            await apiService.updateStudentProgress(currentUser.student_id, {
                concept_id: 1, // Mock concept ID
                correct_answers: results.correctAnswers || results.correctMatches || 0,
                total_questions: results.totalCards || results.totalQuestions || results.totalItems || 0,
                activity_type: activityType
            });

            // Reload progress
            await loadUserProgress(currentUser.student_id);

            toast.success(`${activityType} completed! Progress updated.`);
        } catch (error) {
            console.error('Progress update error:', error);
            toast.error('Failed to update progress');
        }
    };

    // Navigation items
    const navigationItems = [
        { id: 'progress', label: 'Progress', icon: BarChart3 },
    ];

    const Sidebar = () => (
        <motion.div
            className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-xl transform transition-transform duration-300 lg:translate-x-0 lg:static lg:inset-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}
            initial={false}
            animate={{ x: sidebarOpen ? 0 : -256 }}
        >
            <div className="flex flex-col h-full">
                {/* Header */}
                <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <Brain className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-gray-800">Study Genie</h1>
                            <p className="text-sm text-gray-600">AI-Powered Learning</p>
                        </div>
                    </div>
                </div>

                {/* User Info */}
                <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                            <User className="w-4 h-4 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-800 truncate">
                                {currentUser?.full_name || currentUser?.username}
                            </p>
                            <p className="text-xs text-gray-500 truncate">{currentUser?.email}</p>
                        </div>
                    </div>
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4">
                    <ul className="space-y-2">
                        {navigationItems.map((item) => (
                            <li key={item.id}>
                                <button
                                    onClick={() => {
                                        if (item.id === 'progress') {
                                            setActiveTab(item.id);
                                        }
                                        setSidebarOpen(false);
                                    }}
                                    className={`
                    w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200
                    ${activeTab === item.id
                                            ? 'bg-blue-100 text-blue-700 font-medium'
                                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-800'
                                        }
                  `}
                                >
                                    <item.icon className="w-5 h-5" />
                                    <span>{item.label}</span>
                                </button>
                            </li>
                        ))}
                    </ul>
                </nav>

                {/* Logout */}
                <div className="p-4 border-t border-gray-200">
                    <button
                        onClick={handleLogout}
                        className="w-full flex items-center space-x-3 px-4 py-3 text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200"
                    >
                        <LogOut className="w-5 h-5" />
                        <span>Logout</span>
                    </button>
                </div>
            </div>
        </motion.div>
    );

    const MainContent = () => {
        return (
            <div className="flex-1 flex flex-col min-h-screen">
                {/* Header */}
                <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={() => setSidebarOpen(!sidebarOpen)}
                                className="lg:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
                            >
                                <Menu className="w-6 h-6 text-gray-600" />
                            </button>

                            <div className="flex items-center space-x-4">
                                <h1 className="text-xl font-semibold text-gray-800">
                                    Study Genie
                                </h1>
                                <button
                                    onClick={() => setShowDashboard(!showDashboard)}
                                    className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                                >
                                    <Home className="w-4 h-4 text-gray-600" />
                                    <span className="text-sm text-gray-600">Dashboard</span>
                                </button>
                            </div>
                        </div>

                        <div className="flex items-center space-x-4">
                            <div className="text-right">
                                <p className="text-sm font-medium text-gray-800">
                                    {currentUser?.full_name || currentUser?.username}
                                </p>
                                <p className="text-xs text-gray-500">{currentUser?.email}</p>
                            </div>
                            <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative">
                                <Bell className="w-5 h-5 text-gray-600" />
                                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
                            </button>
                        </div>
                    </div>
                </header>

                {/* Dashboard Dropdown */}
                <AnimatePresence>
                    {showDashboard && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white overflow-hidden"
                        >
                            <div className="p-6">
                                <h2 className="text-2xl font-bold mb-2">
                                    Welcome back, {currentUser?.full_name || currentUser?.username}!
                                </h2>
                                <p className="text-blue-100 mb-4">Ready to continue your learning journey?</p>

                                {progressData && (
                                    <div className="bg-white/10 rounded-lg p-4">
                                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                                            <div>
                                                <div className="text-2xl font-bold">{progressData.total_concepts || 0}</div>
                                                <div className="text-sm text-blue-100">Total Concepts</div>
                                            </div>
                                            <div>
                                                <div className="text-2xl font-bold">{progressData.mastered_concepts || 0}</div>
                                                <div className="text-sm text-blue-100">Mastered</div>
                                            </div>
                                            <div>
                                                <div className="text-2xl font-bold">{progressData.weak_concepts || 0}</div>
                                                <div className="text-sm text-blue-100">Need Review</div>
                                            </div>
                                            <div>
                                                <div className="text-2xl font-bold">
                                                    {Math.round((progressData.mastered_concepts / progressData.total_concepts) * 100) || 0}%
                                                </div>
                                                <div className="text-sm text-blue-100">Progress</div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Main Content Area */}
                <main className="flex-1 p-6 bg-gray-50">
                    {activeTab === 'progress' ? (
                        // Progress View
                        <div className="space-y-6">
                            <h2 className="text-2xl font-bold text-gray-800">Progress Analytics</h2>
                            {progressData ? (
                                <ProgressDashboard progressData={progressData} />
                            ) : (
                                <div className="text-center py-8">
                                    <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                                    <p className="text-gray-600">No progress data available yet.</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        // Main Study Interface
                        <div className="h-full flex gap-6">
                            {/* Left Side - Chat & Upload (30%) */}
                            <div className="w-[30%] flex flex-col space-y-4">
                                {/* File Upload Section */}
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                                    <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                                        <Upload className="w-5 h-5 mr-2 text-blue-600" />
                                        Upload Study Material
                                    </h3>
                                    <FileUpload
                                        onFileProcessed={handleFileProcessed}
                                        isProcessing={isProcessing}
                                    />
                                </div>

                                {/* Chat Section */}
                                <div className="flex-1">
                                    <ChatBox
                                        chatHistory={chatHistory}
                                        userPrompt={userPrompt}
                                        setUserPrompt={setUserPrompt}
                                        onPromptSubmit={handlePromptSubmit}
                                        isProcessing={isProcessing}
                                    />
                                </div>
                            </div>

                            {/* Right Side - Study Content Sections (70%) */}
                            <div className="w-[70%]">
                                <StudyContentSections
                                    studyContent={studyContent}
                                    expandedSection={expandedSection}
                                    setExpandedSection={setExpandedSection}
                                    onActivityComplete={handleActivityComplete}
                                />
                            </div>
                        </div>
                    )}
                </main>
            </div>
        );
    };

    // Show loading screen during initial load
    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <Loading size="large" text="Loading Study Genie..." />
            </div>
        );
    }

    // Show login form if not authenticated
    if (!isAuthenticated) {
        return <LoginForm onLogin={handleLogin} />;
    }

    // Main app layout
    return (
        <div className="flex h-screen bg-gray-50">
            {/* Sidebar */}
            <Sidebar />
            <MainContent />

            {/* Mobile overlay */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}
        </div>
    );
};

// Root App component
const App = () => {
    return (
        <Router>
            <div className="App">
                <Toaster
                    position="top-right"
                    toastOptions={{
                        duration: 4000,
                        style: {
                            background: '#363636',
                            color: '#fff',
                        },
                        success: {
                            duration: 3000,
                            theme: {
                                primary: 'green',
                                secondary: 'black',
                            },
                        },
                    }}
                />
                <Routes>
                    <Route path="/" element={<StudyGenie />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </div>
        </Router>
    );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
