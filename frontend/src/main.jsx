// import React, { useState, useEffect } from 'react';
// import ReactDOM from 'react-dom/client';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import { Toaster } from 'react-hot-toast';
// import toast from 'react-hot-toast';
// import { motion, AnimatePresence } from 'framer-motion';
// import {
//     Upload,
//     User,
//     BookOpen,
//     Brain,
//     Target,
//     Home,
//     BarChart3,
//     Menu,
//     LogOut,
//     Bell
// } from 'lucide-react';

// // Import components
// import LoginForm from './components/LoginForm';
// import FileUpload from './components/FileUpload';
// import FlashCardDeck from './components/FlashCardDeck';
// import Quiz from './components/Quiz';
// import MatchTheFollowing from './components/MatchTheFollowing';
// import ProgressDashboard from './components/ProgressDashboard';
// import Loading from './components/Loading';
// import ChatBox from './components/ChatBox';
// import StudyContentSections from './components/StudyContentSections';

// // Import services and utilities
// import { apiService } from './services/api';
// import { storage } from './utils';

// // Import CSS
// import './index.css';

// const StudyGenie = () => {
//     // App state
//     const [isAuthenticated, setIsAuthenticated] = useState(false);
//     const [currentUser, setCurrentUser] = useState(null);
//     const [isLoading, setIsLoading] = useState(true);
//     const [activeTab, setActiveTab] = useState('main');
//     const [sidebarOpen, setSidebarOpen] = useState(false);

//     // Study content state
//     const [studyContent, setStudyContent] = useState(null);
//     const [isProcessing, setIsProcessing] = useState(false);
//     const [progressData, setProgressData] = useState(null);

//     // Chat and content state
//     const [userPrompt, setUserPrompt] = useState('');
//     const [chatHistory, setChatHistory] = useState([]);
//     const [expandedSection, setExpandedSection] = useState(null);
//     const [showDashboard, setShowDashboard] = useState(false);

//     // Check for existing user session on load
//     useEffect(() => {
//         const checkSession = async () => {
//             try {
//                 const savedUser = storage.get('currentUser');
//                 if (savedUser) {
//                     setCurrentUser(savedUser);
//                     setIsAuthenticated(true);
//                     await loadUserProgress(savedUser.student_id);
//                 }
//             } catch (error) {
//                 console.error('Session check error:', error);
//             } finally {
//                 setIsLoading(false);
//             }
//         };

//         checkSession();
//     }, []);

//     // API health check
//     useEffect(() => {
//         const healthCheck = async () => {
//             try {
//                 await apiService.checkHealth();
//                 console.log('Backend API is healthy');
//             } catch (error) {
//                 console.warn('Backend API is not available:', error.message);
//                 toast.error('Backend API is not available. Using demo mode.');
//             }
//         };

//         if (isAuthenticated) {
//             healthCheck();
//         }
//     }, [isAuthenticated]);

//     const handleLogin = async (formData) => {
//         try {
//             setIsLoading(true);
//             const response = await apiService.createStudent(formData);

//             const user = {
//                 student_id: response.student_id,
//                 username: response.username,
//                 email: response.email,
//                 full_name: response.full_name
//             };

//             setCurrentUser(user);
//             setIsAuthenticated(true);
//             storage.set('currentUser', user);

//             // Load initial progress data
//             await loadUserProgress(user.student_id);

//             return response;
//         } catch (error) {
//             console.error('Login error:', error);
//             throw error;
//         } finally {
//             setIsLoading(false);
//         }
//     };

//     const handleLogout = () => {
//         setIsAuthenticated(false);
//         setCurrentUser(null);
//         setStudyContent(null);
//         setProgressData(null);
//         storage.remove('currentUser');
//         toast.success('Logged out successfully');
//     };

//     const loadUserProgress = async (studentId) => {
//         try {
//             const progress = await apiService.getStudentProgress(studentId);
//             setProgressData(progress);
//         } catch (error) {
//             console.error('Failed to load progress:', error);
//             // Set mock progress data for demo
//             setProgressData({
//                 student_id: studentId,
//                 total_concepts: 25,
//                 mastered_concepts: 8,
//                 weak_concepts: 3,
//                 concept_progress: [
//                     {
//                         concept_id: 1,
//                         concept_name: 'Basic Algebra',
//                         status: 'mastered',
//                         mastery_score: 95,
//                         attempts_count: 5,
//                         correct_answers: 19,
//                         total_questions: 20
//                     },
//                     {
//                         concept_id: 2,
//                         concept_name: 'Quadratic Equations',
//                         status: 'in_progress',
//                         mastery_score: 75,
//                         attempts_count: 3,
//                         correct_answers: 15,
//                         total_questions: 20
//                     }
//                 ],
//                 subject_progress: [
//                     { subject_name: 'Mathematics', progress_percentage: 75 },
//                     { subject_name: 'Physics', progress_percentage: 60 },
//                     { subject_name: 'Chemistry', progress_percentage: 85 }
//                 ],
//                 recent_activity: [
//                     { activity_type: 'quiz_attempt', concept_name: 'Algebra Basics', timestamp: new Date(), score: 85 }
//                 ]
//             });
//         }
//     };

//     const handleFileProcessed = async (fileData) => {
//         try {
//             setIsProcessing(true);

//             // Add to chat history
//             setChatHistory(prev => [...prev, {
//                 type: 'file',
//                 content: `File uploaded: ${fileData.fileName}`,
//                 timestamp: new Date()
//             }]);

//             // Extract metadata from filename or use defaults
//             const metadata = {
//                 subject: 'General Studies',
//                 chapter: 'Chapter 1',
//                 concept: fileData.fileName.replace(/\.[^/.]+$/, ''), // Remove extension
//                 difficulty: 'Medium'
//             };

//             // Process file content with backend
//             const response = await apiService.processFileContent(
//                 fileData.content,
//                 currentUser.student_id,
//                 metadata
//             );

//             // Create study content from LLM response
//             const content = response.enhanced_response || response.llm_response;

//             // Convert backend response format to frontend format
//             const studyContent = {
//                 flashcards: Object.entries(content.flashcards || {}).map(([key, card]) => ({
//                     id: key,
//                     question: card.question,
//                     answer: card.answer,
//                     key_concepts: card.key_concepts,
//                     key_concepts_data: card.key_concepts_data,
//                     difficulty: card.difficulty
//                 })),
//                 quiz: Object.entries(content.quiz || {}).map(([key, question]) => ({
//                     id: key,
//                     question: question.question,
//                     options: question.options,
//                     correct_answer: question.correct_answer,
//                     explanation: question.explanation
//                 })),
//                 matchTheFollowing: content.match_the_following,
//                 summary: content.summary,
//                 learningObjectives: content.learning_objectives || []
//             };

//             setStudyContent(studyContent);

//             // Add success message to chat
//             setChatHistory(prev => [...prev, {
//                 type: 'success',
//                 content: 'Study content generated successfully! Check the sections on the right.',
//                 timestamp: new Date()
//             }]);

//             // Update progress
//             await loadUserProgress(currentUser.student_id);

//             toast.success('Study content generated successfully!');
//         } catch (error) {
//             console.error('File processing error:', error);
//             setChatHistory(prev => [...prev, {
//                 type: 'error',
//                 content: 'Failed to process file. Please try again.',
//                 timestamp: new Date()
//             }]);
//             toast.error('Failed to process file. Please try again.');
//         } finally {
//             setIsProcessing(false);
//         }
//     };

//     const handlePromptSubmit = async () => {
//         if (!userPrompt.trim()) return;

//         try {
//             setIsProcessing(true);

//             // Add user message to chat
//             setChatHistory(prev => [...prev, {
//                 type: 'user',
//                 content: userPrompt,
//                 timestamp: new Date()
//             }]);

//             // Create mock LLM response based on prompt
//             const mockLLMResponse = {
//                 flashcards: {
//                     card1: {
//                         question: `What is the main concept about: ${userPrompt}?`,
//                         answer: "Based on your query, here's a comprehensive answer.",
//                         key_concepts: "Core concept identification",
//                         key_concepts_data: "Detailed analysis of the main topics",
//                         difficulty: "Medium"
//                     },
//                     // Add more cards...
//                 },
//                 quiz: {
//                     Q1: {
//                         question: `Which statement best describes: ${userPrompt}?`,
//                         options: ["Option A", "Option B", "Option C", "Option D"],
//                         correct_answer: "Option A",
//                         explanation: "This is correct because..."
//                     },
//                     // Add more questions...
//                 },
//                 match_the_following: {
//                     columnA: ["Term 1", "Term 2", "Term 3"],
//                     columnB: ["Definition 1", "Definition 2", "Definition 3"],
//                     mappings: [
//                         { A: "Term 1", B: "Definition 1" },
//                         { A: "Term 2", B: "Definition 2" },
//                         { A: "Term 3", B: "Definition 3" }
//                     ]
//                 },
//                 summary: `This is a comprehensive summary based on your query: ${userPrompt}`,
//                 learning_objectives: [
//                     "Understand the main concepts presented",
//                     "Apply knowledge in practical scenarios",
//                     "Analyze and synthesize information"
//                 ]
//             };

//             const requestData = {
//                 student_id: currentUser.student_id,
//                 subject_name: "General Studies",
//                 chapter_name: "Chapter 1",
//                 concept_name: "User Query",
//                 llm_response: mockLLMResponse,
//                 user_query: userPrompt,
//                 difficulty_level: "Medium"
//             };

//             const response = await apiService.processLLMResponse(requestData);
//             const content = response.enhanced_response || response.llm_response;

//             // Convert backend response format to frontend format
//             const studyContent = {
//                 flashcards: Object.entries(content.flashcards || {}).map(([key, card]) => ({
//                     id: key,
//                     question: card.question,
//                     answer: card.answer,
//                     key_concepts: card.key_concepts,
//                     key_concepts_data: card.key_concepts_data,
//                     difficulty: card.difficulty
//                 })),
//                 quiz: Object.entries(content.quiz || {}).map(([key, question]) => ({
//                     id: key,
//                     question: question.question,
//                     options: question.options,
//                     correct_answer: question.correct_answer,
//                     explanation: question.explanation
//                 })),
//                 matchTheFollowing: content.match_the_following,
//                 summary: content.summary,
//                 learningObjectives: content.learning_objectives || []
//             };

//             setStudyContent(studyContent);

//             // Add AI response to chat
//             setChatHistory(prev => [...prev, {
//                 type: 'ai',
//                 content: 'Study content generated based on your query! Check the sections on the right.',
//                 timestamp: new Date()
//             }]);

//             setUserPrompt('');

//             // Update progress
//             await loadUserProgress(currentUser.student_id);

//             toast.success('Study content generated successfully!');
//         } catch (error) {
//             console.error('Prompt processing error:', error);
//             setChatHistory(prev => [...prev, {
//                 type: 'error',
//                 content: 'Failed to process your query. Please try again.',
//                 timestamp: new Date()
//             }]);
//             toast.error('Failed to process your query. Please try again.');
//         } finally {
//             setIsProcessing(false);
//         }
//     };

//     const handleActivityComplete = async (activityType, results) => {
//         try {
//             // Update progress in backend
//             await apiService.updateStudentProgress(currentUser.student_id, {
//                 concept_id: 1, // Mock concept ID
//                 correct_answers: results.correctAnswers || results.correctMatches || 0,
//                 total_questions: results.totalCards || results.totalQuestions || results.totalItems || 0,
//                 activity_type: activityType
//             });

//             // Reload progress
//             await loadUserProgress(currentUser.student_id);

//             toast.success(`${activityType} completed! Progress updated.`);
//         } catch (error) {
//             console.error('Progress update error:', error);
//             toast.error('Failed to update progress');
//         }
//     };

//     // Navigation items
//     const navigationItems = [
//         { id: 'progress', label: 'Progress', icon: BarChart3 },
//     ];

//     const Sidebar = () => (
//         <motion.div
//             className={`
//         fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-xl transform transition-transform duration-300 lg:translate-x-0 lg:static lg:inset-0
//         ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
//       `}
//             initial={false}
//             animate={{ x: sidebarOpen ? 0 : -256 }}
//         >
//             <div className="flex flex-col h-full">
//                 {/* Header */}
//                 <div className="p-6 border-b border-gray-200">
//                     <div className="flex items-center space-x-3">
//                         <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
//                             <Brain className="w-6 h-6 text-white" />
//                         </div>
//                         <div>
//                             <h1 className="text-xl font-bold text-gray-800">Study Genie</h1>
//                             <p className="text-sm text-gray-600">AI-Powered Learning</p>
//                         </div>
//                     </div>
//                 </div>

//                 {/* User Info */}
//                 <div className="p-6 border-b border-gray-200">
//                     <div className="flex items-center space-x-3">
//                         <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
//                             <User className="w-4 h-4 text-blue-600" />
//                         </div>
//                         <div className="flex-1 min-w-0">
//                             <p className="text-sm font-medium text-gray-800 truncate">
//                                 {currentUser?.full_name || currentUser?.username}
//                             </p>
//                             <p className="text-xs text-gray-500 truncate">{currentUser?.email}</p>
//                         </div>
//                     </div>
//                 </div>

//                 {/* Navigation */}
//                 <nav className="flex-1 p-4">
//                     <ul className="space-y-2">
//                         {navigationItems.map((item) => (
//                             <li key={item.id}>
//                                 <button
//                                     onClick={() => {
//                                         if (item.id === 'progress') {
//                                             setActiveTab(item.id);
//                                         }
//                                         setSidebarOpen(false);
//                                     }}
//                                     className={`
//                     w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200
//                     ${activeTab === item.id
//                                             ? 'bg-blue-100 text-blue-700 font-medium'
//                                             : 'text-gray-600 hover:bg-gray-100 hover:text-gray-800'
//                                         }
//                   `}
//                                 >
//                                     <item.icon className="w-5 h-5" />
//                                     <span>{item.label}</span>
//                                 </button>
//                             </li>
//                         ))}
//                     </ul>
//                 </nav>

//                 {/* Logout */}
//                 <div className="p-4 border-t border-gray-200">
//                     <button
//                         onClick={handleLogout}
//                         className="w-full flex items-center space-x-3 px-4 py-3 text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200"
//                     >
//                         <LogOut className="w-5 h-5" />
//                         <span>Logout</span>
//                     </button>
//                 </div>
//             </div>
//         </motion.div>
//     );

//     const MainContent = () => {
//         return (
//             <div className="flex-1 flex flex-col min-h-screen">
//                 {/* Header */}
//                 <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
//                     <div className="flex items-center justify-between">
//                         <div className="flex items-center space-x-4">
//                             <button
//                                 onClick={() => setSidebarOpen(!sidebarOpen)}
//                                 className="lg:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
//                             >
//                                 <Menu className="w-6 h-6 text-gray-600" />
//                             </button>

//                             <div className="flex items-center space-x-4">
//                                 <h1 className="text-xl font-semibold text-gray-800">
//                                     Study Genie
//                                 </h1>
//                                 <button
//                                     onClick={() => setShowDashboard(!showDashboard)}
//                                     className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
//                                 >
//                                     <Home className="w-4 h-4 text-gray-600" />
//                                     <span className="text-sm text-gray-600">Dashboard</span>
//                                 </button>
//                             </div>
//                         </div>

//                         <div className="flex items-center space-x-4">
//                             <div className="text-right">
//                                 <p className="text-sm font-medium text-gray-800">
//                                     {currentUser?.full_name || currentUser?.username}
//                                 </p>
//                                 <p className="text-xs text-gray-500">{currentUser?.email}</p>
//                             </div>
//                             <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative">
//                                 <Bell className="w-5 h-5 text-gray-600" />
//                                 <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
//                             </button>
//                         </div>
//                     </div>
//                 </header>

//                 {/* Dashboard Dropdown */}
//                 <AnimatePresence>
//                     {showDashboard && (
//                         <motion.div
//                             initial={{ opacity: 0, height: 0 }}
//                             animate={{ opacity: 1, height: 'auto' }}
//                             exit={{ opacity: 0, height: 0 }}
//                             className="bg-gradient-to-r from-blue-500 to-purple-600 text-white overflow-hidden"
//                         >
//                             <div className="p-6">
//                                 <h2 className="text-2xl font-bold mb-2">
//                                     Welcome back, {currentUser?.full_name || currentUser?.username}!
//                                 </h2>
//                                 <p className="text-blue-100 mb-4">Ready to continue your learning journey?</p>

//                                 {progressData && (
//                                     <div className="bg-white/10 rounded-lg p-4">
//                                         <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
//                                             <div>
//                                                 <div className="text-2xl font-bold">{progressData.total_concepts || 0}</div>
//                                                 <div className="text-sm text-blue-100">Total Concepts</div>
//                                             </div>
//                                             <div>
//                                                 <div className="text-2xl font-bold">{progressData.mastered_concepts || 0}</div>
//                                                 <div className="text-sm text-blue-100">Mastered</div>
//                                             </div>
//                                             <div>
//                                                 <div className="text-2xl font-bold">{progressData.weak_concepts || 0}</div>
//                                                 <div className="text-sm text-blue-100">Need Review</div>
//                                             </div>
//                                             <div>
//                                                 <div className="text-2xl font-bold">
//                                                     {Math.round((progressData.mastered_concepts / progressData.total_concepts) * 100) || 0}%
//                                                 </div>
//                                                 <div className="text-sm text-blue-100">Progress</div>
//                                             </div>
//                                         </div>
//                                     </div>
//                                 )}
//                             </div>
//                         </motion.div>
//                     )}
//                 </AnimatePresence>

//                 {/* Main Content Area */}
//                 <main className="flex-1 p-6 bg-gray-50">
//                     {activeTab === 'progress' ? (
//                         // Progress View
//                         <div className="space-y-6">
//                             <h2 className="text-2xl font-bold text-gray-800">Progress Analytics</h2>
//                             {progressData ? (
//                                 <ProgressDashboard progressData={progressData} />
//                             ) : (
//                                 <div className="text-center py-8">
//                                     <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-400" />
//                                     <p className="text-gray-600">No progress data available yet.</p>
//                                 </div>
//                             )}
//                         </div>
//                     ) : (
//                         // Main Study Interface
//                         <div className="h-full flex gap-6">
//                             {/* Left Side - Chat & Upload (30%) */}
//                             <div className="w-[30%] flex flex-col space-y-4">
//                                 {/* File Upload Section */}
//                                 <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
//                                     <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
//                                         <Upload className="w-5 h-5 mr-2 text-blue-600" />
//                                         Upload Study Material
//                                     </h3>
//                                     <FileUpload
//                                         onFileProcessed={handleFileProcessed}
//                                         isProcessing={isProcessing}
//                                     />
//                                 </div>

//                                 {/* Chat Section */}
//                                 <div className="flex-1">
//                                     <ChatBox
//                                         chatHistory={chatHistory}
//                                         userPrompt={userPrompt}
//                                         setUserPrompt={setUserPrompt}
//                                         onPromptSubmit={handlePromptSubmit}
//                                         isProcessing={isProcessing}
//                                     />
//                                 </div>
//                             </div>

//                             {/* Right Side - Study Content Sections (70%) */}
//                             <div className="w-[70%]">
//                                 <StudyContentSections
//                                     studyContent={studyContent}
//                                     expandedSection={expandedSection}
//                                     setExpandedSection={setExpandedSection}
//                                     onActivityComplete={handleActivityComplete}
//                                 />
//                             </div>
//                         </div>
//                     )}
//                 </main>
//             </div>
//         );
//     };

//     // Show loading screen during initial load
//     if (isLoading) {
//         return (
//             <div className="min-h-screen flex items-center justify-center bg-gray-50">
//                 <Loading size="large" text="Loading Study Genie..." />
//             </div>
//         );
//     }

//     // Show login form if not authenticated
//     if (!isAuthenticated) {
//         return <LoginForm onLogin={handleLogin} />;
//     }

//     // Main app layout
//     return (
//         <div className="flex h-screen bg-gray-50">
//             {/* Sidebar */}
//             <Sidebar />
//             <MainContent />

//             {/* Mobile overlay */}
//             {sidebarOpen && (
//                 <div
//                     className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
//                     onClick={() => setSidebarOpen(false)}
//                 />
//             )}
//         </div>
//     );
// };

// // Root App component
// const App = () => {
//     return (
//         <Router>
//             <div className="App">
//                 <Toaster
//                     position="top-right"
//                     toastOptions={{
//                         duration: 4000,
//                         style: {
//                             background: '#363636',
//                             color: '#fff',
//                         },
//                         success: {
//                             duration: 3000,
//                             theme: {
//                                 primary: 'green',
//                                 secondary: 'black',
//                             },
//                         },
//                     }}
//                 />
//                 <Routes>
//                     <Route path="/" element={<StudyGenie />} />
//                     <Route path="*" element={<Navigate to="/" replace />} />
//                 </Routes>
//             </div>
//         </Router>
//     );
// };

// // Render the app
// ReactDOM.createRoot(document.getElementById('root')).render(
//     <React.StrictMode>
//         <App />
//     </React.StrictMode>
// );




import React, { useEffect, useMemo, useRef, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import ReactDOM from "react-dom/client";
import { motion, AnimatePresence } from "framer-motion";
import toast, { Toaster } from "react-hot-toast";
import {
    Sparkles,
    Brain,
    Upload,
    MessageSquareText,
    Command,
    BarChart3,
    Home,
    LogOut,
    Bell,
    Settings,
    Flame,
    Star,
    Crown,
    Menu,
    X,
    ChevronRight,
    ChevronLeft,
    HelpCircle,
    Bot,
    Rocket,
    Trophy,
} from "lucide-react";

// === Bring your existing components/services ===
// These are assumed to already exist in your project, unchanged.
import LoginForm from "./components/LoginForm";
import FileUpload from "./components/FileUpload";
import ProgressDashboard from "./components/ProgressDashboard";
import Loading from "./components/Loading";
import ChatBox from "./components/ChatBox";
import StudyContentSections from "./components/StudyContentSections";
import { apiService } from "./services/api";
import { storage } from "./utils";
import "./index.css";

// =====================
// Utility Components UI
// =====================
const GlassCard = ({ className = "", children }) => (
    <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className={`rounded-2xl border border-white/20 bg-white/60 dark:bg-white/5 backdrop-blur-xl shadow-[0_8px_30px_rgba(0,0,0,0.08)] ${className}`}
    >
        {children}
    </motion.div>
);

const Pill = ({ icon: Icon, label, value, className = "" }) => (
    <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm bg-white/70 dark:bg-white/10 border border-black/5 ${className}`}>
        {Icon && <Icon className="w-4 h-4 opacity-80" />}
        <span className="opacity-70">{label}</span>
        {value !== undefined && (
            <span className="font-semibold text-gray-900 dark:text-white">{value}</span>
        )}
    </div>
);

const GradientText = ({ children }) => (
    <span className="bg-gradient-to-r from-violet-500 via-fuchsia-500 to-amber-400 bg-clip-text text-transparent">
        {children}
    </span>
);

// XP ring component
const XPRing = ({ value = 62 }) => {
    const R = 26;
    const C = 2 * Math.PI * R;
    const dash = Math.max(0, Math.min(100, value)) * (C / 100);
    return (
        <svg viewBox="0 0 64 64" className="w-14 h-14">
            <circle cx="32" cy="32" r={R} className="fill-none stroke-gray-200/60" strokeWidth="8" />
            <motion.circle
                cx="32"
                cy="32"
                r={R}
                className="fill-none stroke-[url(#grad)]"
                strokeLinecap="round"
                strokeWidth="8"
                initial={{ strokeDasharray: `0 ${C}` }}
                animate={{ strokeDasharray: `${dash} ${C}` }}
                transition={{ duration: 1, ease: "easeOut" }}
            />
            <defs>
                <linearGradient id="grad" x1="0" x2="1" y1="0" y2="1">
                    <stop offset="0%" stopColor="#A78BFA" />
                    <stop offset="50%" stopColor="#F472B6" />
                    <stop offset="100%" stopColor="#F59E0B" />
                </linearGradient>
            </defs>
            <text
                x="50%"
                y="52%"
                dominantBaseline="middle"
                textAnchor="middle"
                className="fill-gray-800 dark:fill-white font-semibold text-[10px]"
            >
                {value}%
            </text>
        </svg>
    );
};

// Floating background orbs
const Orbs = () => (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-24 -left-16 h-72 w-72 rounded-full bg-fuchsia-400/20 blur-3xl" />
        <div className="absolute top-48 -right-12 h-72 w-72 rounded-full bg-violet-400/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-amber-400/20 blur-3xl" />
    </div>
);

// =====================
// Main App (New UI)
// =====================
const StudyGenieWow = () => {
    // Core state
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progressData, setProgressData] = useState(null);
    const [studyContent, setStudyContent] = useState(null);
    const [userPrompt, setUserPrompt] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [expandedSection, setExpandedSection] = useState(null);
    const [active, setActive] = useState("learn"); // learn | progress
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [showDash, setShowDash] = useState(true);

    // === Session on load ===
    useEffect(() => {
        (async () => {
            try {
                const saved = storage.get("currentUser");
                if (saved) {
                    setCurrentUser(saved);
                    setIsAuthenticated(true);
                    await loadUserProgress(saved.student_id);
                }
            } catch (e) {
                console.error("Session check error:", e);
            } finally {
                setIsLoading(false);
            }
        })();
    }, []);

    // === Health check after login ===
    useEffect(() => {
        if (!isAuthenticated) return;
        (async () => {
            try {
                await apiService.checkHealth();
            } catch (e) {
                toast.error("Backend API is not available. Using demo mode.");
            }
        })();
    }, [isAuthenticated]);

    const loadUserProgress = async (studentId) => {
        try {
            const progress = await apiService.getStudentProgress(studentId);
            setProgressData(progress);
        } catch (e) {
            console.warn("Using mock progress");
            setProgressData({
                student_id: studentId,
                total_concepts: 25,
                mastered_concepts: 8,
                weak_concepts: 3,
                concept_progress: [
                    {
                        concept_id: 1,
                        concept_name: "Basic Algebra",
                        status: "mastered",
                        mastery_score: 95,
                        attempts_count: 5,
                        correct_answers: 19,
                        total_questions: 20,
                    },
                    {
                        concept_id: 2,
                        concept_name: "Quadratic Equations",
                        status: "in_progress",
                        mastery_score: 75,
                        attempts_count: 3,
                        correct_answers: 15,
                        total_questions: 20,
                    },
                ],
                subject_progress: [
                    { subject_name: "Mathematics", progress_percentage: 75 },
                    { subject_name: "Physics", progress_percentage: 60 },
                    { subject_name: "Chemistry", progress_percentage: 85 },
                ],
                recent_activity: [
                    {
                        activity_type: "quiz_attempt",
                        concept_name: "Algebra Basics",
                        timestamp: new Date(),
                        score: 85,
                    },
                ],
            });
        }
    };

    const handleLogin = async (formData) => {
        try {
            setIsLoading(true);
            const res = await apiService.createStudent(formData);
            const user = {
                student_id: res.student_id,
                username: res.username,
                email: res.email,
                full_name: res.full_name,
            };
            setCurrentUser(user);
            setIsAuthenticated(true);
            storage.set("currentUser", user);
            await loadUserProgress(user.student_id);
            return res;
        } catch (e) {
            console.error("Login error:", e);
            throw e;
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogout = () => {
        setIsAuthenticated(false);
        setCurrentUser(null);
        setStudyContent(null);
        setProgressData(null);
        storage.remove("currentUser");
        toast.success("Logged out successfully");
    };

    const handleFileProcessed = async (fileData) => {
        try {
            setIsProcessing(true);
            setChatHistory((p) => [
                ...p,
                { type: "file", content: `File uploaded: ${fileData.fileName}`, timestamp: new Date() },
            ]);

            // Use the enhanced response directly from the upload
            const content = fileData.enhancedResponse;

            const mapped = {
                flashcards: Object.entries(content.flashcards || {}).map(([id, card]) => ({
                    id,
                    question: card.question,
                    answer: card.answer,
                    key_concepts: card.key_concepts,
                    key_concepts_data: card.key_concepts_data,
                    difficulty: card.difficulty,
                })),
                quiz: Object.entries(content.quiz || {}).map(([id, q]) => ({
                    id,
                    question: q.question,
                    options: q.options,
                    correct_answer: q.correct_answer,
                    explanation: q.explanation,
                })),
                matchTheFollowing: content.match_the_following,
                summary: content.summary,
                learningObjectives: content.learning_objectives || [],
            };

            setStudyContent(mapped);
            setChatHistory((p) => [
                ...p,
                { type: "success", content: "Study content generated!", timestamp: new Date() },
            ]);
            await loadUserProgress(currentUser.student_id);
            toast.success("Study content generated successfully!");
        } catch (e) {
            console.error("File processing error:", e);
            setChatHistory((p) => [
                ...p,
                { type: "error", content: "Failed to process file.", timestamp: new Date() },
            ]);
            toast.error("Failed to process file. Please try again.");
        } finally {
            setIsProcessing(false);
        }
    };

    const handlePromptSubmit = async () => {
        if (!userPrompt.trim()) return;
        try {
            setIsProcessing(true);
            setChatHistory((p) => [
                ...p,
                { type: "user", content: userPrompt, timestamp: new Date() },
            ]);

            const mockLLMResponse = {
                flashcards: {
                    card1: {
                        question: `What is the main concept about: ${userPrompt}?`,
                        answer: "Based on your query, here's a comprehensive answer.",
                        key_concepts: "Core concept identification",
                        key_concepts_data: "Detailed analysis of the main topics",
                        difficulty: "Medium",
                    },
                },
                quiz: {
                    Q1: {
                        question: `Which statement best describes: ${userPrompt}?`,
                        options: ["Option A", "Option B", "Option C", "Option D"],
                        correct_answer: "Option A",
                        explanation: "This is correct because...",
                    },
                },
                match_the_following: {
                    columnA: ["Term 1", "Term 2", "Term 3"],
                    columnB: ["Definition 1", "Definition 2", "Definition 3"],
                    mappings: [
                        { A: "Term 1", B: "Definition 1" },
                        { A: "Term 2", B: "Definition 2" },
                        { A: "Term 3", B: "Definition 3" },
                    ],
                },
                summary: `This is a comprehensive summary based on your query: ${userPrompt}`,
                learning_objectives: [
                    "Understand the main concepts presented",
                    "Apply knowledge in practical scenarios",
                    "Analyze and synthesize information",
                ],
            };

            const req = {
                student_id: currentUser.student_id,
                subject_name: "General Studies",
                chapter_name: "Chapter 1",
                concept_name: "User Query",
                llm_response: mockLLMResponse,
                user_query: userPrompt,
                difficulty_level: "Medium",
            };

            const response = await apiService.processLLMResponse(req);
            const content = response.enhanced_response || response.llm_response;
            const mapped = {
                flashcards: Object.entries(content.flashcards || {}).map(([id, card]) => ({
                    id,
                    question: card.question,
                    answer: card.answer,
                    key_concepts: card.key_concepts,
                    key_concepts_data: card.key_concepts_data,
                    difficulty: card.difficulty,
                })),
                quiz: Object.entries(content.quiz || {}).map(([id, q]) => ({
                    id,
                    question: q.question,
                    options: q.options,
                    correct_answer: q.correct_answer,
                    explanation: q.explanation,
                })),
                matchTheFollowing: content.match_the_following,
                summary: content.summary,
                learningObjectives: content.learning_objectives || [],
            };
            setStudyContent(mapped);
            setChatHistory((p) => [
                ...p,
                { type: "ai", content: "Study content generated from your prompt!", timestamp: new Date() },
            ]);
            setUserPrompt("");
            await loadUserProgress(currentUser.student_id);
            toast.success("Study content generated successfully!");
        } catch (e) {
            console.error("Prompt processing error:", e);
            setChatHistory((p) => [
                ...p,
                { type: "error", content: "Failed to process your query.", timestamp: new Date() },
            ]);
            toast.error("Failed to process your query. Please try again.");
        } finally {
            setIsProcessing(false);
        }
    };

    const handleActivityComplete = async (activityType, results) => {
        try {
            await apiService.updateStudentProgress(currentUser.student_id, {
                concept_id: 1,
                correct_answers:
                    results.correctAnswers || results.correctMatches || 0,
                total_questions:
                    results.totalCards || results.totalQuestions || results.totalItems || 0,
                activity_type: activityType,
            });
            await loadUserProgress(currentUser.student_id);
            toast.success(`${activityType} completed! Progress updated.`);
        } catch (e) {
            console.error("Progress update error:", e);
            toast.error("Failed to update progress");
        }
    };

    // =====================
    // Render branches
    // =====================
    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-950">
                <Loading size="large" text="Summoning StudyGenie…" />
            </div>
        );
    }
    if (!isAuthenticated) return <LoginForm onLogin={handleLogin} />;

    return (
        <div className="relative min-h-screen bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-950 dark:to-black">
            <Orbs />
            <header className="sticky top-0 z-40 backdrop-blur-xl border-b border-black/5 bg-white/60 dark:bg-white/5">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center gap-3">
                    <button
                        onClick={() => setSidebarOpen((s) => !s)}
                        className="p-2 rounded-xl hover:bg-black/5 transition lg:hidden"
                        aria-label="Toggle sidebar"
                    >
                        {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                    </button>
                    <div className="flex items-center gap-2">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-500 to-amber-400 text-white flex items-center justify-center shadow-md">
                            <Brain className="w-5 h-5" />
                        </div>
                        <h1 className="font-semibold text-lg sm:text-xl">
                            <GradientText>StudyGenie</GradientText>
                        </h1>
                        <Pill icon={Flame} label="Streak" value={7} className="ml-2 hidden md:flex" />
                        <Pill icon={Trophy} label="Level" value={"3"} className="hidden md:flex" />
                    </div>
                    <div className="ml-auto flex items-center gap-3">
                        <GlassCard className="px-3 py-1.5 hidden sm:flex items-center gap-2">
                            <Command className="w-4 h-4 opacity-70" />
                            <span className="text-xs opacity-70">Press</span>
                            <kbd className="text-xs font-semibold px-1.5 py-0.5 rounded bg-black/10">Ctrl</kbd>
                            <span className="text-xs">+</span>
                            <kbd className="text-xs font-semibold px-1.5 py-0.5 rounded bg-black/10">K</kbd>
                        </GlassCard>
                        <button className="relative p-2 rounded-xl hover:bg-black/5">
                            <Bell className="w-5 h-5" />
                            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-rose-500" />
                        </button>
                        <button
                            onClick={handleLogout}
                            className="px-3 py-1.5 rounded-xl bg-black text-white hover:opacity-90 flex items-center gap-2"
                        >
                            <LogOut className="w-4 h-4" />
                            <span className="text-sm">Logout</span>
                        </button>
                    </div>
                </div>
            </header>

            {/* Sidebar */}
            <AnimatePresence>
                {sidebarOpen && (
                    <motion.aside
                        initial={{ x: -260, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: -260, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 280, damping: 30 }}
                        className="fixed inset-y-0 left-0 z-40 w-[260px] p-4 bg-white/80 dark:bg-white/10 backdrop-blur-xl border-r border-black/5 shadow-xl lg:hidden"
                    >
                        <SidebarContent
                            active={active}
                            onSelect={(id) => {
                                setActive(id);
                                setSidebarOpen(false);
                            }}
                            progressData={progressData}
                        />
                    </motion.aside>
                )}
            </AnimatePresence>

            <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-6 px-4 sm:px-6 py-6">
                {/* Static sidebar for lg+ */}
                <aside className="hidden lg:block">
                    <SidebarContent
                        active={active}
                        onSelect={(id) => setActive(id)}
                        progressData={progressData}
                    />
                </aside>

                {/* Main content */}
                <div className="space-y-6">
                    {/* Welcome/Gamified banner */}
                    <GlassCard className="p-5">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <h2 className="text-xl sm:text-2xl font-bold">
                                    Welcome back, <GradientText>{currentUser?.full_name || currentUser?.username}</GradientText> 👋
                                </h2>
                                <p className="text-sm opacity-70 mt-1">
                                    Ready to level up your knowledge today? Upload, chat, or jump into activities.
                                </p>
                            </div>
                            <div className="flex items-center gap-4">
                                <div className="text-center">
                                    <XPRing value={Math.round(((progressData?.mastered_concepts || 0) / (progressData?.total_concepts || 1)) * 100)} />
                                    <div className="text-[11px] opacity-70 mt-1">Overall</div>
                                </div>
                                <div className="text-center hidden sm:block">
                                    <div className="text-xs opacity-70">Mastered</div>
                                    <div className="text-xl font-bold">{progressData?.mastered_concepts ?? 0}</div>
                                </div>
                                <div className="text-center hidden sm:block">
                                    <div className="text-xs opacity-70">Total</div>
                                    <div className="text-xl font-bold">{progressData?.total_concepts ?? 0}</div>
                                </div>
                            </div>
                        </div>
                    </GlassCard>

                    {/* Two-column workbench */}
                    {active === "progress" ? (
                        <GlassCard className="p-5">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <BarChart3 className="w-5 h-5" /> Progress Analytics
                            </h3>
                            {progressData ? (
                                <ProgressDashboard progressData={progressData} />
                            ) : (
                                <div className="text-center py-10 opacity-70">No progress yet. Start an activity!</div>
                            )}
                        </GlassCard>
                    ) : (
                        <div className="grid grid-cols-1 xl:grid-cols-[380px_1fr] gap-6">
                            {/* Left column: Upload + Chat */}
                            <div className="space-y-6">
                                <GlassCard className="p-5">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-sky-400 text-white flex items-center justify-center">
                                                <Upload className="w-5 h-5" />
                                            </div>
                                            <h3 className="text-lg font-semibold">Upload Study Material</h3>
                                        </div>
                                        <Pill icon={Rocket} label="Boost AI" />
                                    </div>
                                    <FileUpload onFileProcessed={handleFileProcessed} isProcessing={isProcessing} />
                                </GlassCard>

                                <GlassCard className="p-5">
                                    <div className="flex items-center gap-2 mb-3">
                                        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-500 to-orange-400 text-white flex items-center justify-center">
                                            <Bot className="w-5 h-5" />
                                        </div>
                                        <h3 className="text-lg font-semibold">Chat & Create</h3>
                                    </div>
                                    <ChatBox
                                        chatHistory={chatHistory}
                                        userPrompt={userPrompt}
                                        setUserPrompt={setUserPrompt}
                                        onPromptSubmit={handlePromptSubmit}
                                        isProcessing={isProcessing}
                                    />
                                    <div className="mt-3 text-xs opacity-70 flex items-center gap-2">
                                        <HelpCircle className="w-4 h-4" /> Try prompts like: "Summarise this chapter", "Create 5 flashcards on optics", "Give a 10-question quiz".
                                    </div>
                                </GlassCard>
                            </div>

                            {/* Right column: Study content */}
                            <div>
                                <GlassCard className="p-3 sm:p-5">
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center gap-2">
                                            <Sparkles className="w-5 h-5" />
                                            <h3 className="text-lg font-semibold">Your Learning Canvas</h3>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <Pill icon={Star} label="Focus" value={"On"} />
                                            <Pill icon={Crown} label="Mode" value={"Pro"} />
                                        </div>
                                    </div>

                                    {studyContent ? (
                                        <StudyContentSections
                                            studyContent={studyContent}
                                            expandedSection={expandedSection}
                                            setExpandedSection={setExpandedSection}
                                            onActivityComplete={handleActivityComplete}
                                        />
                                    ) : (
                                        <EmptyState />
                                    )}
                                </GlassCard>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <Toaster
                position="top-right"
                toastOptions={{
                    duration: 3500,
                    style: { background: "#18181b", color: "#fff" },
                }}
            />
        </div>
    );
};

const SidebarContent = ({ active, onSelect, progressData }) => {
    const items = [
        { id: "learn", label: "Learn", icon: Home },
        { id: "progress", label: "Progress", icon: BarChart3 },
    ];
    return (
        <div className="flex flex-col gap-4 h-full">
            <div className="flex items-center gap-3 px-2">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-violet-500 to-amber-400 text-white flex items-center justify-center">
                    <Brain className="w-6 h-6" />
                </div>
                <div>
                    <div className="text-sm font-semibold">StudyGenie</div>
                    <div className="text-[11px] opacity-70">AI-Powered Learning</div>
                </div>
            </div>
            <nav className="flex-1">
                <ul className="space-y-1">
                    {items.map((it) => (
                        <li key={it.id}>
                            <button
                                onClick={() => onSelect(it.id)}
                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition border
                ${active === it.id
                                        ? "bg-black text-white border-black"
                                        : "bg-white/60 dark:bg-white/10 border-black/5 hover:bg-black/5"
                                    }`}
                            >
                                <it.icon className="w-5 h-5" />
                                <span className="text-sm font-medium">{it.label}</span>
                                {it.id === "progress" && (
                                    <span className="ml-auto text-xs opacity-70">
                                        {Math.round(((progressData?.mastered_concepts || 0) / (progressData?.total_concepts || 1)) * 100)}%
                                    </span>
                                )}
                                <ChevronRight className="w-4 h-4 ml-auto opacity-60" />
                            </button>
                        </li>
                    ))}
                </ul>
            </nav>
            <div className="mt-auto p-3 rounded-xl bg-gradient-to-br from-violet-500/15 to-amber-400/15 border border-black/5">
                <div className="text-sm font-semibold mb-1">Tip</div>
                <div className="text-xs opacity-80">Use <kbd className="px-1 rounded bg-black/10">Ctrl</kbd> + <kbd className="px-1 rounded bg-black/10">K</kbd> to open command search.</div>
            </div>
        </div>
    );
};

const EmptyState = () => (
    <div className="text-center py-16">
        <div className="inline-flex p-3 rounded-2xl bg-gradient-to-tr from-indigo-500 to-sky-400 text-white shadow-lg mb-4">
            <Sparkles className="w-6 h-6" />
        </div>
        <h4 className="text-xl font-semibold">No content yet</h4>
        <p className="text-sm opacity-70 mt-1">
            Upload a file or send a prompt to generate flashcards, quizzes, summaries and more.
        </p>
        <div className="mt-4 inline-flex items-center gap-2 text-xs opacity-70">
            <span className="rounded-full px-2 py-1 bg-black/5">"Create 10 MCQs on Algebra"</span>
            <span className="rounded-full px-2 py-1 bg-black/5">"Summarise Chapter 2 in 5 bullets"</span>
        </div>
    </div>
);

// =====================
// Root with Router (drop-in)
// =====================
const App = () => (
    <Router>
        <Routes>
            <Route path="/" element={<StudyGenieWow />} />
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    </Router>
);

// Mount
ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);