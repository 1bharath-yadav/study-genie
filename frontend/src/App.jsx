import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
    BookOpen,
    Target,
    Zap,
    Users,
    Calendar,
    Award,
    TrendingUp,
    Clock,
    PlayCircle,
    FileText,
    MessageCircle,
    Search
} from "lucide-react";

// Import our components
import FileUpload from "./components/FileUpload";
import FlashCardDeck from "./components/FlashCardDeck";
import Quiz from "./components/Quiz";
import MatchTheFollowing from "./components/MatchTheFollowing";
import ProgressDashboard from "./components/ProgressDashboard";
import ChatBox from "./components/ChatBox";
import StudyContentSections from "./components/StudyContentSections";
import LoginForm from "./components/LoginForm";
import Loading from "./components/Loading";

// Import API service
import * as apiService from "./services/api";

// =====================
// Mock Data
// =====================
const mockProgressData = {
    student_id: "demo-student",
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
};

const mockUser = {
    student_id: "demo-student",
    username: "demo_user",
    email: "demo@studygenie.com",
    full_name: "Alex Johnson",
};

// =====================
// Utility Components
// =====================
const GlassCard = ({ className = "", children, hover = false }) => (
    <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        whileHover={hover ? { y: -2, scale: 1.01 } : {}}
        className={`rounded-2xl border border-white/20 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-[0_8px_30px_rgba(0,0,0,0.08)] ${className}`}
    >
        {children}
    </motion.div>
);

const GradientText = ({ children, gradient = "from-violet-500 via-fuchsia-500 to-amber-400" }) => (
    <span className={`bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>
        {children}
    </span>
);

const Orbs = () => (
    <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-40 -right-32 w-80 h-80 bg-gradient-to-br from-purple-400 via-violet-400 to-indigo-600 rounded-full opacity-20 blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-32 w-80 h-80 bg-gradient-to-tr from-cyan-400 via-blue-500 to-indigo-600 rounded-full opacity-20 blur-3xl animate-pulse" style={{ animationDelay: "2s" }} />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-pink-400 via-purple-500 to-violet-600 rounded-full opacity-10 blur-3xl animate-pulse" style={{ animationDelay: "4s" }} />
    </div>
);

const Sidebar = ({ isOpen, onClose, active, onSelect, progressData, isMobile }) => {
    const mainMenuItems = [
        { id: "upload", label: "Upload Files", icon: Upload },
        { id: "chat", label: "AI Chat", icon: MessageSquareText },
    ];

    const extendedMenuItems = [
        { id: "progress", label: "Progress", icon: BarChart3 },
        { id: "settings", label: "Settings", icon: Settings },
    ];

    return (
        <>
            {/* Overlay for mobile */}
            <AnimatePresence>
                {isMobile && isOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
                        onClick={onClose}
                    />
                )}
            </AnimatePresence>

            {/* Main Sidebar - Always visible on desktop */}
            <motion.aside
                initial={isMobile ? { x: -80 } : false}
                animate={{ x: 0 }}
                className="fixed top-0 left-0 h-screen bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-r border-white/20 z-50 w-20"
            >
                <div className="p-4">
                    {/* Logo */}
                    <div className="flex items-center justify-center mb-8">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-500 to-purple-600 text-white flex items-center justify-center">
                            <Sparkles className="w-6 h-6" />
                        </div>
                    </div>

                    {/* Main Menu Items */}
                    <nav className="space-y-4">
                        {mainMenuItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => onSelect(item.id)}
                                className={`w-12 h-12 flex items-center justify-center rounded-xl transition-all ${active === item.id
                                    ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-lg"
                                    : "hover:bg-white/60 dark:hover:bg-white/10"
                                    }`}
                                title={item.label}
                            >
                                <item.icon className="w-5 h-5" />
                            </button>
                        ))}

                        {/* Menu Toggle Button */}
                        <button
                            onClick={() => onSelect(isOpen ? null : "menu")}
                            className="w-12 h-12 flex items-center justify-center rounded-xl hover:bg-white/60 dark:hover:bg-white/10 transition-all"
                            title="More Options"
                        >
                            <Menu className="w-5 h-5" />
                        </button>
                    </nav>
                </div>
            </motion.aside>

            {/* Extended Sidebar - Toggleable */}
            <AnimatePresence>
                {isOpen && (
                    <motion.aside
                        initial={{ x: -240 }}
                        animate={{ x: 80 }}
                        exit={{ x: -240 }}
                        transition={{ type: "spring", damping: 25, stiffness: 200 }}
                        className="fixed top-0 left-0 h-screen bg-white/90 dark:bg-gray-900/90 backdrop-blur-xl border-r border-white/20 z-40 w-60"
                    >
                        <div className="p-6 pt-20">
                            <div className="flex items-center justify-between mb-6">
                                <GradientText className="text-lg font-bold">StudyGenie</GradientText>
                                <button
                                    onClick={onClose}
                                    className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <nav className="space-y-2">
                                {extendedMenuItems.map((item) => (
                                    <button
                                        key={item.id}
                                        onClick={() => {
                                            onSelect(item.id);
                                            if (isMobile) onClose();
                                        }}
                                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${active === item.id
                                            ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-lg"
                                            : "hover:bg-white/60 dark:hover:bg-white/10"
                                            }`}
                                    >
                                        <item.icon className="w-5 h-5" />
                                        <span className="font-medium">{item.label}</span>
                                    </button>
                                ))}
                            </nav>
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>
        </>
    );
};

const EmptyState = () => (
    <GlassCard className="p-12 text-center">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-tr from-violet-500 to-purple-600 text-white flex items-center justify-center mb-6">
            <Upload className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-semibold mb-2">Ready to Learn?</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
            Upload your study materials to get started with AI-powered learning.
        </p>
    </GlassCard>
);

// =====================
// Main App Component
// =====================
const StudyGenieApp = () => {
    // State management
    const [currentUser] = useState(mockUser);
    const [progressData] = useState(mockProgressData);
    const [studyContent, setStudyContent] = useState(null);
    const [userPrompt, setUserPrompt] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [expandedSection, setExpandedSection] = useState(null);
    const [active, setActive] = useState("upload");
    const [extendedMenuOpen, setExtendedMenuOpen] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    // Responsive handling
    useEffect(() => {
        const checkMobile = () => {
            setIsMobile(window.innerWidth < 1024);
        };

        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    // Handlers
    const handleFileProcessed = async (fileData) => {
        setIsProcessing(true);
        setChatHistory(prev => [
            ...prev,
            { type: "file", content: `File uploaded: ${fileData.fileName || 'Unknown file'}`, timestamp: new Date() }
        ]);

        try {
            // fileData should contain the response from our API
            const content = fileData.enhanced_response || fileData.enhancedResponse;

            if (content) {
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
                setChatHistory(prev => [
                    ...prev,
                    { type: "success", content: "Study content generated successfully!", timestamp: new Date() }
                ]);
            } else {
                setChatHistory(prev => [
                    ...prev,
                    { type: "error", content: "Failed to process file. Please try again.", timestamp: new Date() }
                ]);
            }
        } catch (error) {
            console.error("Error processing file:", error);
            setChatHistory(prev => [
                ...prev,
                { type: "error", content: "An error occurred while processing the file.", timestamp: new Date() }
            ]);
        } finally {
            setIsProcessing(false);
        }
    };

    const handlePromptSubmit = async () => {
        if (!userPrompt.trim()) return;

        setIsProcessing(true);
        setChatHistory(prev => [
            ...prev,
            { type: "user", content: userPrompt, timestamp: new Date() }
        ]);

        try {
            // TODO: Implement actual chat API call
            // For now, simulate a response
            setTimeout(() => {
                setChatHistory(prev => [
                    ...prev,
                    { type: "assistant", content: `I understand you're asking about: "${userPrompt}". Let me help you with that concept.`, timestamp: new Date() }
                ]);
                setIsProcessing(false);
            }, 1000);
        } catch (error) {
            setChatHistory(prev => [
                ...prev,
                { type: "error", content: "Failed to get AI response. Please try again.", timestamp: new Date() }
            ]);
            setIsProcessing(false);
        }

        setUserPrompt("");
    };

    const handleActivityComplete = (activityType, score) => {
        setChatHistory(prev => [
            ...prev,
            {
                type: "activity",
                content: `Completed ${activityType} with score: ${score}%`,
                timestamp: new Date()
            }
        ]);
    };

    const handleMenuSelection = (selection) => {
        if (selection === "menu") {
            setExtendedMenuOpen(!extendedMenuOpen);
        } else if (selection === "progress" || selection === "settings") {
            setActive(selection);
            setExtendedMenuOpen(false);
        } else {
            setActive(selection);
        }
    };

    const renderSidebarContent = () => {
        switch (active) {
            case "upload":
                return (
                    <div className="space-y-6">
                        <h2 className="text-lg font-bold text-gray-800 dark:text-white">Upload Files</h2>
                        <FileUpload onFileProcessed={handleFileProcessed} isProcessing={isProcessing} />
                    </div>
                );

            case "chat":
                return (
                    <div className="space-y-6">
                        <h2 className="text-lg font-bold text-gray-800 dark:text-white">AI Study Assistant</h2>
                        <ChatBox
                            chatHistory={chatHistory}
                            userPrompt={userPrompt}
                            setUserPrompt={setUserPrompt}
                            onPromptSubmit={handlePromptSubmit}
                            isProcessing={isProcessing}
                        />
                    </div>
                );

            default:
                return null;
        }
    };

    const renderMainContent = () => {
        return (
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold">
                        <GradientText>Study Materials</GradientText>
                    </h1>
                </div>

                {studyContent ? (
                    <div className="space-y-6">
                        {/* Flash Cards Section */}
                        <GlassCard className="p-6">
                            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                <Sparkles className="w-6 h-6 text-violet-500" />
                                Flash Cards ({studyContent.flashcards?.length || 0})
                            </h2>
                            {studyContent.flashcards && studyContent.flashcards.length > 0 ? (
                                <FlashCardDeck
                                    flashcards={studyContent.flashcards}
                                    onActivityComplete={handleActivityComplete}
                                />
                            ) : (
                                <p className="text-center opacity-70">No flashcards available yet.</p>
                            )}
                        </GlassCard>

                        {/* Quiz Section */}
                        <GlassCard className="p-6">
                            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                <Target className="w-6 h-6 text-violet-500" />
                                Quiz ({studyContent.quiz?.length || 0} questions)
                            </h2>
                            {studyContent.quiz && studyContent.quiz.length > 0 ? (
                                <Quiz
                                    questions={studyContent.quiz}
                                    onActivityComplete={handleActivityComplete}
                                />
                            ) : (
                                <p className="text-center opacity-70">No quiz questions available yet.</p>
                            )}
                        </GlassCard>

                        {/* Match the Following Section */}
                        {studyContent.matchTheFollowing && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                    <Zap className="w-6 h-6 text-violet-500" />
                                    Match the Following
                                </h2>
                                <MatchTheFollowing
                                    data={studyContent.matchTheFollowing}
                                    onActivityComplete={handleActivityComplete}
                                />
                            </GlassCard>
                        )}

                        {/* Learning Objectives */}
                        {studyContent.learningObjectives && studyContent.learningObjectives.length > 0 && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                    <BookOpen className="w-6 h-6 text-violet-500" />
                                    Learning Objectives
                                </h2>
                                <ul className="space-y-2">
                                    {studyContent.learningObjectives.map((objective, idx) => (
                                        <li key={idx} className="flex items-start gap-3">
                                            <span className="w-6 h-6 rounded-full bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400 flex items-center justify-center text-sm font-medium mt-0.5">
                                                {idx + 1}
                                            </span>
                                            <span className="text-gray-700 dark:text-gray-300">{objective}</span>
                                        </li>
                                    ))}
                                </ul>
                            </GlassCard>
                        )}

                        {/* Summary Section - Moved to bottom */}
                        <GlassCard className="p-6">
                            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                <FileText className="w-6 h-6 text-violet-500" />
                                Summary
                            </h2>
                            <div className="prose prose-sm max-w-none text-gray-700 dark:text-gray-300">
                                <p>{studyContent.summary || "No summary available yet."}</p>
                            </div>
                        </GlassCard>
                    </div>
                ) : (
                    <EmptyState />
                )}
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-blue-900 dark:to-indigo-900">
            <Orbs />

            <Sidebar
                isOpen={extendedMenuOpen}
                onClose={() => setExtendedMenuOpen(false)}
                active={active}
                onSelect={handleMenuSelection}
                progressData={progressData}
                isMobile={isMobile}
            />

            {/* Main layout */}
            <div className="flex min-h-screen">
                {/* Left sidebar for upload/chat - only show when not on progress page */}
                {active !== 'progress' && active !== 'settings' && (
                    <div
                        className={`w-80 bg-white/60 dark:bg-gray-800/60 backdrop-blur-xl border-r border-white/20 flex-shrink-0 transition-all duration-300 ${isMobile ? 'hidden' : 'ml-20'
                            }`}
                    >
                        <div className="p-6 h-full overflow-y-auto">
                            {renderSidebarContent()}
                        </div>
                    </div>
                )}

                {/* Main content area */}
                <div className={`flex-1 p-6 ${isMobile ? 'ml-20' : active === 'progress' || active === 'settings' ? 'ml-20' : ''}`}>
                    <div className="max-w-6xl mx-auto">
                        {isMobile && (
                            <div className="flex items-center justify-between mb-6">
                                <button
                                    onClick={() => setExtendedMenuOpen(true)}
                                    className="p-2 hover:bg-white/60 dark:hover:bg-white/10 rounded-lg"
                                >
                                    <Menu className="w-6 h-6" />
                                </button>
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-violet-500 to-purple-600 text-white flex items-center justify-center">
                                        <Sparkles className="w-5 h-5" />
                                    </div>
                                    <GradientText>StudyGenie</GradientText>
                                </div>
                            </div>
                        )}

                        {/* Conditional content rendering */}
                        {active === 'progress' ? (
                            <div className="space-y-6">
                                <h1 className="text-3xl font-bold">
                                    <GradientText>Your Progress</GradientText>
                                </h1>
                                <ProgressDashboard progressData={progressData} />
                            </div>
                        ) : active === 'settings' ? (
                            <div className="space-y-6">
                                <h1 className="text-3xl font-bold">
                                    <GradientText>Settings</GradientText>
                                </h1>
                                <GlassCard className="p-6">
                                    <p className="text-center opacity-70">Settings panel coming soon...</p>
                                </GlassCard>
                            </div>
                        ) : (
                            renderMainContent()
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudyGenieApp;
