import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Sparkles,
    Upload,
    BarChart3,
    Settings,
    Menu,
    MessageCircle,
    X
} from "lucide-react";

import FileUpload from "./components/FileUpload";
import FlashCardDeck from "./components/FlashCardDeck";
import Quiz from "./components/Quiz";
import MatchTheFollowing from "./components/MatchTheFollowing";
import ProgressDashboard from "./components/ProgressDashboard";
import LoginForm from "./components/LoginForm";
import Loading from "./components/Loading";

const mockProgressData = {
    student_id: "demo-student",
    total_concepts: 25,
    mastered_concepts: 8,
    weak_concepts: 3,
    subject_progress: [
        { subject_name: "Mathematics", progress_percentage: 75 },
        { subject_name: "Physics", progress_percentage: 60 },
        { subject_name: "Chemistry", progress_percentage: 85 },
    ],
};

const mockUser = {
    student_id: 1,
    username: "demo_user",
    email: "demo@studygenie.com",
    full_name: "Alex Johnson",
};

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

const GradientText = ({ children }) => (
    <span className="bg-gradient-to-r from-violet-500 via-fuchsia-500 to-amber-400 bg-clip-text text-transparent">
        {children}
    </span>
);

const Orbs = () => (
    <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-40 -right-32 w-80 h-80 bg-gradient-to-br from-purple-400 via-violet-400 to-indigo-600 rounded-full opacity-20 blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-32 w-80 h-80 bg-gradient-to-tr from-cyan-400 via-blue-500 to-indigo-600 rounded-full opacity-20 blur-3xl animate-pulse" style={{ animationDelay: "2s" }} />
    </div>
);

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

const StudyGenieApp = () => {
    const [progressData] = useState(mockProgressData);
    const [studyContent, setStudyContent] = useState(null);
    const [active, setActive] = useState("upload");
    const [isProcessing, setIsProcessing] = useState(false);
    const [isMobile, setIsMobile] = useState(false);
    const [showAssistant, setShowAssistant] = useState(false);

    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth < 1024);
        checkMobile();
        window.addEventListener("resize", checkMobile);
        return () => window.removeEventListener("resize", checkMobile);
    }, []);

    const handleFileProcessed = async (fileData) => {
        setIsProcessing(true);
        try {
            const enhancedResponse = fileData.enhanced_response;
            if (enhancedResponse && (enhancedResponse.flashcards || enhancedResponse.quiz || enhancedResponse.match_the_following)) {
                const flashcardsArr = Object.entries(enhancedResponse.flashcards || {}).map(([id, card]) => ({
                    id,
                    question: card.question,
                    answer: card.answer,
                    difficulty: card.difficulty,
                }));

                const quizArr = Object.entries(enhancedResponse.quiz || {}).map(([id, q]) => ({
                    id,
                    question: q.question,
                    options: q.options,
                    correct_answer: q.correct_answer,
                }));

                setStudyContent({
                    flashcards: flashcardsArr,
                    quiz: quizArr,
                    matchTheFollowing: enhancedResponse.match_the_following || [],
                    summary: enhancedResponse.summary || "",
                    learningObjectives: enhancedResponse.learning_objectives || [],
                });
            }
        } catch (error) {
            console.error("Error processing file:", error);
        } finally {
            setIsProcessing(false);
        }
    };

    const renderTopMenu = () => (
        <div className="fixed top-0 left-0 w-full z-50 bg-white/70 dark:bg-gray-900/70 backdrop-blur-xl border-b border-white/20 shadow-lg">
            <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-3">
                {/* Left: Logo */}
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-violet-500 to-purple-600 flex items-center justify-center">
                        <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <h1 className="font-bold text-lg">
                        <GradientText>StudyGenie</GradientText>
                    </h1>
                </div>

                {/* Center: Navigation */}
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => setActive("upload")}
                        className={`px-4 py-2 rounded-xl flex items-center gap-2 transition-all ${active === "upload"
                            ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow"
                            : "hover:bg-white/60 dark:hover:bg-white/10"
                            }`}
                    >
                        <Upload className="w-4 h-4" /> Upload
                    </button>
                    <button
                        onClick={() => setActive("progress")}
                        className={`px-4 py-2 rounded-xl flex items-center gap-2 transition-all ${active === "progress"
                            ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow"
                            : "hover:bg-white/60 dark:hover:bg-white/10"
                            }`}
                    >
                        <BarChart3 className="w-4 h-4" /> Progress
                    </button>
                    <button
                        onClick={() => setActive("settings")}
                        className={`px-4 py-2 rounded-xl flex items-center gap-2 transition-all ${active === "settings"
                            ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow"
                            : "hover:bg-white/60 dark:hover:bg-white/10"
                            }`}
                    >
                        <Settings className="w-4 h-4" /> Settings
                    </button>
                </div>
            </div>
        </div>
    );

    const renderMainContent = () => {
        if (active === "progress") {
            return (
                <div className="space-y-6">
                    <h1 className="text-3xl font-bold">
                        <GradientText>Your Progress</GradientText>
                    </h1>
                    <ProgressDashboard progressData={progressData} />
                </div>
            );
        }

        if (active === "settings") {
            return (
                <GlassCard className="p-6">
                    <p className="text-center opacity-70">Settings panel coming soon...</p>
                </GlassCard>
            );
        }

        return (
            <div className="space-y-6">
                <h1 className="text-3xl font-bold">
                    <GradientText>Study Materials</GradientText>
                </h1>
                {studyContent ? (
                    <div className="space-y-6">
                        {studyContent.summary && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4">Summary</h2>
                                <p className="text-gray-800 dark:text-gray-200 whitespace-pre-line">{studyContent.summary}</p>
                            </GlassCard>
                        )}
                        {studyContent.learningObjectives && studyContent.learningObjectives.length > 0 && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4">Learning Objectives</h2>
                                <ul className="list-disc pl-6 text-gray-800 dark:text-gray-200">
                                    {studyContent.learningObjectives.map((obj, idx) => (
                                        <li key={idx}>{obj}</li>
                                    ))}
                                </ul>
                            </GlassCard>
                        )}
                        {studyContent.flashcards && studyContent.flashcards.length > 0 && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4">Flash Cards</h2>
                                <FlashCardDeck flashcards={studyContent.flashcards} />
                            </GlassCard>
                        )}
                        {studyContent.quiz && studyContent.quiz.length > 0 && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4">Quiz</h2>
                                <Quiz questions={studyContent.quiz} />
                            </GlassCard>
                        )}
                        {studyContent.matchTheFollowing && studyContent.matchTheFollowing.length > 0 && (
                            <GlassCard className="p-6">
                                <h2 className="text-xl font-semibold mb-4">Match the Following</h2>
                                <MatchTheFollowing data={studyContent.matchTheFollowing} />
                            </GlassCard>
                        )}
                    </div>
                ) : (
                    <div className="max-w-lg">
                        <EmptyState />
                    </div>
                )}
            </div>
        );
    };

    const FloatingBubble = () => (
        <div className="fixed bottom-6 right-6 z-50">
            <motion.button
                onClick={() => setShowAssistant(!showAssistant)}
                className="relative w-16 h-16 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl flex items-center justify-center transition-all duration-300 group"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                initial={{ opacity: 0, scale: 0, y: 100 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{
                    delay: 0.5,
                    type: "spring",
                    stiffness: 200,
                    damping: 20
                }}
            >
                <motion.div
                    animate={{ rotate: showAssistant ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                >
                    {showAssistant ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
                </motion.div>

                {/* Pulse animation when closed */}
                {!showAssistant && (
                    <motion.div
                        className="absolute inset-0 rounded-full bg-gradient-to-r from-violet-500 to-purple-600 opacity-20"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ duration: 3, repeat: Infinity }}
                        style={{ zIndex: -1 }}
                    />
                )}

                {/* Notification badge */}
                {!showAssistant && studyContent && (
                    <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-semibold"
                    >
                        !
                    </motion.div>
                )}

                {/* Hover tooltip */}
                <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-black/80 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
                    {showAssistant ? 'Close Assistant' : 'Open AI Assistant'}
                </div>
            </motion.button>
        </div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100 dark:from-gray-900 dark:via-blue-900 dark:to-indigo-900">
            <Orbs />
            {renderTopMenu()}
            <div className="pt-24 px-6 max-w-7xl mx-auto">
                {renderMainContent()}
            </div>

            {/* Floating Bubble */}
            <FloatingBubble />

            {/* Assistant Component */}
            <AnimatePresence>
                {showAssistant && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.3, type: "spring", stiffness: 300, damping: 30 }}
                    >
                        <FileUpload
                            studentId={mockUser.student_id}
                            onFileProcessed={handleFileProcessed}
                            isProcessing={isProcessing}
                            hasContent={!!studyContent}
                            onClose={() => setShowAssistant(false)}
                        />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default StudyGenieApp;
