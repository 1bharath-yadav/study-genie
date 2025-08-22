import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
);
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
// Utility Components UI
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

const Pill = ({ icon: Icon, label, value, className = "", variant = "default" }) => {
    const variants = {
        default: "bg-white/70 dark:bg-white/10 border-black/5",
        success: "bg-emerald-100 dark:bg-emerald-900/30 border-emerald-200 text-emerald-700 dark:text-emerald-300",
        warning: "bg-amber-100 dark:bg-amber-900/30 border-amber-200 text-amber-700 dark:text-amber-300",
        info: "bg-blue-100 dark:bg-blue-900/30 border-blue-200 text-blue-700 dark:text-blue-300"
    };

    return (
        <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm border ${variants[variant]} ${className}`}>
            {Icon && <Icon className="w-4 h-4 opacity-80" />}
            <span className="opacity-70">{label}</span>
            {value !== undefined && (
                <span className="font-semibold text-gray-900 dark:text-white">{value}</span>
            )}
        </div>
    );
};

const GradientText = ({ children, gradient = "from-violet-500 via-fuchsia-500 to-amber-400" }) => (
    <span className={`bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>
        {children}
    </span>
);

// Enhanced XP Ring with animation
const XPRing = ({ value = 62, size = "md" }) => {
    const sizes = {
        sm: { w: 12, h: 12, r: 20, stroke: 6, text: 8 },
        md: { w: 14, h: 14, r: 26, stroke: 8, text: 10 },
        lg: { w: 20, h: 20, r: 38, stroke: 10, text: 14 }
    };

    const { w, h, r, stroke, text } = sizes[size];
    const C = 2 * Math.PI * r;
    const dash = Math.max(0, Math.min(100, value)) * (C / 100);

    return (
        <svg viewBox="0 0 64 64" className={`w-${w} h-${h}`}>
            <circle cx="32" cy="32" r={r} className="fill-none stroke-gray-200/60" strokeWidth={stroke / 2} />
            <motion.circle
                cx="32"
                cy="32"
                r={r}
                className="fill-none stroke-[url(#grad)]"
                strokeLinecap="round"
                strokeWidth={stroke}
                initial={{ strokeDasharray: `0 ${C}` }}
                animate={{ strokeDasharray: `${dash} ${C}` }}
                transition={{ duration: 1.5, ease: "easeOut", delay: 0.2 }}
                transform="rotate(-90 32 32)"
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
                className={`fill-gray-800 dark:fill-white font-semibold text-[${text}px]`}
            >
                {value}%
            </text>
        </svg>
    );
};

// Floating background orbs with animation
const Orbs = () => (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <motion.div
            className="absolute -top-24 -left-16 h-72 w-72 rounded-full bg-fuchsia-400/20 blur-3xl"
            animate={{
                x: [0, 30, 0],
                y: [0, -20, 0],
                scale: [1, 1.1, 1]
            }}
            transition={{
                duration: 20,
                repeat: Infinity,
                ease: "linear"
            }}
        />
        <motion.div
            className="absolute top-48 -right-12 h-72 w-72 rounded-full bg-violet-400/20 blur-3xl"
            animate={{
                x: [0, -40, 0],
                y: [0, 30, 0],
                scale: [1, 0.9, 1]
            }}
            transition={{
                duration: 25,
                repeat: Infinity,
                ease: "linear"
            }}
        />
        <motion.div
            className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-amber-400/20 blur-3xl"
            animate={{
                x: [0, 20, 0],
                y: [0, -15, 0],
                scale: [1, 1.05, 1]
            }}
            transition={{
                duration: 18,
                repeat: Infinity,
                ease: "linear"
            }}
        />
    </div>
);

// Enhanced Sidebar Component
const Sidebar = ({ isOpen, onClose, active, onSelect, progressData, isMobile }) => {
    const navigationItems = [
        { id: "learn", label: "Learn", icon: Home, description: "Start learning" },
        { id: "progress", label: "Progress", icon: BarChart3, description: "Track your growth" },
        { id: "library", label: "Library", icon: BookOpen, description: "Study materials" },
        { id: "practice", label: "Practice", icon: Target, description: "Test your skills" },
    ];

    const quickActions = [
        { id: "upload", label: "Upload File", icon: Upload, color: "from-blue-500 to-cyan-400" },
        { id: "chat", label: "AI Chat", icon: MessageCircle, color: "from-purple-500 to-pink-400" },
        { id: "quiz", label: "Quick Quiz", icon: Zap, color: "from-green-500 to-emerald-400" },
    ];

    const sidebarContent = (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-violet-500 to-amber-400 text-white flex items-center justify-center shadow-lg">
                        <Brain className="w-6 h-6" />
                    </div>
                    <div className="min-w-0">
                        <div className="text-lg font-bold">
                            <GradientText>StudyGenie</GradientText>
                        </div>
                        <div className="text-xs opacity-70">AI-Powered Learning</div>
                    </div>
                </div>
                {isMobile && (
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                )}
            </div>

            {/* User Stats */}
            <div className="p-4 border-b border-white/10">
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-500 text-white flex items-center justify-center font-bold text-lg">
                        A
                    </div>
                    <div className="min-w-0 flex-1">
                        <div className="font-semibold truncate">Alex Johnson</div>
                        <div className="text-xs opacity-70">Level 3 • 7 day streak</div>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <XPRing value={Math.round(((progressData?.mastered_concepts || 0) / (progressData?.total_concepts || 1)) * 100)} size="sm" />
                    <span className="opacity-70">Overall Progress</span>
                </div>
            </div>

            {/* Navigation */}
            <div className="flex-1 p-4 space-y-1">
                <div className="text-xs font-semibold opacity-70 mb-3 uppercase tracking-wider">Navigation</div>
                {navigationItems.map((item) => (
                    <motion.button
                        key={item.id}
                        onClick={() => onSelect(item.id)}
                        whileHover={{ x: 4 }}
                        whileTap={{ scale: 0.98 }}
                        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all border text-left ${active === item.id
                            ? "bg-black text-white border-black shadow-lg"
                            : "bg-white/60 dark:bg-white/10 border-transparent hover:bg-white/80 hover:border-white/20"
                            }`}
                    >
                        <item.icon className="w-5 h-5 flex-shrink-0" />
                        <div className="min-w-0 flex-1">
                            <div className="font-medium">{item.label}</div>
                            <div className={`text-xs ${active === item.id ? 'opacity-70' : 'opacity-50'}`}>
                                {item.description}
                            </div>
                        </div>
                        {item.id === "progress" && (
                            <span className="text-xs opacity-70 font-medium">
                                {Math.round(((progressData?.mastered_concepts || 0) / (progressData?.total_concepts || 1)) * 100)}%
                            </span>
                        )}
                    </motion.button>
                ))}
            </div>

            {/* Quick Actions */}
            <div className="p-4 border-t border-white/10">
                <div className="text-xs font-semibold opacity-70 mb-3 uppercase tracking-wider">Quick Actions</div>
                <div className="grid grid-cols-1 gap-2">
                    {quickActions.map((action) => (
                        <motion.button
                            key={action.id}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="flex items-center gap-3 p-3 rounded-xl bg-gradient-to-r text-white shadow-lg hover:shadow-xl transition-all"
                            style={{
                                background: `linear-gradient(135deg, ${action.color.split(' ')[0].replace('from-', '')} 0%, ${action.color.split(' ')[1].replace('to-', '')} 100%)`
                            }}
                        >
                            <action.icon className="w-4 h-4" />
                            <span className="text-sm font-medium">{action.label}</span>
                        </motion.button>
                    ))}
                </div>
            </div>

            {/* Bottom Tip */}
            <div className="p-4">
                <div className="p-3 rounded-xl bg-gradient-to-br from-violet-500/15 to-amber-400/15 border border-white/10">
                    <div className="flex items-center gap-2 mb-1">
                        <Sparkles className="w-4 h-4" />
                        <span className="text-sm font-semibold">Pro Tip</span>
                    </div>
                    <div className="text-xs opacity-80">
                        Use <kbd className="px-1 py-0.5 rounded bg-black/10 text-[10px]">Ctrl+K</kbd> for quick search
                    </div>
                </div>
            </div>
        </div>
    );

    return (
        <>
            {/* Mobile Sidebar */}
            <AnimatePresence>
                {isMobile && isOpen && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
                            onClick={onClose}
                        />
                        <motion.aside
                            initial={{ x: -320, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: -320, opacity: 0 }}
                            transition={{ type: "spring", stiffness: 300, damping: 30 }}
                            className="fixed inset-y-0 left-0 z-50 w-80 bg-white/90 dark:bg-black/90 backdrop-blur-xl border-r border-white/20 shadow-2xl lg:hidden"
                        >
                            {sidebarContent}
                        </motion.aside>
                    </>
                )}
            </AnimatePresence>

            {/* Desktop Sidebar */}
            <motion.aside
                animate={{ width: isOpen ? 320 : 72 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="hidden lg:flex flex-col bg-white/80 dark:bg-black/80 backdrop-blur-xl border-r border-white/20 shadow-xl h-screen sticky top-0"
            >
                {isOpen ? (
                    sidebarContent
                ) : (
                    <div className="p-4 space-y-4">
                        <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-violet-500 to-amber-400 text-white flex items-center justify-center shadow-lg">
                            <Brain className="w-6 h-6" />
                        </div>
                        {navigationItems.map((item) => (
                            <motion.button
                                key={item.id}
                                onClick={() => onSelect(item.id)}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                                className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${active === item.id
                                    ? "bg-black text-white shadow-lg"
                                    : "bg-white/60 hover:bg-white/80"
                                    }`}
                                title={item.label}
                            >
                                <item.icon className="w-5 h-5" />
                            </motion.button>
                        ))}
                    </div>
                )}
            </motion.aside>
        </>
    );
};

// Mock Study Content Sections
const StudyContentSections = ({ studyContent, expandedSection, setExpandedSection, onActivityComplete }) => {
    const sections = [
        {
            id: "summary",
            title: "Summary",
            icon: FileText,
            description: "Key points overview",
            content: studyContent?.summary
        },
        {
            id: "flashcards",
            title: "Flashcards",
            icon: Sparkles,
            description: `${studyContent?.flashcards?.length || 0} cards`,
            content: studyContent?.flashcards
        },
        {
            id: "quiz",
            title: "Quiz",
            icon: Target,
            description: `${studyContent?.quiz?.length || 0} questions`,
            content: studyContent?.quiz
        },
    ];

    return (
        <div className="space-y-4">
            {sections.map((section) => (
                <GlassCard key={section.id} className="overflow-hidden">
                    <button
                        onClick={() => setExpandedSection(expandedSection === section.id ? null : section.id)}
                        className="w-full p-4 flex items-center justify-between hover:bg-white/50 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-500 to-purple-600 text-white flex items-center justify-center">
                                <section.icon className="w-4 h-4" />
                            </div>
                            <div className="text-left">
                                <div className="font-semibold">{section.title}</div>
                                <div className="text-xs opacity-70">{section.description}</div>
                            </div>
                        </div>
                        <ChevronRight className={`w-5 h-5 transition-transform ${expandedSection === section.id ? 'rotate-90' : ''}`} />
                    </button>

                    <AnimatePresence>
                        {expandedSection === section.id && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                className="border-t border-white/20"
                            >
                                <div className="p-4">
                                    {section.id === "summary" && (
                                        <div className="prose prose-sm max-w-none">
                                            <p>{section.content || "No summary available yet."}</p>
                                        </div>
                                    )}
                                    {section.id === "flashcards" && (
                                        <div className="space-y-3">
                                            {section.content?.map((card, idx) => (
                                                <div key={idx} className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200/50">
                                                    <div className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Q: {card.question}</div>
                                                    <div className="text-blue-800 dark:text-blue-200">A: {card.answer}</div>
                                                </div>
                                            )) || <p className="text-center opacity-70">No flashcards available yet.</p>}
                                        </div>
                                    )}
                                    {section.id === "quiz" && (
                                        <div className="space-y-4">
                                            {section.content?.map((q, idx) => (
                                                <div key={idx} className="p-4 rounded-xl bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200/50">
                                                    <div className="font-semibold text-green-900 dark:text-green-100 mb-3">{idx + 1}. {q.question}</div>
                                                    <div className="space-y-2">
                                                        {q.options?.map((option, optIdx) => (
                                                            <div key={optIdx} className={`p-2 rounded-lg ${option === q.correct_answer ? 'bg-green-200 dark:bg-green-800/50' : 'bg-white/60 dark:bg-white/10'}`}>
                                                                {option}
                                                            </div>
                                                        ))}
                                                    </div>
                                                    {q.explanation && (
                                                        <div className="mt-3 p-3 rounded-lg bg-green-100 dark:bg-green-900/30 text-sm">
                                                            <strong>Explanation:</strong> {q.explanation}
                                                        </div>
                                                    )}
                                                </div>
                                            )) || <p className="text-center opacity-70">No quiz questions available yet.</p>}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </GlassCard>
            ))}
        </div>
    );
};

// Mock Progress Dashboard
const ProgressDashboard = ({ progressData }) => {
    const subjects = progressData?.subject_progress || [];
    const concepts = progressData?.concept_progress || [];

    return (
        <div className="space-y-6">
            {/* Stats Overview */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <GlassCard className="p-4 text-center" hover>
                    <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-tr from-blue-500 to-cyan-400 text-white flex items-center justify-center mb-3">
                        <Trophy className="w-6 h-6" />
                    </div>
                    <div className="text-2xl font-bold">{progressData?.mastered_concepts || 0}</div>
                    <div className="text-sm opacity-70">Mastered</div>
                </GlassCard>

                <GlassCard className="p-4 text-center" hover>
                    <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-tr from-green-500 to-emerald-400 text-white flex items-center justify-center mb-3">
                        <TrendingUp className="w-6 h-6" />
                    </div>
                    <div className="text-2xl font-bold">{progressData?.total_concepts || 0}</div>
                    <div className="text-sm opacity-70">Total Concepts</div>
                </GlassCard>

                <GlassCard className="p-4 text-center" hover>
                    <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-tr from-orange-500 to-red-400 text-white flex items-center justify-center mb-3">
                        <Target className="w-6 h-6" />
                    </div>
                    <div className="text-2xl font-bold">{progressData?.weak_concepts || 0}</div>
                    <div className="text-sm opacity-70">Need Focus</div>
                </GlassCard>

                <GlassCard className="p-4 text-center" hover>
                    <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-tr from-purple-500 to-pink-400 text-white flex items-center justify-center mb-3">
                        <Flame className="w-6 h-6" />
                    </div>
                    <div className="text-2xl font-bold">7</div>
                    <div className="text-sm opacity-70">Day Streak</div>
                </GlassCard>
            </div>

            {/* Subject Progress */}
            <GlassCard className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <BookOpen className="w-5 h-5" />
                    Subject Progress
                </h3>
                <div className="space-y-4">
                    {subjects.map((subject, idx) => (
                        <div key={idx} className="space-y-2">
                            <div className="flex justify-between items-center">
                                <span className="font-medium">{subject.subject_name}</span>
                                <span className="text-sm font-semibold">{subject.progress_percentage}%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <motion.div
                                    className="h-2 rounded-full bg-gradient-to-r from-violet-500 to-purple-600"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${subject.progress_percentage}%` }}
                                    transition={{ duration: 1, delay: idx * 0.1 }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </GlassCard>

            {/* Concept Progress */}
            <GlassCard className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Recent Concepts
                </h3>
                <div className="space-y-3">
                    {concepts.map((concept, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-white/60 dark:bg-white/5 border border-white/20">
                            <div className="flex items-center gap-3">
                                <div className={`w-3 h-3 rounded-full ${concept.status === 'mastered' ? 'bg-green-500' :
                                    concept.status === 'in_progress' ? 'bg-yellow-500' : 'bg-red-500'
                                    }`} />
                                <div>
                                    <div className="font-medium">{concept.concept_name}</div>
                                    <div className="text-sm opacity-70">{concept.attempts_count} attempts</div>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="font-semibold">{concept.mastery_score}%</div>
                                <div className="text-sm opacity-70">{concept.correct_answers}/{concept.total_questions}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </GlassCard>
        </div>
    );
};

// Empty State Component
const EmptyState = () => (
    <div className="text-center py-16">
        <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            <div className="inline-flex p-4 rounded-2xl bg-gradient-to-tr from-indigo-500 to-sky-400 text-white shadow-lg mb-4">
                <Sparkles className="w-8 h-8" />
            </div>
            <h4 className="text-xl font-semibold mb-2">Ready to start learning?</h4>
            <p className="text-sm opacity-70 mb-6 max-w-md mx-auto">
                Upload a file or chat with AI to generate personalized flashcards, quizzes, summaries and more.
            </p>
            <div className="flex flex-wrap justify-center gap-2 text-xs">
                <span className="px-3 py-1.5 rounded-full bg-black/5 border border-black/10">"Create 10 MCQs on Algebra"</span>
                <span className="px-3 py-1.5 rounded-full bg-black/5 border border-black/10">"Summarize Chapter 2"</span>
                <span className="px-3 py-1.5 rounded-full bg-black/5 border border-black/10">"Make flashcards on Physics"</span>
            </div>
        </motion.div>
    </div>
);

// Main App Component
const StudyGenieApp = () => {
    // State management
    const [currentUser] = useState(mockUser);
    const [progressData] = useState(mockProgressData);
    const [studyContent, setStudyContent] = useState(null);
    const [userPrompt, setUserPrompt] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [expandedSection, setExpandedSection] = useState(null);
    const [active, setActive] = useState("learn");
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    // Responsive handling
    useEffect(() => {
        const checkMobile = () => {
            setIsMobile(window.innerWidth < 1024);
            if (window.innerWidth >= 1024) {
                setSidebarOpen(true);
            }
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

        setTimeout(() => {
            const mockContent = {
                flashcards: [{
                    id: "1",
                    question: `What is the main concept about: ${userPrompt}?`,
                    answer: "Based on your query, here's a comprehensive answer covering the key points you need to understand.",
                    key_concepts: "Core concept identification",
                    key_concepts_data: "Detailed analysis of the main topics",
                    difficulty: "Medium",
                }],
                quiz: [{
                    id: "1",
                    question: `Which statement best describes: ${userPrompt}?`,
                    options: ["Option A - Correct answer", "Option B - Incorrect", "Option C - Incorrect", "Option D - Incorrect"],
                    correct_answer: "Option A - Correct answer",
                    explanation: "This is correct because it accurately describes the main concept.",
                }],
                summary: `Here's a comprehensive summary based on your query: "${userPrompt}". This covers the essential points and provides a clear understanding of the topic.`,
                learningObjectives: [
                    "Understand the main concepts presented",
                    "Apply knowledge in practical scenarios",
                    "Analyze and synthesize information effectively",
                ],
            };

            setStudyContent(mockContent);
            setChatHistory(prev => [
                ...prev,
                { type: "ai", content: "Study content generated from your prompt!", timestamp: new Date() }
            ]);
            setUserPrompt("");
            setIsProcessing(false);
        }, 1500);
    };

    const handleActivityComplete = (activityType, results) => {
        setChatHistory(prev => [
            ...prev,
            { type: "success", content: `${activityType} completed! Great job!`, timestamp: new Date() }
        ]);
    };

    return (
        <div className="relative min-h-screen bg-gradient-to-br from-slate-50 via-white to-violet-50 dark:from-slate-950 dark:via-slate-900 dark:to-black overflow-hidden">
            <Orbs />

            {/* Header */}
            <header className="sticky top-0 z-30 backdrop-blur-xl border-b border-white/20 bg-white/70 dark:bg-black/70">
                <div className="flex items-center justify-between px-4 lg:px-6 py-3">
                    {/* Left side */}
                    <div className="flex items-center gap-3">
                        {/* Mobile Menu Button */}
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="lg:hidden p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                            aria-label="Toggle sidebar"
                        >
                            <div className="space-y-1.5">
                                <div className="w-6 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                                <div className="w-6 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                                <div className="w-6 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                            </div>
                        </button>

                        {/* Desktop Sidebar Toggle */}
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="hidden lg:flex p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                            aria-label="Toggle sidebar"
                        >
                            <div className="space-y-1">
                                <div className="w-5 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                                <div className="w-5 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                                <div className="w-5 h-0.5 bg-gray-600 dark:bg-gray-400"></div>
                            </div>
                        </button>

                        {/* Brand */}
                        <div className="flex items-center gap-2">
                            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-500 to-amber-400 text-white flex items-center justify-center shadow-lg">
                                <Brain className="w-5 h-5" />
                            </div>
                            <h1 className="hidden sm:block font-bold text-xl">
                                <GradientText>StudyGenie</GradientText>
                            </h1>
                        </div>
                    </div>

                    {/* Center - Quick Stats */}
                    <div className="hidden md:flex items-center gap-3">
                        <Pill icon={Flame} label="Streak" value={7} variant="warning" />
                        <Pill icon={Trophy} label="Level" value="3" variant="success" />
                        <Pill icon={Star} label="XP" value="1,247" variant="info" />
                    </div>

                    {/* Right side */}
                    <div className="flex items-center gap-3">
                        {/* Search shortcut */}
                        <GlassCard className="hidden sm:flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-white/80 transition-colors">
                            <Search className="w-4 h-4 opacity-70" />
                            <span className="text-xs opacity-70">Search</span>
                            <kbd className="text-xs font-semibold px-1.5 py-0.5 rounded bg-black/10">⌘K</kbd>
                        </GlassCard>

                        {/* Notifications */}
                        <button className="relative p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                            <Bell className="w-5 h-5" />
                            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-gradient-to-r from-red-500 to-pink-500 animate-pulse" />
                        </button>

                        {/* Settings */}
                        <button className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                            <Settings className="w-5 h-5" />
                        </button>

                        {/* User Profile */}
                        <div className="flex items-center gap-3 pl-3 border-l border-white/20">
                            <div className="hidden sm:block text-right">
                                <div className="font-semibold text-sm">{currentUser.full_name}</div>
                                <div className="text-xs opacity-70">Level 3 Explorer</div>
                            </div>
                            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 text-white flex items-center justify-center font-bold text-sm">
                                {currentUser.full_name.charAt(0)}
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <div className="flex min-h-screen">
                {/* Sidebar */}
                <Sidebar
                    isOpen={sidebarOpen}
                    onClose={() => setSidebarOpen(false)}
                    active={active}
                    onSelect={setActive}
                    progressData={progressData}
                    isMobile={isMobile}
                />

                {/* Main Content */}
                <main className="flex-1 min-w-0">
                    <div className="p-4 lg:p-6 space-y-6">
                        {/* Welcome Banner */}
                        <GlassCard className="p-6" hover>
                            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                                <div className="space-y-2">
                                    <h2 className="text-2xl lg:text-3xl font-bold">
                                        Welcome back, <GradientText>{currentUser.full_name.split(' ')[0]}</GradientText>! 👋
                                    </h2>
                                    <p className="opacity-70">
                                        Ready to supercharge your learning journey today? Let's make it awesome!
                                    </p>
                                </div>
                                <div className="flex items-center gap-6">
                                    <div className="text-center">
                                        <XPRing
                                            value={Math.round(((progressData?.mastered_concepts || 0) / (progressData?.total_concepts || 1)) * 100)}
                                            size="lg"
                                        />
                                        <div className="text-xs opacity-70 mt-2">Overall Progress</div>
                                    </div>
                                    <div className="hidden sm:grid grid-cols-2 gap-4 text-center">
                                        <div>
                                            <div className="text-2xl font-bold">{progressData?.mastered_concepts}</div>
                                            <div className="text-xs opacity-70">Mastered</div>
                                        </div>
                                        <div>
                                            <div className="text-2xl font-bold">{progressData?.total_concepts}</div>
                                            <div className="text-xs opacity-70">Total</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </GlassCard>

                        {/* Content Area */}
                        {active === "progress" ? (
                            <GlassCard className="p-6">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-500 to-cyan-400 text-white flex items-center justify-center">
                                        <BarChart3 className="w-6 h-6" />
                                    </div>
                                    <div>
                                        <h3 className="text-xl font-bold">Progress Analytics</h3>
                                        <p className="text-sm opacity-70">Track your learning journey</p>
                                    </div>
                                </div>
                                <ProgressDashboard progressData={progressData} />
                            </GlassCard>
                        ) : (
                            <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6">
                                {/* Left Panel */}
                                <div className="space-y-6">
                                    {/* File Upload */}
                                    <GlassCard className="p-6">
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-sky-400 text-white flex items-center justify-center">
                                                    <Upload className="w-6 h-6" />
                                                </div>
                                                <div>
                                                    <h3 className="font-bold">Upload Materials</h3>
                                                    <p className="text-xs opacity-70">AI-powered content generation</p>
                                                </div>
                                            </div>
                                            <Pill icon={Rocket} label="AI Boost" variant="info" />
                                        </div>
                                        <FileUpload onFileProcessed={handleFileProcessed} isProcessing={isProcessing} />
                                    </GlassCard>

                                    {/* AI Chat */}
                                    <GlassCard className="p-6">
                                        <div className="flex items-center gap-3 mb-4">
                                            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-500 to-orange-400 text-white flex items-center justify-center">
                                                <Bot className="w-6 h-6" />
                                            </div>
                                            <div>
                                                <h3 className="font-bold">AI Learning Assistant</h3>
                                                <p className="text-xs opacity-70">Chat, create, and learn</p>
                                            </div>
                                        </div>
                                        <ChatBox
                                            chatHistory={chatHistory}
                                            userPrompt={userPrompt}
                                            setUserPrompt={setUserPrompt}
                                            onPromptSubmit={handlePromptSubmit}
                                            isProcessing={isProcessing}
                                        />
                                        <div className="mt-4 p-3 rounded-xl bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 border border-violet-200/50">
                                            <div className="flex items-start gap-2 text-sm">
                                                <HelpCircle className="w-4 h-4 mt-0.5 text-violet-600" />
                                                <div>
                                                    <div className="font-medium text-violet-900 dark:text-violet-100 mb-1">Try these prompts:</div>
                                                    <div className="text-xs space-y-1 opacity-80">
                                                        <div>• "Create 5 flashcards on photosynthesis"</div>
                                                        <div>• "Generate a quiz about ancient history"</div>
                                                        <div>• "Summarize this chapter in bullet points"</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </GlassCard>
                                </div>

                                {/* Right Panel - Study Content */}
                                <div>
                                    <GlassCard className="p-6">
                                        <div className="flex items-center justify-between mb-6">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-500 to-purple-600 text-white flex items-center justify-center">
                                                    <Sparkles className="w-6 h-6" />
                                                </div>
                                                <div>
                                                    <h3 className="font-bold text-xl">Learning Canvas</h3>
                                                    <p className="text-sm opacity-70">Your personalized study content</p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Pill icon={Star} label="Focus Mode" variant="success" />
                                                <Pill icon={Crown} label="Pro" variant="warning" />
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
                </main>
            </div>
        </div>
    );
};

export default StudyGenieApp;