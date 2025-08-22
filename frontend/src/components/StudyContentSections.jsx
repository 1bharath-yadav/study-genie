import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight, BookOpen, Brain, Target } from 'lucide-react';
import FlashCardDeck from './FlashCardDeck';
import Quiz from './Quiz';
import MatchTheFollowing from './MatchTheFollowing';

const StudyContentSections = ({
    studyContent,
    expandedSection,
    setExpandedSection,
    onActivityComplete
}) => {
    const sections = [
        {
            id: 'flashcards',
            title: 'Flashcards',
            icon: BookOpen,
            count: studyContent?.flashcards?.length || 0,
            component: FlashCardDeck,
            props: {
                flashcards: studyContent?.flashcards || [],
                onComplete: (results) => onActivityComplete('flashcard_practice', results)
            }
        },
        {
            id: 'quiz',
            title: 'Quiz',
            icon: Brain,
            count: studyContent?.quiz?.length || 0,
            component: Quiz,
            props: {
                questions: studyContent?.quiz || [],
                onComplete: (results) => onActivityComplete('quiz_attempt', results)
            }
        },
        {
            id: 'match',
            title: 'Match the Following',
            icon: Target,
            count: studyContent?.matchTheFollowing?.columnA?.length || 0,
            component: MatchTheFollowing,
            props: {
                matchData: studyContent?.matchTheFollowing,
                onComplete: (results) => onActivityComplete('concept_review', results)
            }
        }
    ];

    const toggleSection = (sectionId) => {
        setExpandedSection(expandedSection === sectionId ? null : sectionId);
    };

    return (
        <div className="h-full space-y-4">
            {sections.map((section) => {
                const Icon = section.icon;
                const isExpanded = expandedSection === section.id;
                const hasContent = section.count > 0;

                return (
                    <motion.div
                        key={section.id}
                        className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
                        layout
                    >
                        {/* Section Header */}
                        <button
                            onClick={() => toggleSection(section.id)}
                            className={`w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors ${!hasContent ? 'opacity-50 cursor-not-allowed' : ''
                                }`}
                            disabled={!hasContent}
                        >
                            <div className="flex items-center space-x-3">
                                <Icon className="w-5 h-5 text-blue-600" />
                                <div className="text-left">
                                    <h3 className="text-lg font-semibold text-gray-800">
                                        {section.title}
                                    </h3>
                                    <p className="text-sm text-gray-600">
                                        {hasContent
                                            ? `${section.count} ${section.title.toLowerCase()} available`
                                            : 'No content available'
                                        }
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                {hasContent && (
                                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm font-medium">
                                        {section.count}
                                    </span>
                                )}
                                {hasContent && (
                                    isExpanded ? (
                                        <ChevronDown className="w-5 h-5 text-gray-400" />
                                    ) : (
                                        <ChevronRight className="w-5 h-5 text-gray-400" />
                                    )
                                )}
                            </div>
                        </button>

                        {/* Section Content */}
                        <AnimatePresence>
                            {isExpanded && hasContent && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.3 }}
                                    className="border-t border-gray-200"
                                >
                                    <div className="p-4">
                                        <section.component {...section.props} />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                );
            })}

            {/* Summary Section */}
            {studyContent?.summary && (
                <motion.div
                    className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-100"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Summary</h3>
                    <p className="text-gray-700 leading-relaxed">{studyContent.summary}</p>

                    {studyContent.learningObjectives?.length > 0 && (
                        <div className="mt-4">
                            <h4 className="text-md font-medium text-gray-800 mb-2">Learning Objectives</h4>
                            <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                                {studyContent.learningObjectives.map((objective, index) => (
                                    <li key={index}>{objective}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </motion.div>
            )}

            {/* No Content State */}
            {!studyContent && (
                <div className="h-full flex items-center justify-center bg-white rounded-lg shadow-sm border border-gray-200">
                    <div className="text-center py-12">
                        <BookOpen className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                        <h3 className="text-xl font-semibold text-gray-800 mb-2">No Study Content Yet</h3>
                        <p className="text-gray-600">
                            Upload a file or ask a question to generate flashcards, quizzes, and more!
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default StudyContentSections;
