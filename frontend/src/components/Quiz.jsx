import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, ArrowRight, RotateCcw } from 'lucide-react';

const Quiz = ({ questions, onComplete, className = '' }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState('');
    const [showExplanation, setShowExplanation] = useState(false);
    const [answers, setAnswers] = useState([]);
    const [isComplete, setIsComplete] = useState(false);

    const currentQuestion = questions && questions.length > 0 ? questions[currentIndex] : null;
    const totalQuestions = questions ? questions.length : 0;

    const handleAnswerSelect = (answer) => {
        if (showExplanation) return; // Prevent changing answer after submission
        setSelectedAnswer(answer);
    };

    const handleSubmitAnswer = () => {
        if (!selectedAnswer) return;

        const isCorrect = selectedAnswer === currentQuestion.correct_answer;
        const newAnswers = [...answers, {
            questionIndex: currentIndex,
            selectedAnswer,
            correctAnswer: currentQuestion.correct_answer,
            isCorrect
        }];
        setAnswers(newAnswers);
        setShowExplanation(true);
    };

    const handleNextQuestion = () => {
        if (currentIndex < totalQuestions - 1) {
            setCurrentIndex(currentIndex + 1);
            setSelectedAnswer('');
            setShowExplanation(false);
        } else {
            setIsComplete(true);
            if (onComplete) {
                const correctCount = answers.filter(a => a.isCorrect).length + (selectedAnswer === currentQuestion.correct_answer ? 1 : 0);
                onComplete({
                    totalQuestions,
                    correctAnswers: correctCount,
                    accuracy: Math.round((correctCount / totalQuestions) * 100),
                    answers: [...answers, {
                        questionIndex: currentIndex,
                        selectedAnswer,
                        correctAnswer: currentQuestion.correct_answer,
                        isCorrect: selectedAnswer === currentQuestion.correct_answer
                    }]
                });
            }
        }
    };

    const resetQuiz = () => {
        setCurrentIndex(0);
        setSelectedAnswer('');
        setShowExplanation(false);
        setAnswers([]);
        setIsComplete(false);
    };

    if (!questions || questions.length === 0) {
        return (
            <div className="text-center py-8">
                <p className="text-gray-500">No quiz questions available</p>
            </div>
        );
    }

    if (isComplete) {
        const correctCount = answers.filter(a => a.isCorrect).length;
        const accuracy = Math.round((correctCount / totalQuestions) * 100);

        return (
            <motion.div
                className={`text-center py-8 ${className}`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
            >
                <div className="bg-white rounded-xl shadow-lg p-8 max-w-2xl mx-auto">
                    <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Check className="w-8 h-8 text-green-600" />
                    </div>

                    <h3 className="text-2xl font-bold text-gray-800 mb-2">Quiz Complete!</h3>

                    <div className="space-y-2 mb-6">
                        <p className="text-gray-600">
                            You scored <span className="font-bold text-green-600">{correctCount}</span> out of{' '}
                            <span className="font-bold">{totalQuestions}</span>
                        </p>
                        <p className="text-lg font-semibold text-gray-800">{accuracy}% Score</p>
                    </div>

                    {/* Score breakdown */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="p-3 bg-green-50 rounded-lg">
                            <p className="text-sm text-green-600 font-medium">Correct</p>
                            <p className="text-xl font-bold text-green-700">{correctCount}</p>
                        </div>
                        <div className="p-3 bg-red-50 rounded-lg">
                            <p className="text-sm text-red-600 font-medium">Incorrect</p>
                            <p className="text-xl font-bold text-red-700">{totalQuestions - correctCount}</p>
                        </div>
                        <div className="p-3 bg-blue-50 rounded-lg">
                            <p className="text-sm text-blue-600 font-medium">Accuracy</p>
                            <p className="text-xl font-bold text-blue-700">{accuracy}%</p>
                        </div>
                    </div>

                    <button
                        onClick={resetQuiz}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                        <RotateCcw className="w-4 h-4" />
                        <span>Take Quiz Again</span>
                    </button>
                </div>
            </motion.div>
        );
    }

    return (
        <div className={className}>
            {/* Progress indicator */}
            <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-600">
                        Question {currentIndex + 1} of {totalQuestions}
                    </span>
                    <span className="text-sm text-gray-500">
                        {answers.length > 0 && `${Math.round((answers.filter(a => a.isCorrect).length / answers.length) * 100)}% score`}
                    </span>
                </div>

                <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                        className="bg-blue-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${((currentIndex) / totalQuestions) * 100}%` }}
                        transition={{ duration: 0.3 }}
                    />
                </div>
            </div>

            {/* Current question */}
            <AnimatePresence mode="wait">
                {currentQuestion && (
                    <motion.div
                        key={currentIndex}
                        className="bg-white rounded-xl shadow-lg p-8"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        {/* Question */}
                        <div className="mb-6">
                            <h3 className="text-xl font-semibold text-gray-800 mb-4">
                                {currentQuestion.question}
                            </h3>
                        </div>

                        {/* Options */}
                        <div className="space-y-3 mb-6">
                            {currentQuestion.options.map((option, index) => {
                                const isSelected = selectedAnswer === option;
                                const isCorrect = option === currentQuestion.correct_answer;
                                const isIncorrect = showExplanation && isSelected && !isCorrect;
                                const shouldHighlight = showExplanation && isCorrect;

                                return (
                                    <motion.button
                                        key={index}
                                        onClick={() => handleAnswerSelect(option)}
                                        className={`
                      w-full p-4 text-left rounded-lg border-2 transition-all duration-200
                      ${isSelected && !showExplanation
                                                ? 'border-blue-500 bg-blue-50'
                                                : 'border-gray-200 hover:border-gray-300'
                                            }
                      ${shouldHighlight ? 'border-green-500 bg-green-50' : ''}
                      ${isIncorrect ? 'border-red-500 bg-red-50' : ''}
                      ${showExplanation ? 'cursor-default' : 'cursor-pointer hover:bg-gray-50'}
                    `}
                                        disabled={showExplanation}
                                        whileHover={!showExplanation ? { scale: 1.02 } : {}}
                                        whileTap={!showExplanation ? { scale: 0.98 } : {}}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="flex-1">{option}</span>
                                            {showExplanation && (
                                                <div className="ml-2">
                                                    {shouldHighlight && <Check className="w-5 h-5 text-green-500" />}
                                                    {isIncorrect && <X className="w-5 h-5 text-red-500" />}
                                                </div>
                                            )}
                                        </div>
                                    </motion.button>
                                );
                            })}
                        </div>

                        {/* Explanation */}
                        <AnimatePresence>
                            {showExplanation && (
                                <motion.div
                                    className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200"
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <h4 className="font-semibold text-blue-800 mb-2">Explanation:</h4>
                                    <p className="text-blue-700">{currentQuestion.explanation}</p>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Action buttons */}
                        <div className="flex justify-end space-x-4">
                            {!showExplanation ? (
                                <button
                                    onClick={handleSubmitAnswer}
                                    disabled={!selectedAnswer}
                                    className={`
                    px-6 py-2 rounded-lg font-medium transition-colors duration-200
                    ${selectedAnswer
                                            ? 'bg-blue-600 hover:bg-blue-700 text-white'
                                            : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                                        }
                  `}
                                >
                                    Submit Answer
                                </button>
                            ) : (
                                <button
                                    onClick={handleNextQuestion}
                                    className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2 rounded-lg transition-colors duration-200 flex items-center space-x-2"
                                >
                                    <span>{currentIndex < totalQuestions - 1 ? 'Next Question' : 'Finish Quiz'}</span>
                                    <ArrowRight className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Quiz;
