import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Check, X, Eye, EyeOff } from 'lucide-react';
import { getDifficultyColor } from '../utils';

const FlashCard = ({ card, onAnswer, showDifficulty = true, className = '' }) => {
    const [isFlipped, setIsFlipped] = useState(false);
    const [showAnswer, setShowAnswer] = useState(false);

    const handleFlip = () => {
        setIsFlipped(!isFlipped);
        setShowAnswer(!showAnswer);
    };

    const handleAnswer = (isCorrect) => {
        onAnswer(isCorrect);
        // Reset card after a short delay
        setTimeout(() => {
            setIsFlipped(false);
            setShowAnswer(false);
        }, 1000);
    };

    return (
        <motion.div
            className={`w-full max-w-md mx-auto ${className}`}
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.3 }}
        >
            <div className="flip-card h-80">
                <motion.div
                    className="flip-card-inner relative w-full h-full"
                    animate={{ rotateY: isFlipped ? 180 : 0 }}
                    transition={{ duration: 0.6, type: 'spring', stiffness: 100 }}
                    style={{ transformStyle: 'preserve-3d' }}
                >
                    {/* Front of card (Question) */}
                    <div
                        className="flip-card-front absolute inset-0 w-full h-full"
                        style={{ backfaceVisibility: 'hidden' }}
                    >
                        <div className="h-full bg-white rounded-xl shadow-lg border border-gray-200 p-6 flex flex-col">
                            {/* Header */}
                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center space-x-2">
                                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                                    <span className="text-sm font-medium text-gray-600">Question</span>
                                </div>
                                {showDifficulty && card.difficulty && (
                                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(card.difficulty)}`}>
                                        {card.difficulty}
                                    </span>
                                )}
                            </div>

                            {/* Question content */}
                            <div className="flex-1 flex items-center justify-center">
                                <p className="text-lg font-medium text-gray-800 text-center leading-relaxed">
                                    {card.question}
                                </p>
                            </div>

                            {/* Key concept */}
                            {card.key_concepts && (
                                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                                    <p className="text-sm font-medium text-blue-800 mb-1">Key Concept:</p>
                                    <p className="text-sm text-blue-700">{card.key_concepts}</p>
                                </div>
                            )}

                            {/* Action button */}
                            <button
                                onClick={handleFlip}
                                className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                            >
                                <Eye className="w-4 h-4" />
                                <span>Show Answer</span>
                            </button>
                        </div>
                    </div>

                    {/* Back of card (Answer) */}
                    <div
                        className="flip-card-back absolute inset-0 w-full h-full"
                        style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                    >
                        <div className="h-full bg-white rounded-xl shadow-lg border border-gray-200 p-6 flex flex-col">
                            {/* Header */}
                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center space-x-2">
                                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                    <span className="text-sm font-medium text-gray-600">Answer</span>
                                </div>
                                <button
                                    onClick={handleFlip}
                                    className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                                >
                                    <EyeOff className="w-4 h-4" />
                                </button>
                            </div>

                            {/* Answer content */}
                            <div className="flex-1 flex items-center justify-center">
                                <p className="text-lg font-medium text-gray-800 text-center leading-relaxed">
                                    {card.answer}
                                </p>
                            </div>

                            {/* Key concept data */}
                            {card.key_concepts_data && (
                                <div className="mt-4 p-3 bg-green-50 rounded-lg">
                                    <p className="text-sm font-medium text-green-800 mb-1">Additional Info:</p>
                                    <p className="text-sm text-green-700">{card.key_concepts_data}</p>
                                </div>
                            )}

                            {/* Answer buttons */}
                            <div className="mt-4 flex space-x-3">
                                <button
                                    onClick={() => handleAnswer(false)}
                                    className="flex-1 bg-red-100 hover:bg-red-200 text-red-700 font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                                >
                                    <X className="w-4 h-4" />
                                    <span>Incorrect</span>
                                </button>
                                <button
                                    onClick={() => handleAnswer(true)}
                                    className="flex-1 bg-green-100 hover:bg-green-200 text-green-700 font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                                >
                                    <Check className="w-4 h-4" />
                                    <span>Correct</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
};

const FlashCardDeck = ({ flashcards, onComplete, className = '' }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [answers, setAnswers] = useState([]);
    const [isComplete, setIsComplete] = useState(false);

    const currentCard = flashcards && flashcards.length > 0 ? flashcards[currentIndex] : null;
    const totalCards = flashcards ? flashcards.length : 0;

    const handleAnswer = (isCorrect) => {
        const newAnswers = [...answers, { cardIndex: currentIndex, isCorrect }];
        setAnswers(newAnswers);

        if (currentIndex < totalCards - 1) {
            setCurrentIndex(currentIndex + 1);
        } else {
            setIsComplete(true);
            if (onComplete) {
                const correctCount = newAnswers.filter(a => a.isCorrect).length;
                onComplete({
                    totalCards,
                    correctAnswers: correctCount,
                    accuracy: Math.round((correctCount / totalCards) * 100),
                    answers: newAnswers
                });
            }
        }
    };

    const resetDeck = () => {
        setCurrentIndex(0);
        setAnswers([]);
        setIsComplete(false);
    };

    if (!flashcards || flashcards.length === 0) {
        return (
            <div className="text-center py-8">
                <p className="text-gray-500">No flashcards available</p>
            </div>
        );
    }

    if (isComplete) {
        const correctCount = answers.filter(a => a.isCorrect).length;
        const accuracy = Math.round((correctCount / totalCards) * 100);

        return (
            <motion.div
                className={`text-center py-8 ${className}`}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
            >
                <div className="bg-white rounded-xl shadow-lg p-8 max-w-md mx-auto">
                    <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Check className="w-8 h-8 text-green-600" />
                    </div>

                    <h3 className="text-2xl font-bold text-gray-800 mb-2">Deck Complete!</h3>

                    <div className="space-y-2 mb-6">
                        <p className="text-gray-600">
                            You got <span className="font-bold text-green-600">{correctCount}</span> out of{' '}
                            <span className="font-bold">{totalCards}</span> correct
                        </p>
                        <p className="text-lg font-semibold text-gray-800">{accuracy}% Accuracy</p>
                    </div>

                    <button
                        onClick={resetDeck}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                        <RotateCcw className="w-4 h-4" />
                        <span>Practice Again</span>
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
                        Card {currentIndex + 1} of {totalCards}
                    </span>
                    <span className="text-sm text-gray-500">
                        {answers.length > 0 && `${Math.round((answers.filter(a => a.isCorrect).length / answers.length) * 100)}% accuracy`}
                    </span>
                </div>

                <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                        className="bg-blue-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${((currentIndex) / totalCards) * 100}%` }}
                        transition={{ duration: 0.3 }}
                    />
                </div>
            </div>

            {/* Current flashcard */}
            <AnimatePresence mode="wait">
                {currentCard && (
                    <FlashCard
                        key={currentIndex}
                        card={currentCard}
                        onAnswer={handleAnswer}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

export default FlashCardDeck;
