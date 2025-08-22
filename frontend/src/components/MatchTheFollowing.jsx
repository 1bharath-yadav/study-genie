import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shuffle, Check, X, RotateCcw } from 'lucide-react';

const MatchTheFollowing = ({ matchData, onComplete, className = '' }) => {
    const [columnAItems, setColumnAItems] = useState([]);
    const [columnBItems, setColumnBItems] = useState([]);
    const [matches, setMatches] = useState({});
    const [selectedA, setSelectedA] = useState(null);
    const [selectedB, setSelectedB] = useState(null);
    const [showResults, setShowResults] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (matchData) {
            // Shuffle column B items to make it challenging
            const shuffledB = [...matchData.columnB].sort(() => Math.random() - 0.5);
            setColumnAItems(matchData.columnA);
            setColumnBItems(shuffledB);
        }
    }, [matchData]);

    const handleItemClick = (item, column) => {
        if (showResults) return;

        if (column === 'A') {
            setSelectedA(selectedA === item ? null : item);
            setSelectedB(null);
        } else {
            if (selectedA) {
                // Create a match
                const newMatches = { ...matches, [selectedA]: item };
                setMatches(newMatches);
                setSelectedA(null);
                setSelectedB(null);

                // Check if all items are matched
                if (Object.keys(newMatches).length === columnAItems.length) {
                    setTimeout(() => {
                        setShowResults(true);
                        checkAnswers(newMatches);
                    }, 500);
                }
            } else {
                setSelectedB(selectedB === item ? null : item);
            }
        }
    };

    const checkAnswers = (finalMatches) => {
        const correctMatches = matchData.mappings.reduce((acc, mapping) => {
            acc[mapping.A] = mapping.B;
            return acc;
        }, {});

        let correctCount = 0;
        Object.entries(finalMatches).forEach(([a, b]) => {
            if (correctMatches[a] === b) {
                correctCount++;
            }
        });

        const accuracy = Math.round((correctCount / Object.keys(finalMatches).length) * 100);

        setTimeout(() => {
            setIsComplete(true);
            if (onComplete) {
                onComplete({
                    totalItems: columnAItems.length,
                    correctMatches: correctCount,
                    accuracy,
                    userMatches: finalMatches
                });
            }
        }, 2000);
    };

    const resetExercise = () => {
        setMatches({});
        setSelectedA(null);
        setSelectedB(null);
        setShowResults(false);
        setIsComplete(false);
        // Re-shuffle column B
        const shuffledB = [...matchData.columnB].sort(() => Math.random() - 0.5);
        setColumnBItems(shuffledB);
    };

    const isCorrectMatch = (itemA, itemB) => {
        if (!showResults) return false;
        const correctMapping = matchData.mappings.find(m => m.A === itemA);
        return correctMapping && correctMapping.B === itemB;
    };

    const isIncorrectMatch = (itemA, itemB) => {
        if (!showResults) return false;
        return matches[itemA] === itemB && !isCorrectMatch(itemA, itemB);
    };

    if (!matchData || !columnAItems.length || !columnBItems.length) {
        return (
            <div className="text-center py-8">
                <p className="text-gray-500">No matching exercise available</p>
            </div>
        );
    }

    if (isComplete) {
        const correctCount = Object.entries(matches).filter(([a, b]) => isCorrectMatch(a, b)).length;
        const accuracy = Math.round((correctCount / columnAItems.length) * 100);

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

                    <h3 className="text-2xl font-bold text-gray-800 mb-2">Exercise Complete!</h3>

                    <div className="space-y-2 mb-6">
                        <p className="text-gray-600">
                            You matched <span className="font-bold text-green-600">{correctCount}</span> out of{' '}
                            <span className="font-bold">{columnAItems.length}</span> correctly
                        </p>
                        <p className="text-lg font-semibold text-gray-800">{accuracy}% Accuracy</p>
                    </div>

                    <button
                        onClick={resetExercise}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                        <RotateCcw className="w-4 h-4" />
                        <span>Try Again</span>
                    </button>
                </div>
            </motion.div>
        );
    }

    return (
        <div className={className}>
            <div className="bg-white rounded-xl shadow-lg p-6">
                {/* Instructions */}
                <div className="mb-6 text-center">
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">Match the Following</h3>
                    <p className="text-gray-600">Click an item from Column A, then click its match in Column B</p>
                </div>

                {/* Progress */}
                <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-gray-600">
                            Progress: {Object.keys(matches).length} / {columnAItems.length}
                        </span>
                        <button
                            onClick={resetExercise}
                            className="text-sm text-blue-600 hover:text-blue-700 flex items-center space-x-1"
                        >
                            <Shuffle className="w-4 h-4" />
                            <span>Reset</span>
                        </button>
                    </div>

                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <motion.div
                            className="bg-blue-600 h-2 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${(Object.keys(matches).length / columnAItems.length) * 100}%` }}
                            transition={{ duration: 0.3 }}
                        />
                    </div>
                </div>

                {/* Matching interface */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Column A */}
                    <div>
                        <h4 className="font-semibold text-gray-700 mb-3 text-center">Column A</h4>
                        <div className="space-y-2">
                            {columnAItems.map((item, index) => (
                                <motion.button
                                    key={index}
                                    onClick={() => handleItemClick(item, 'A')}
                                    className={`
                    w-full p-3 text-left rounded-lg border-2 transition-all duration-200
                    ${selectedA === item
                                            ? 'border-blue-500 bg-blue-50'
                                            : 'border-gray-200 hover:border-gray-300'
                                        }
                    ${matches[item] && showResults
                                            ? isCorrectMatch(item, matches[item])
                                                ? 'border-green-500 bg-green-50'
                                                : 'border-red-500 bg-red-50'
                                            : ''
                                        }
                    ${matches[item] && !showResults ? 'border-purple-500 bg-purple-50' : ''}
                    ${showResults ? 'cursor-default' : 'cursor-pointer hover:bg-gray-50'}
                  `}
                                    disabled={showResults}
                                    whileHover={!showResults ? { scale: 1.02 } : {}}
                                    whileTap={!showResults ? { scale: 0.98 } : {}}
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="flex-1">{item}</span>
                                        {matches[item] && showResults && (
                                            <div className="ml-2">
                                                {isCorrectMatch(item, matches[item]) ? (
                                                    <Check className="w-5 h-5 text-green-500" />
                                                ) : (
                                                    <X className="w-5 h-5 text-red-500" />
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    {matches[item] && (
                                        <div className="mt-2 text-sm text-gray-600">
                                            Matched with: {matches[item]}
                                        </div>
                                    )}
                                </motion.button>
                            ))}
                        </div>
                    </div>

                    {/* Column B */}
                    <div>
                        <h4 className="font-semibold text-gray-700 mb-3 text-center">Column B</h4>
                        <div className="space-y-2">
                            {columnBItems.map((item, index) => {
                                const isMatched = Object.values(matches).includes(item);
                                const matchedWithA = Object.entries(matches).find(([a, b]) => b === item)?.[0];

                                return (
                                    <motion.button
                                        key={index}
                                        onClick={() => handleItemClick(item, 'B')}
                                        className={`
                      w-full p-3 text-left rounded-lg border-2 transition-all duration-200
                      ${selectedB === item && !selectedA
                                                ? 'border-blue-500 bg-blue-50'
                                                : 'border-gray-200 hover:border-gray-300'
                                            }
                      ${isMatched && showResults
                                                ? isCorrectMatch(matchedWithA, item)
                                                    ? 'border-green-500 bg-green-50'
                                                    : 'border-red-500 bg-red-50'
                                                : ''
                                            }
                      ${isMatched && !showResults ? 'border-purple-500 bg-purple-50' : ''}
                      ${showResults || isMatched ? 'cursor-default' : 'cursor-pointer hover:bg-gray-50'}
                    `}
                                        disabled={showResults || isMatched}
                                        whileHover={!showResults && !isMatched ? { scale: 1.02 } : {}}
                                        whileTap={!showResults && !isMatched ? { scale: 0.98 } : {}}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="flex-1">{item}</span>
                                            {isMatched && showResults && (
                                                <div className="ml-2">
                                                    {isCorrectMatch(matchedWithA, item) ? (
                                                        <Check className="w-5 h-5 text-green-500" />
                                                    ) : (
                                                        <X className="w-5 h-5 text-red-500" />
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </motion.button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Instructions */}
                {selectedA && (
                    <motion.div
                        className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-200"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <p className="text-blue-800 text-sm">
                            Selected: <strong>{selectedA}</strong> - Now click its match in Column B
                        </p>
                    </motion.div>
                )}
            </div>
        </div>
    );
};

export default MatchTheFollowing;
