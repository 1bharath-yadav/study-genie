import React from 'react';
import { motion } from 'framer-motion';
import { Send, Bot, User, FileText, AlertCircle, CheckCircle } from 'lucide-react';

const ChatBox = ({
    chatHistory,
    userPrompt,
    setUserPrompt,
    onPromptSubmit,
    isProcessing
}) => {
    const handleSubmit = (e) => {
        e.preventDefault();
        onPromptSubmit();
    };

    const MessageIcon = ({ type }) => {
        switch (type) {
            case 'user':
                return <User className="w-4 h-4 text-blue-600" />;
            case 'ai':
                return <Bot className="w-4 h-4 text-green-600" />;
            case 'file':
                return <FileText className="w-4 h-4 text-purple-600" />;
            case 'error':
                return <AlertCircle className="w-4 h-4 text-red-600" />;
            case 'success':
                return <CheckCircle className="w-4 h-4 text-green-600" />;
            default:
                return <Bot className="w-4 h-4 text-gray-600" />;
        }
    };

    const getMessageStyle = (type) => {
        switch (type) {
            case 'user':
                return 'bg-blue-100 text-blue-800 border-blue-200';
            case 'ai':
                return 'bg-green-100 text-green-800 border-green-200';
            case 'file':
                return 'bg-purple-100 text-purple-800 border-purple-200';
            case 'error':
                return 'bg-red-100 text-red-800 border-red-200';
            case 'success':
                return 'bg-green-100 text-green-800 border-green-200';
            default:
                return 'bg-gray-100 text-gray-800 border-gray-200';
        }
    };

    return (
        <div className="h-full flex flex-col bg-white rounded-lg shadow-sm border border-gray-200">
            {/* Chat Header */}
            <div className="p-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                    <Bot className="w-5 h-5 mr-2 text-blue-600" />
                    AI Study Assistant
                </h3>
                <p className="text-sm text-gray-600">Ask questions or upload files to generate study content</p>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chatHistory.length === 0 ? (
                    <div className="text-center text-gray-500 py-8">
                        <Bot className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                        <p>Start a conversation by typing a question or uploading a file!</p>
                    </div>
                ) : (
                    chatHistory.map((message, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex items-start space-x-2 p-3 rounded-lg border ${getMessageStyle(message.type)}`}
                        >
                            <MessageIcon type={message.type} />
                            <div className="flex-1">
                                <p className="text-sm">{message.content}</p>
                                <span className="text-xs opacity-70">
                                    {message.timestamp.toLocaleTimeString()}
                                </span>
                            </div>
                        </motion.div>
                    ))
                )}

                {isProcessing && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center space-x-2 p-3 rounded-lg bg-blue-50 border border-blue-200"
                    >
                        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-sm text-blue-800">AI is processing your request...</p>
                    </motion.div>
                )}
            </div>

            {/* Chat Input */}
            <div className="p-4 border-t border-gray-200">
                <form onSubmit={handleSubmit} className="flex space-x-2">
                    <input
                        type="text"
                        value={userPrompt}
                        onChange={(e) => setUserPrompt(e.target.value)}
                        placeholder="Ask me anything about your studies..."
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        disabled={isProcessing}
                    />
                    <button
                        type="submit"
                        disabled={!userPrompt.trim() || isProcessing}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatBox;
