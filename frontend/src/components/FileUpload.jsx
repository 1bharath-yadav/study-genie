import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import {
    PlayCircle, FileText, File, X, Check, Send,
    Bot, User, AlertCircle
} from 'lucide-react';
import { fileUtils } from '../utils';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

const AssistantBox = ({ studentId, onFileProcessed, isProcessing: externalProcessing, hasContent = false, onClose }) => {
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [pendingFile, setPendingFile] = useState(null); // Store file until submit
    const [chatHistory, setChatHistory] = useState([
        {
            type: 'assistant',
            content: '👋 Hi! I\'m your AI Study Assistant. Upload a file or ask me anything to get started!',
            timestamp: new Date()
        }
    ]);
    const [userPrompt, setUserPrompt] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const messagesEndRef = useRef(null);

    // Track window size dynamically
    const [windowSize, setWindowSize] = useState({
        width: window.innerWidth,
        height: window.innerHeight,
    });

    useEffect(() => {
        const handleResize = () => {
            setWindowSize({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Auto-scroll to bottom when chat updates
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory]);

    // File upload logic
    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;
        setPendingFile(file);
        toast.success(`Ready to upload: "${file.name}". Click send to process.`);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'text/plain': ['.txt'],
            'text/markdown': ['.md']
        },
        maxFiles: 1,
        disabled: isProcessing || externalProcessing
    });

    // Handle prompt submission
    const handlePromptSubmit = async (e) => {
        e.preventDefault();
        if (!userPrompt.trim() && !pendingFile) return;

        if (userPrompt.trim()) {
            setChatHistory(prev => [...prev, {
                type: 'user',
                content: userPrompt,
                timestamp: new Date()
            }]);
        }

        try {
            setIsProcessing(true);
            let resp = null;
            if (pendingFile) {
                // File upload (with or without prompt)
                const fileWithId = {
                    id: Date.now(),
                    file: pendingFile,
                    name: pendingFile.name,
                    size: pendingFile.size,
                    type: pendingFile.type,
                    status: 'processing'
                };
                setUploadedFiles(prev => [...prev, fileWithId]);
                setChatHistory(prev => [...prev, {
                    type: 'file',
                    content: `Uploaded file: ${pendingFile.name}`,
                    timestamp: new Date()
                }]);
                setChatHistory(prev => [...prev, {
                    type: 'assistant',
                    content: `🔄 Processing ${pendingFile.name}... This may take a moment.`,
                    timestamp: new Date()
                }]);
                resp = await apiService.uploadFileSimple(
                    pendingFile,
                    studentId,
                    userPrompt && userPrompt.trim() ? userPrompt : `Generate study materials from this file: ${pendingFile.name}`
                );
                setUploadedFiles(prev =>
                    prev.map(f => f.id === fileWithId.id
                        ? { ...f, status: 'completed', response: resp }
                        : f
                    )
                );
                setChatHistory(prev => [...prev, {
                    type: 'assistant',
                    content: resp?.message || 'Study materials generated successfully!',
                    timestamp: new Date()
                }]);
                if (onFileProcessed && resp) {
                    onFileProcessed({
                        fileName: pendingFile.name,
                        enhanced_response: resp,
                        ...resp
                    });
                }
            } else if (userPrompt.trim()) {
                // Only prompt
                resp = await apiService.processLLMResponse({
                    prompt: userPrompt,
                    student_id: studentId
                });
                setChatHistory(prev => [...prev, {
                    type: 'assistant',
                    content: resp?.message || resp?.answer || 'Got it!',
                    timestamp: new Date()
                }]);
            }
        } catch {
            setChatHistory(prev => [...prev, {
                type: 'error',
                content: 'Something went wrong. Try again later.',
                timestamp: new Date()
            }]);
        } finally {
            setIsProcessing(false);
            setUserPrompt('');
            setPendingFile(null);
        }
    };

    // Icons for messages
    const getMessageIcon = (type) => {
        switch (type) {
            case 'user': return <User className="w-4 h-4 text-indigo-600" />;
            case 'assistant': return <Bot className="w-4 h-4 text-green-600" />;
            case 'file': return <FileText className="w-4 h-4 text-purple-600" />;
            case 'error': return <AlertCircle className="w-4 h-4 text-red-600" />;
            default: return <Bot className="w-4 h-4 text-gray-600" />;
        }
    };

    // Calculate dynamic dimensions
    const boxWidth = Math.min(windowSize.width * 0.3, 500);
    const boxHeight = hasContent
        ? Math.min(windowSize.height * 0.45, 500)
        : Math.min(windowSize.height * 0.7, 700);

    return (
        <div
            className="fixed bottom-20 right-6 z-50 bg-gradient-to-b from-indigo-900/95 to-indigo-700/90
            text-white shadow-2xl rounded-xl flex flex-col transition-all border border-indigo-500/20 backdrop-blur-md"
            style={{
                width: `${boxWidth}px`,
                height: `${boxHeight}px`
            }}
        >
            {/* Header */}
            <div className="p-3 border-b border-indigo-500/20 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <Bot className="w-5 h-5 text-indigo-200" />
                    <h3 className="text-lg font-semibold text-white">AI Study Assistant</h3>
                </div>
                <div className="flex items-center space-x-2">
                    <span className="text-xs text-indigo-200/80">Ask, upload & learn</span>
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="p-1 hover:bg-white/10 rounded-md transition-colors"
                        >
                            <X className="w-4 h-4 text-indigo-200" />
                        </button>
                    )}
                </div>
            </div>

            {/* Chat History */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chatHistory.map((msg, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex items-start space-x-3 p-3 rounded-lg max-w-[80%]
                            ${msg.type === 'user'
                                ? 'bg-indigo-600/20 ml-auto border border-indigo-500/20'
                                : msg.type === 'assistant'
                                    ? 'bg-white/5 border border-white/10'
                                    : msg.type === 'file'
                                        ? 'bg-purple-600/20 border border-purple-500/20'
                                        : 'bg-red-600/20 border border-red-500/20'
                            }`}
                    >
                        {getMessageIcon(msg.type)}
                        <div>
                            <p className="text-sm text-indigo-100">
                                {msg.content.includes('🔄 Processing') ? (
                                    <span className="flex items-center gap-2">
                                        {msg.content.replace('🔄', '')}
                                        <span className="flex space-x-1">
                                            <span className="w-1 h-1 bg-indigo-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                            <span className="w-1 h-1 bg-indigo-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                            <span className="w-1 h-1 bg-indigo-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                        </span>
                                    </span>
                                ) : (
                                    msg.content
                                )}
                            </p>
                            <span className="text-xs text-indigo-200/70">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                            </span>
                        </div>
                    </motion.div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input + Upload Area */}
            <div className="p-3 border-t border-indigo-500/20 bg-gradient-to-r from-indigo-900/20 to-indigo-700/10 backdrop-blur-sm">
                {/* Quick Action Buttons */}
                {chatHistory.length <= 1 && (
                    <div className="mb-3 flex flex-wrap gap-2">
                        {[
                            "Create flashcards",
                            "Generate quiz",
                            "Explain concepts",
                            "Study tips"
                        ].map((prompt) => (
                            <button
                                key={prompt}
                                onClick={() => setUserPrompt(prompt)}
                                className="px-3 py-1 text-xs bg-white/10 hover:bg-white/20 rounded-full text-indigo-100 transition-colors"
                            >
                                {prompt}
                            </button>
                        ))}
                    </div>
                )}

                <form onSubmit={handlePromptSubmit} className="flex items-center space-x-3">
                    {/* Dropzone */}
                    <motion.div
                        {...getRootProps()}
                        className={`cursor-pointer flex items-center justify-center w-10 h-10 rounded-full transition-colors text-indigo-100/90
                            ${isDragActive
                                ? 'bg-indigo-600/20 ring-1 ring-indigo-400'
                                : 'bg-white/5 hover:bg-white/10'
                            }
                        `}
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                    >
                        <input {...getInputProps()} />
                        {/* Show file name if pending */}
                        {pendingFile && (
                            <span className="text-xs text-indigo-200 block truncate max-w-[80px]">{pendingFile.name}</span>
                        )}
                        <File className="w-5 h-5 text-indigo-200/90" />
                    </motion.div>

                    {/* Text Input */}
                    <input
                        type="text"
                        value={userPrompt}
                        onChange={(e) => setUserPrompt(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Escape' && onClose) {
                                onClose();
                            }
                        }}
                        placeholder={isProcessing ? "Processing..." : "Ask me or upload a file... (ESC to close)"}
                        className="flex-1 px-4 py-2 rounded-full bg-black/40 text-white placeholder:text-indigo-200/60 border border-transparent focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-50 shadow-sm transition-all"
                        disabled={isProcessing || externalProcessing}
                        autoFocus
                    />

                    {/* Send Button */}
                    <button
                        type="submit"
                        disabled={(!userPrompt.trim() && !pendingFile) || isProcessing || externalProcessing}
                        className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-full hover:from-indigo-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-50 shadow-md transition-all"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default AssistantBox;
