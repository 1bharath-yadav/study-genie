import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, Image, File, X, Check } from 'lucide-react';
import { fileUtils } from '../utils';
import toast from 'react-hot-toast';

const FileUpload = ({ onFileProcessed, isProcessing = false, className = '' }) => {
    const [uploadedFiles, setUploadedFiles] = useState([]);

    const onDrop = useCallback((acceptedFiles) => {
        const files = acceptedFiles;
        if (!files || files.length === 0) return;

        const filesWithStatus = files.map(file => ({
            id: Date.now() + Math.random(),
            file,
            name: file.name,
            size: file.size,
            type: file.type,
            status: 'pending'
        }));

        setUploadedFiles(filesWithStatus);
        toast.success(`${files.length} file(s) ready for processing.`);
        onFileProcessed(files); // Pass the raw files to the parent
    }, [onFileProcessed]);

    const removeFile = (fileId) => {
        setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/plain': ['.txt'],
            'application/pdf': ['.pdf'],
            'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        },
        maxFiles: 1,
        disabled: isProcessing
    });

    const getFileIcon = (type) => {
        if (type.startsWith('image/')) return <Image className="w-8 h-8" />;
        if (type === 'application/pdf') return <FileText className="w-8 h-8" />;
        return <File className="w-8 h-8" />;
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <Check className="w-5 h-5 text-green-500" />;
            case 'processing':
                return (
                    <motion.div
                        className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    />
                );
            case 'error':
                return <X className="w-5 h-5 text-red-500" />;
            default:
                return null;
        }
    };

    return (
        <div className={`w-full ${className}`}>
            <motion.div
                {...getRootProps()}
                className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
          ${isDragActive
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 hover:border-gray-400'
                    }
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
                whileHover={!isProcessing ? { scale: 1.02 } : {}}
                whileTap={!isProcessing ? { scale: 0.98 } : {}}
            >
                <input {...getInputProps()} />

                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.3 }}
                >
                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />

                    <h3 className="text-lg font-semibold text-gray-700 mb-2">
                        {isDragActive ? 'Drop your file here' : 'Upload Study Material'}
                    </h3>

                    <p className="text-gray-500 mb-4">
                        Drag and drop or click to upload PDF, images, or text files
                    </p>

                    <div className="flex justify-center space-x-4 text-sm text-gray-400">
                        <span className="flex items-center">
                            <FileText className="w-4 h-4 mr-1" /> PDF
                        </span>
                        <span className="flex items-center">
                            <Image className="w-4 h-4 mr-1" /> Images
                        </span>
                        <span className="flex items-center">
                            <File className="w-4 h-4 mr-1" /> Text
                        </span>
                    </div>
                </motion.div>
            </motion.div>

            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
                <motion.div
                    className="mt-4 space-y-2"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    {uploadedFiles.map((file) => (
                        <div
                            key={file.id}
                            className="flex items-center p-3 bg-gray-50 rounded-lg border"
                        >
                            <div className="flex-shrink-0 mr-3 text-gray-400">
                                {getFileIcon(file.type)}
                            </div>

                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-900 truncate">
                                    {file.name}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {fileUtils.formatFileSize(file.size)}
                                </p>
                            </div>

                            <div className="flex items-center space-x-2">
                                {getStatusIcon(file.status)}

                                <button
                                    onClick={() => removeFile(file.id)}
                                    className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))}
                </motion.div>
            )}
        </div>
    );
};

export default FileUpload;
