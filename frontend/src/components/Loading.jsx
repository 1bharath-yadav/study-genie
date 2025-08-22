import React from 'react';
import { motion } from 'framer-motion';

const Loading = ({ size = 'medium', text = 'Loading...', className = '' }) => {
    const sizeClasses = {
        small: 'w-4 h-4',
        medium: 'w-8 h-8',
        large: 'w-12 h-12',
    };

    return (
        <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
            <motion.div
                className={`border-4 border-gray-200 border-t-blue-600 rounded-full ${sizeClasses[size]}`}
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            />
            {text && (
                <motion.p
                    className="text-gray-600 text-sm font-medium"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    {text}
                </motion.p>
            )}
        </div>
    );
};

export default Loading;
