import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility function to merge Tailwind classes
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

// Format date utilities
export const formatDate = (date) => {
    if (!date) return 'Never';
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
    });
};

export const formatDateTime = (date) => {
    if (!date) return 'Never';
    return new Date(date).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};

// Progress calculation utilities
export const calculateProgress = (correct, total) => {
    if (total === 0) return 0;
    return Math.round((correct / total) * 100);
};

export const getProgressColor = (percentage) => {
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 60) return 'bg-yellow-500';
    if (percentage >= 40) return 'bg-orange-500';
    return 'bg-red-500';
};

export const getStatusColor = (status) => {
    const colors = {
        'not_started': 'bg-gray-500',
        'in_progress': 'bg-blue-500',
        'mastered': 'bg-green-500',
        'needs_review': 'bg-red-500',
    };
    return colors[status] || 'bg-gray-500';
};

export const getStatusText = (status) => {
    const texts = {
        'not_started': 'Not Started',
        'in_progress': 'In Progress',
        'mastered': 'Mastered',
        'needs_review': 'Needs Review',
    };
    return texts[status] || 'Unknown';
};

// Difficulty level utilities
export const getDifficultyColor = (difficulty) => {
    const colors = {
        'Easy': 'text-green-600 bg-green-100',
        'Medium': 'text-yellow-600 bg-yellow-100',
        'Hard': 'text-red-600 bg-red-100',
    };
    return colors[difficulty] || 'text-gray-600 bg-gray-100';
};

// Local storage utilities
export const storage = {
    set: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Error saving to localStorage:', error);
        }
    },

    get: (key, defaultValue = null) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Error reading from localStorage:', error);
            return defaultValue;
        }
    },

    remove: (key) => {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('Error removing from localStorage:', error);
        }
    },
};

// Validation utilities
export const validators = {
    email: (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },

    username: (username) => {
        return username && username.length >= 3 && username.length <= 50;
    },

    fullName: (name) => {
        return name && name.length >= 2 && name.length <= 100;
    },
};

// Animation utilities
export const animations = {
    fadeIn: {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.5 }
    },

    slideUp: {
        initial: { y: 20, opacity: 0 },
        animate: { y: 0, opacity: 1 },
        transition: { duration: 0.3 }
    },

    slideInLeft: {
        initial: { x: -20, opacity: 0 },
        animate: { x: 0, opacity: 1 },
        transition: { duration: 0.3 }
    },

    scale: {
        initial: { scale: 0.9, opacity: 0 },
        animate: { scale: 1, opacity: 1 },
        transition: { duration: 0.2 }
    },
};

// File type checking utilities
export const fileUtils = {
    isImage: (file) => file.type.startsWith('image/'),
    isPDF: (file) => file.type === 'application/pdf',
    isText: (file) => file.type === 'text/plain',

    getFileIcon: (file) => {
        if (fileUtils.isImage(file)) return '🖼️';
        if (fileUtils.isPDF(file)) return '📄';
        if (fileUtils.isText(file)) return '📝';
        return '📎';
    },

    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
};

// Debounce utility for search and inputs
export const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Generate unique IDs
export const generateId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

// Study session utilities
export const studyUtils = {
    calculateStudyTime: (startTime, endTime) => {
        const diff = endTime - startTime;
        const minutes = Math.floor(diff / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        return `${minutes}m ${seconds}s`;
    },

    getRecommendedBreak: (studyMinutes) => {
        if (studyMinutes >= 90) return 'Take a 15-20 minute break';
        if (studyMinutes >= 45) return 'Take a 10-15 minute break';
        if (studyMinutes >= 25) return 'Take a 5-10 minute break';
        return 'Keep going!';
    },

    calculateAccuracy: (correct, total) => {
        if (total === 0) return 0;
        return Math.round((correct / total) * 100);
    },
};

// Chart.js default options
export const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
        },
    },
    scales: {
        y: {
            beginAtZero: true,
            grid: {
                color: 'rgba(0, 0, 0, 0.1)',
            },
        },
        x: {
            grid: {
                color: 'rgba(0, 0, 0, 0.1)',
            },
        },
    },
};

export default {
    cn,
    formatDate,
    formatDateTime,
    calculateProgress,
    getProgressColor,
    getStatusColor,
    getStatusText,
    getDifficultyColor,
    storage,
    validators,
    animations,
    fileUtils,
    debounce,
    generateId,
    studyUtils,
    chartDefaults,
};
