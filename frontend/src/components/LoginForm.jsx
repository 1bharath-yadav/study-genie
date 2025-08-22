import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Mail, UserPlus, LogIn } from 'lucide-react';
import { validators } from '../utils';
import toast from 'react-hot-toast';

const LoginForm = ({ onLogin, className = '' }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        full_name: ''
    });
    const [errors, setErrors] = useState({});

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));

        // Clear error when user starts typing
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: '' }));
        }
    };

    const validateForm = () => {
        const newErrors = {};

        if (!validators.username(formData.username)) {
            newErrors.username = 'Username must be 3-50 characters long';
        }

        if (!validators.email(formData.email)) {
            newErrors.email = 'Please enter a valid email address';
        }

        if (!isLogin && !validators.fullName(formData.full_name)) {
            newErrors.full_name = 'Full name must be 2-100 characters long';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) return;

        setIsLoading(true);

        try {
            console.log('=== LOGIN FORM SUBMISSION ===');
            console.log('Form data being sent:', formData);
            console.log('Is login mode:', isLogin);
            console.log('==============================');

            if (isLogin) {
                // For login, we still create/get the student (since no auth is implemented)
                const response = await onLogin(formData);
                console.log('Login response received:', response);
                toast.success('Welcome back!');
            } else {
                // For registration
                const response = await onLogin(formData);
                console.log('Registration response received:', response);
                toast.success('Account created successfully!');
            }
        } catch (error) {
            console.error('=== LOGIN FORM ERROR ===');
            console.error('Error object:', error);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            console.error('========================');
            toast.error(error.message || 'Login failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const switchMode = () => {
        setIsLogin(!isLogin);
        setErrors({});
        setFormData({
            username: '',
            email: '',
            full_name: ''
        });
    };

    return (
        <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 px-4 ${className}`}>
            <motion.div
                className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                {/* Header */}
                <div className="text-center mb-8">
                    <motion.div
                        className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                    >
                        <User className="w-8 h-8 text-white" />
                    </motion.div>

                    <h1 className="text-3xl font-bold text-gray-800 mb-2">Study Genie</h1>
                    <p className="text-gray-600">
                        {isLogin ? 'Welcome back! Sign in to continue' : 'Create your account to get started'}
                    </p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Username */}
                    <div>
                        <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                            Username
                        </label>
                        <div className="relative">
                            <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                            <input
                                type="text"
                                id="username"
                                name="username"
                                value={formData.username}
                                onChange={handleInputChange}
                                className={`
                  w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent
                  ${errors.username ? 'border-red-500' : 'border-gray-300'}
                `}
                                placeholder="Enter your username"
                                required
                            />
                        </div>
                        {errors.username && (
                            <p className="mt-1 text-sm text-red-600">{errors.username}</p>
                        )}
                    </div>

                    {/* Email */}
                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                            Email
                        </label>
                        <div className="relative">
                            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                            <input
                                type="email"
                                id="email"
                                name="email"
                                value={formData.email}
                                onChange={handleInputChange}
                                className={`
                  w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent
                  ${errors.email ? 'border-red-500' : 'border-gray-300'}
                `}
                                placeholder="Enter your email"
                                required
                            />
                        </div>
                        {errors.email && (
                            <p className="mt-1 text-sm text-red-600">{errors.email}</p>
                        )}
                    </div>

                    {/* Full Name (only for registration) */}
                    {!isLogin && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            <label htmlFor="full_name" className="block text-sm font-medium text-gray-700 mb-2">
                                Full Name
                            </label>
                            <div className="relative">
                                <UserPlus className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                                <input
                                    type="text"
                                    id="full_name"
                                    name="full_name"
                                    value={formData.full_name}
                                    onChange={handleInputChange}
                                    className={`
                    w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent
                    ${errors.full_name ? 'border-red-500' : 'border-gray-300'}
                  `}
                                    placeholder="Enter your full name"
                                    required={!isLogin}
                                />
                            </div>
                            {errors.full_name && (
                                <p className="mt-1 text-sm text-red-600">{errors.full_name}</p>
                            )}
                        </motion.div>
                    )}

                    {/* Submit Button */}
                    <motion.button
                        type="submit"
                        disabled={isLoading}
                        className={`
              w-full py-3 px-4 rounded-lg font-medium text-white transition-all duration-200
              ${isLoading
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transform hover:scale-105'
                            }
            `}
                        whileTap={{ scale: 0.95 }}
                    >
                        {isLoading ? (
                            <div className="flex items-center justify-center space-x-2">
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Processing...</span>
                            </div>
                        ) : (
                            <div className="flex items-center justify-center space-x-2">
                                <LogIn className="w-5 h-5" />
                                <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
                            </div>
                        )}
                    </motion.button>
                </form>

                {/* Switch Mode */}
                <div className="mt-6 text-center">
                    <p className="text-gray-600">
                        {isLogin ? "Don't have an account?" : "Already have an account?"}
                    </p>
                    <button
                        type="button"
                        onClick={switchMode}
                        className="mt-2 text-blue-600 hover:text-blue-700 font-medium transition-colors"
                    >
                        {isLogin ? 'Create Account' : 'Sign In'}
                    </button>
                </div>

                {/* Demo Note */}
                <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-xs text-yellow-800 text-center">
                        <strong>Demo Mode:</strong> This is a demonstration. No actual authentication is required.
                        Just enter any valid username and email to continue.
                    </p>
                </div>
            </motion.div>
        </div>
    );
};

export default LoginForm;
