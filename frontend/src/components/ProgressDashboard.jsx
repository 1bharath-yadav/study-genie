import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, LineElement, PointElement } from 'chart.js';
import { Doughnut, Bar, Line } from 'react-chartjs-2';
import { TrendingUp, Target, Clock, Trophy, Brain, BookOpen } from 'lucide-react';
import { getStatusColor, getProgressColor, formatDate } from '../utils';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, LineElement, PointElement);

const ProgressDashboard = ({ progressData, className = '' }) => {
    const heatmapRef = useRef();

    // Create knowledge heatmap using D3.js
    useEffect(() => {
        if (!progressData?.concept_progress || !heatmapRef.current) return;

        const svg = d3.select(heatmapRef.current);
        svg.selectAll("*").remove(); // Clear previous render

        const margin = { top: 20, right: 30, bottom: 40, left: 100 };
        const width = 600 - margin.left - margin.right;
        const height = 300 - margin.bottom - margin.top;

        const data = progressData.concept_progress.slice(0, 10); // Show top 10 concepts

        const xScale = d3.scaleBand()
            .domain(['Mastery', 'Progress', 'Difficulty'])
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleBand()
            .domain(data.map(d => d.concept_name))
            .range([0, height])
            .padding(0.1);

        const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
            .domain([0, 100]);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Create heatmap cells
        data.forEach(concept => {
            const values = [
                { key: 'Mastery', value: concept.mastery_score || 0 },
                { key: 'Progress', value: (concept.correct_answers / Math.max(concept.total_questions, 1)) * 100 },
                { key: 'Difficulty', value: concept.attempts_count > 0 ? Math.min(concept.attempts_count * 10, 100) : 0 }
            ];

            values.forEach(v => {
                g.append('rect')
                    .attr('x', xScale(v.key))
                    .attr('y', yScale(concept.concept_name))
                    .attr('width', xScale.bandwidth())
                    .attr('height', yScale.bandwidth())
                    .attr('fill', colorScale(v.value))
                    .attr('stroke', 'white')
                    .attr('stroke-width', 1)
                    .on('mouseover', function (event) {
                        // Tooltip
                        const tooltip = d3.select('body').append('div')
                            .attr('class', 'tooltip')
                            .style('opacity', 0)
                            .style('position', 'absolute')
                            .style('background', 'rgba(0, 0, 0, 0.8)')
                            .style('color', 'white')
                            .style('padding', '8px')
                            .style('border-radius', '4px')
                            .style('font-size', '12px')
                            .style('pointer-events', 'none');

                        tooltip.transition()
                            .duration(200)
                            .style('opacity', .9);

                        tooltip.html(`${concept.concept_name}<br/>${v.key}: ${v.value.toFixed(1)}`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                    })
                    .on('mouseout', function () {
                        d3.selectAll('.tooltip').remove();
                    });

                // Add text labels
                g.append('text')
                    .attr('x', xScale(v.key) + xScale.bandwidth() / 2)
                    .attr('y', yScale(concept.concept_name) + yScale.bandwidth() / 2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('fill', v.value > 50 ? 'white' : 'black')
                    .attr('font-size', '10px')
                    .attr('font-weight', 'bold')
                    .text(v.value.toFixed(0));
            });
        });

        // Add axes
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        // Add title
        svg.append('text')
            .attr('x', width / 2 + margin.left)
            .attr('y', margin.top / 2)
            .attr('text-anchor', 'middle')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold')
            .text('Knowledge Heatmap');

    }, [progressData]);

    if (!progressData) {
        return (
            <div className="text-center py-8">
                <p className="text-gray-500">No progress data available</p>
            </div>
        );
    }

    // Chart.js data preparation
    const overallProgressData = {
        labels: ['Mastered', 'In Progress', 'Weak Areas', 'Not Started'],
        datasets: [{
            data: [
                progressData.mastered_concepts || 0,
                progressData.concept_progress?.filter(c => c.status === 'in_progress').length || 0,
                progressData.weak_concepts || 0,
                progressData.total_concepts - (progressData.mastered_concepts || 0) -
                (progressData.concept_progress?.filter(c => c.status === 'in_progress').length || 0) -
                (progressData.weak_concepts || 0)
            ],
            backgroundColor: [
                '#22c55e',
                '#3b82f6',
                '#ef4444',
                '#9ca3af'
            ],
            borderWidth: 2,
            borderColor: '#ffffff'
        }]
    };

    const subjectProgressData = {
        labels: progressData.subject_progress?.map(s => s.subject_name) || [],
        datasets: [{
            label: 'Progress %',
            data: progressData.subject_progress?.map(s => s.progress_percentage || 0) || [],
            backgroundColor: 'rgba(59, 130, 246, 0.6)',
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: 2,
            borderRadius: 4,
        }]
    };

    // Expecting progressData.weekly_study_time = [minutes for Mon, Tue, ...]
    const weeklyProgressData = {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [{
            label: 'Study Time (minutes)',
            data: progressData.weekly_study_time || [0, 0, 0, 0, 0, 0, 0],
            borderColor: 'rgba(34, 197, 94, 1)',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: 'rgba(34, 197, 94, 1)',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 6,
        }]
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true,
                }
            },
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
            }
        }
    };

    return (
        <div className={`space-y-6 ${className}`}>
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Total Concepts</p>
                            <p className="text-2xl font-bold text-gray-900">{progressData.total_concepts || 0}</p>
                        </div>
                        <Brain className="w-8 h-8 text-blue-500" />
                    </div>
                </motion.div>

                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Mastered</p>
                            <p className="text-2xl font-bold text-green-600">{progressData.mastered_concepts || 0}</p>
                        </div>
                        <Trophy className="w-8 h-8 text-green-500" />
                    </div>
                </motion.div>

                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Weak Areas</p>
                            <p className="text-2xl font-bold text-red-600">{progressData.weak_concepts || 0}</p>
                        </div>
                        <Target className="w-8 h-8 text-red-500" />
                    </div>
                </motion.div>

                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Study Streak</p>
                            <p className="text-2xl font-bold text-purple-600">
                                {progressData.overall_stats?.streak_days || 0} days
                            </p>
                        </div>
                        <Clock className="w-8 h-8 text-purple-500" />
                    </div>
                </motion.div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Overall Progress Pie Chart */}
                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Overall Progress</h3>
                    <div className="h-64">
                        <Doughnut data={overallProgressData} options={chartOptions} />
                    </div>
                </motion.div>

                {/* Subject Progress Bar Chart */}
                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.6 }}
                >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Subject Progress</h3>
                    <div className="h-64">
                        <Bar data={subjectProgressData} options={{
                            ...chartOptions,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.1)',
                                    },
                                    ticks: {
                                        callback: function (value) {
                                            return value + '%';
                                        }
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false,
                                    },
                                }
                            }
                        }} />
                    </div>
                </motion.div>
            </div>

            {/* Weekly Progress and Knowledge Heatmap */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Weekly Study Time */}
                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Weekly Study Time</h3>
                    <div className="h-64">
                        <Line data={weeklyProgressData} options={{
                            ...chartOptions,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.1)',
                                    },
                                    ticks: {
                                        callback: function (value) {
                                            return value + 'm';
                                        }
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false,
                                    },
                                }
                            }
                        }} />
                    </div>
                </motion.div>

                {/* Knowledge Heatmap */}
                <motion.div
                    className="bg-white rounded-xl shadow-lg p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Knowledge Heatmap</h3>
                    <div className="h-64 overflow-hidden">
                        <svg ref={heatmapRef} width="100%" height="100%"></svg>
                    </div>
                </motion.div>
            </div>

            {/* Recent Activity */}
            <motion.div
                className="bg-white rounded-xl shadow-lg p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9 }}
            >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Activity</h3>
                <div className="space-y-3">
                    {progressData.recent_activity?.slice(0, 5).map((activity, index) => (
                        <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                            <BookOpen className="w-5 h-5 text-blue-500" />
                            <div className="flex-1">
                                <p className="text-sm font-medium text-gray-800">
                                    {activity.activity_type?.replace('_', ' ').toUpperCase() || 'Study Activity'}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {activity.concept_name || 'General Study'} • {formatDate(activity.timestamp)}
                                </p>
                            </div>
                            <div className="text-right">
                                <p className="text-sm font-semibold text-gray-800">
                                    {activity.score ? `${activity.score}%` : 'Completed'}
                                </p>
                            </div>
                        </div>
                    )) || (
                            <p className="text-gray-500 text-center py-4">No recent activity</p>
                        )}
                </div>
            </motion.div>

            {/* Concept Progress List */}
            <motion.div
                className="bg-white rounded-xl shadow-lg p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0 }}
            >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Concept Progress</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                    {progressData.concept_progress?.map((concept, index) => (
                        <div key={index} className="flex items-center space-x-3 p-3 border border-gray-200 rounded-lg">
                            <div className={`w-3 h-3 rounded-full ${getStatusColor(concept.status)}`}></div>
                            <div className="flex-1">
                                <p className="text-sm font-medium text-gray-800">{concept.concept_name}</p>
                                <div className="flex items-center space-x-2 mt-1">
                                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                                        <div
                                            className={`h-2 rounded-full ${getProgressColor(concept.mastery_score || 0)}`}
                                            style={{ width: `${concept.mastery_score || 0}%` }}
                                        ></div>
                                    </div>
                                    <span className="text-xs text-gray-500">{concept.mastery_score || 0}%</span>
                                </div>
                            </div>
                            <div className="text-right text-xs text-gray-500">
                                <p>{concept.correct_answers || 0}/{concept.total_questions || 0}</p>
                                <p>{concept.attempts_count || 0} attempts</p>
                            </div>
                        </div>
                    )) || (
                            <p className="text-gray-500 text-center py-4">No concept progress available</p>
                        )}
                </div>
            </motion.div>
        </div>
    );
};

export default ProgressDashboard;
