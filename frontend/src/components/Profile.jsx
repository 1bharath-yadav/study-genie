import React, { useEffect, useState } from "react";
import axios from "axios";

const Profile = ({ studentId }) => {
    const [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!studentId) return;
        setLoading(true);
        axios
            .get(`/api/students/${studentId}`)
            .then((res) => {
                setProfile(res.data);
                setError(null);
            })
            .catch((err) => {
                setError("Failed to load profile");
                setProfile(null);
            })
            .finally(() => setLoading(false));
    }, [studentId]);

    if (loading) return <div>Loading profile...</div>;
    if (error) return <div className="text-red-500">{error}</div>;
    if (!profile) return <div>No profile data found.</div>;

    return (
        <div className="p-4 rounded-2xl border border-white/20 bg-white/70 dark:bg-white/5 backdrop-blur-xl shadow-md max-w-md mx-auto">
            <h2 className="text-xl font-bold mb-2">Student Profile</h2>
            <div className="mb-2"><b>Name:</b> {profile.full_name}</div>
            <div className="mb-2"><b>Username:</b> {profile.username}</div>
            <div className="mb-2"><b>Email:</b> {profile.email}</div>
            <div className="mb-2"><b>ID:</b> {profile.student_id}</div>
            {/* Add more fields as needed */}
        </div>
    );
};

export default Profile;
