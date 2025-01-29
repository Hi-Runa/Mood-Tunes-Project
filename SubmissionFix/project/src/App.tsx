/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        This file is the main application component handling routing, authentication, and overall layout structure.
*/

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { Music, Users, LogIn, LogOut } from 'lucide-react';
import { VibeChecker } from './components/VibeChecker';
import { MusicRecommendations } from './components/MusicRecommendations';
import { TeamPage } from './components/TeamPage';
import { AuthForm } from './components/AuthForm';
import { EmotionResult, User } from './types';

// Simple in-memory user storage
const users: Record<string, { password: string; username: string }> = {};

function App() {
  const [currentVibe, setCurrentVibe] = useState<EmotionResult | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(() => {
    const saved = localStorage.getItem('user');
    return saved ? JSON.parse(saved) : null;
  });

  const handleLogin = async (email: string, password: string, username: string | undefined, isSignUp: boolean) => {
    if (isSignUp) {
      if (users[email]) {
        throw new Error('User already exists');
      }
      users[email] = { password, username: username || email.split('@')[0] };
    } else {
      const user = users[email];
      if (!user || user.password !== password) {
        throw new Error('Invalid credentials');
      }
    }

    const user = {
      email,
      username: users[email].username
    };
    
    setCurrentUser(user);
    localStorage.setItem('user', JSON.stringify(user));
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem('user');
  };

  // html headers for main app
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
        <header className="w-full bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <nav className="flex items-center justify-between">
              <Link to="/" className="flex items-center gap-3">
                <Music className="text-purple-600" size={32} />
                <h1 className="text-2xl font-bold text-gray-900">Mood Tunes</h1>
              </Link>
              <div className="flex items-center gap-6">
                {currentUser && (
                  <span className="text-gray-600">
                    Welcome, {currentUser.username}!
                  </span>
                )}
                <Link
                  to="/team"
                  className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
                >
                  <Users size={20} />
                  <span>Team</span>
                </Link>
                {currentUser ? (
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
                  >
                    <LogOut size={20} />
                    <span>Sign Out</span>
                  </button>
                ) : (
                  <Link
                    to="/auth"
                    className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
                  >
                    <LogIn size={20} />
                    <span>Sign In</span>
                  </Link>
                )}
              </div>
            </nav>
          </div>
        </header>

        <Routes>
          <Route
            path="/"
            element={
              <div className="min-h-[calc(100vh-80px)] flex items-center justify-center">
                <div className="max-w-3xl mx-auto px-4 py-16 text-center">
                  <h1 className="text-5xl font-bold text-gray-900 mb-6">
                    Welcome to MoodTunes!
                  </h1>
                  <p className="text-xl text-gray-600 mb-12 leading-relaxed">
                    Discover music that matches your feelings—whether by describing your mood 
                    or letting our AI detect your emotions through facial recognition. 
                    Enjoy personalized song recommendations that fit exactly how you feel!
                  </p>
                  <Link
                    to={currentUser ? "/find-music" : "/auth"}
                    className="inline-flex items-center gap-2 px-8 py-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-lg font-semibold shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all"
                  >
                    <Music size={24} />
                    Find Your Music
                  </Link>
                </div>
              </div>
            }
          />
          <Route
            path="/find-music"
            element={
              currentUser ? (
                <main className="max-w-7xl mx-auto px-4 py-8">
                  <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">
                      Find Your Perfect Music
                    </h2>
                    <p className="text-lg text-gray-600">
                      Take a selfie or tell us how you're feeling to get your personalized playlist
                    </p>
                  </div>

                  <VibeChecker onVibeCheck={setCurrentVibe} />
                  <MusicRecommendations emotion={currentVibe} />
                </main>
              ) : (
                <Navigate to="/auth" replace />
              )
            }
          />
          <Route path="/team" element={<TeamPage />} />
          <Route
            path="/auth"
            element={
              currentUser ? (
                <Navigate to="/find-music" replace />
              ) : (
                <AuthForm onSubmit={handleLogin} />
              )
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;