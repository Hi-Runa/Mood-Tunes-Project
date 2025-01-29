import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { Music, Users, LogIn, Sparkles } from 'lucide-react';
import { VibeChecker } from './components/VibeChecker';
import { MusicRecommendations } from './components/MusicRecommendations';
import { TeamPage } from './components/TeamPage';
import { AuthForm } from './components/AuthForm';
import { EmotionResult } from './types';

function App() {
  const [currentVibe, setCurrentVibe] = useState<EmotionResult | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLogin = (email: string, password: string, isSignUp: boolean) => {
    setIsLoggedIn(true);
  };

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
                <Link
                  to="/team"
                  className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
                >
                  <Users size={20} />
                  <span>Team</span>
                </Link>
                {!isLoggedIn && (
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
                  <div className="mb-8 inline-block">
                    <Sparkles className="w-16 h-16 text-purple-600" />
                  </div>
                  <h1 className="text-5xl font-bold text-gray-900 mb-6">
                    Welcome to MoodTunes!
                  </h1>
                  <p className="text-xl text-gray-600 mb-12 leading-relaxed">
                    Discover music that matches your feelings—whether by describing your mood 
                    or letting our AI detect your emotions through facial recognition. 
                    Enjoy personalized song recommendations that fit exactly how you feel!
                  </p>
                  <Link
                    to="/find-music"
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
            }
          />
          <Route path="/team" element={<TeamPage />} />
          <Route
            path="/auth"
            element={<AuthForm onSubmit={handleLogin} />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;