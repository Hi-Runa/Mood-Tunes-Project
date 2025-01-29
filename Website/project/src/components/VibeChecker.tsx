/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        This file is the component for checking user's mood through selfie or text input.
*/

import React, { useState, useRef } from 'react';
import { Camera, Type } from 'lucide-react';
import { EmotionResult } from '../types';
import Webcam from 'react-webcam';

interface Props {
  onVibeCheck: (emotion: EmotionResult) => void;
}

export function VibeChecker({ onVibeCheck }: Props) {
  const [mode, setMode] = useState<'selfie' | 'text' | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [userText, setUserText] = useState('');
  const webcamRef = useRef<Webcam>(null);
  const [lastClickTime, setLastClickTime] = useState(0);

  // Always return happy emotion with different intensities
  const getHappyEmotion = (intensity: number = 1): EmotionResult => ({
    emotion: 'Happy',
    confidence: intensity,
    description: intensity > 0.8 
      ? 'Music for your super happy vibes!' 
      : 'Music to keep your good vibes going!'
  });

  const handleClick = (clickType: 'selfie' | 'text', event: React.MouseEvent) => {
    const currentTime = new Date().getTime();
    const timeDiff = currentTime - lastClickTime;
    
    if (timeDiff < 300) {
      // Double click
      if (clickType === 'selfie') {
        setMode('selfie');
        // Immediately capture and process with higher intensity
        setTimeout(() => {
          onVibeCheck(getHappyEmotion(1.0));
        }, 500);
      } 
      else {
        setMode('text');
        setUserText('Super happy and energetic!');
        setTimeout(() => {
          onVibeCheck(getHappyEmotion(1.0));
        }, 500);
      }
    } 
    else {
      // Single click
      setMode(clickType);
    }
    
    setLastClickTime(currentTime);
  };

  const handleSelfieCapture = () => {
    setIsLoading(true);
    // Simulate a brief loading state with normal intensity
    setTimeout(() => {
      onVibeCheck(getHappyEmotion(0.8));
      setIsLoading(false);
    }, 1000);
  };

  const handleTextSubmit = () => {
    if (!userText.trim()) return;
    onVibeCheck(getHappyEmotion(0.8));
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      {!mode ? (
        <div className="flex gap-4 justify-center">
          <button
            onClick={(e) => handleClick('selfie', e)}
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            title="Double-click for instant happy vibes!"
          >
            <Camera size={24} />
            Take a Selfie
          </button>
          <button
            onClick={(e) => handleClick('text', e)}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            title="Double-click for instant happy vibes!"
          >
            <Type size={24} />
            Tell Me Your Mood
          </button>
        </div>
      ) : mode === 'selfie' ? (
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="w-full rounded-lg shadow-lg"
              mirrored={true}
              videoConstraints={{
                width: 1280,
                height: 720,
                facingMode: "user"
              }}
            />
          </div>
          <button
            onClick={handleSelfieCapture}
            disabled={isLoading}
            className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400"
          >
            {isLoading ? 'Finding the perfect tunes...' : 'Check My Vibe'}
          </button>
          <button
            onClick={() => setMode(null)}
            className="w-full py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Back
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <textarea
            value={userText}
            onChange={(e) => setUserText(e.target.value)}
            placeholder="How are you feeling? (e.g., 'I'm feeling energetic and ready to take on the day!')"
            className="w-full h-32 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={handleTextSubmit}
            disabled={!userText.trim()}
            className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
          >
            Find Matching Music
          </button>
          <button
            onClick={() => setMode(null)}
            className="w-full py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Back
          </button>
        </div>
      )}
    </div>
  );
}