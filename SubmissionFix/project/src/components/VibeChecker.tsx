import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Type } from 'lucide-react';
import { EmotionResult } from '../types';
import { startBrain, checkVibe, VIBES } from '../utils/moodChecker';

interface Props {
  onVibeCheck: (emotion: EmotionResult) => void;
}

export function VibeChecker({ onVibeCheck }: Props) {
  const [mode, setMode] = useState<'selfie' | 'text' | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const camRef = useRef<Webcam>(null);
  const [userText, setUserText] = useState('');
  const [brainReady, setBrainReady] = useState(false);

  useEffect(() => {
    async function wakeBrain() {
      try {
        const brain = await startBrain();
        setBrainReady(!!brain);
      } catch (error) {
        console.error('Brain malfunction:', error);
      }
    }
    wakeBrain();
  }, []);

  const readVibe = useCallback((text: string) => {
    const vibePatterns = [
      { pattern: /angry|mad|furious|rage/i, vibe: 'Angry' },
      { pattern: /disgust|gross|repulsed/i, vibe: 'Disgust' },
      { pattern: /scared|afraid|fearful|anxious/i, vibe: 'Fear' },
      { pattern: /happy|joy|excited|delighted/i, vibe: 'Happy' },
      { pattern: /sad|down|depressed|unhappy/i, vibe: 'Sad' },
      { pattern: /surprised|shocked|amazed/i, vibe: 'Surprise' },
      { pattern: /neutral|okay|fine/i, vibe: 'Neutral' }
    ];

    for (const { pattern, vibe } of vibePatterns) {
      if (pattern.test(text)) {
        return {
          emotion: vibe,
          confidence: 0.9,
          description: `Music for your ${vibe.toLowerCase()} mood`
        };
      }
    }

    return {
      emotion: 'Neutral',
      confidence: 0.7,
      description: `Music for your mood: ${text}`
    };
  }, []);

  const takeSelfie = useCallback(async () => {
    if (!camRef.current || !brainReady) return;
    
    setIsChecking(true);
    try {
      const pic = camRef.current.getScreenshot();
      if (!pic) return;

      const selfie = new Image();
      selfie.src = pic;
      await selfie.decode();

      const { vibe, sureness } = await checkVibe(selfie);
      
      onVibeCheck({
        emotion: vibe,
        confidence: sureness,
        description: `Music for your ${vibe.toLowerCase()} mood`
      });
    } catch (error) {
      console.error('Vibe check failed:', error);
    } finally {
      setIsChecking(false);
    }
  }, [brainReady, onVibeCheck]);

  const checkVibeFromText = useCallback(() => {
    if (!userText.trim()) return;
    const vibeResult = readVibe(userText.toLowerCase());
    onVibeCheck(vibeResult);
  }, [userText, readVibe, onVibeCheck]);

  // Rest of the component remains the same, just update the text to be more casual
  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      {!mode ? (
        <div className="flex gap-4 justify-center">
          <button
            onClick={() => setMode('selfie')}
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            <Camera size={24} />
            Take a Selfie
          </button>
          <button
            onClick={() => setMode('text')}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Type size={24} />
            Tell Me Your Mood
          </button>
        </div>
      ) : mode === 'selfie' ? (
        <div className="space-y-4">
          <div className="relative">
            <Webcam
              ref={camRef}
              screenshotFormat="image/jpeg"
              className="w-full rounded-lg shadow-lg"
              mirrored={true}
            />
            {!brainReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
                <p className="text-white">Loading vibe checker... 🧠</p>
              </div>
            )}
          </div>
          <button
            onClick={takeSelfie}
            disabled={isChecking || !brainReady}
            className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400"
          >
            {isChecking ? 'Checking your vibe...' : 'Check My Vibe'}
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
            placeholder="What's your vibe rn? (e.g., 'I'm super hyped and ready to party!')"
            className="w-full h-32 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={checkVibeFromText}
            disabled={!userText.trim()}
            className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
          >
            Find My Vibe Music
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