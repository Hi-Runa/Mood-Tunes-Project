import React, { useState, useEffect } from 'react';
import { Music, RefreshCw } from 'lucide-react';
import { EmotionResult } from '../types';
import { getContextualPlaylist } from '../utils/spotify';

interface Props {
  emotion: EmotionResult | null;
}

export function MusicRecommendations({ emotion }: Props) {
  const [currentPlaylistIndex, setCurrentPlaylistIndex] = useState(0);
  const [playlistId, setPlaylistId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function fetchPlaylist() {
      if (emotion) {
        setLoading(true);
        try {
          const playlist = await getContextualPlaylist(emotion);
          setPlaylistId(playlist);
        } catch (error) {
          console.error('Error fetching playlist:', error);
        } finally {
          setLoading(false);
        }
      }
    }
    fetchPlaylist();
  }, [emotion, currentPlaylistIndex]);

  if (!emotion || !playlistId) return null;

  const handleNextPlaylist = () => {
    setCurrentPlaylistIndex(prev => prev + 1);
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Music className="text-purple-600" size={24} />
            <div>
              <h2 className="text-xl font-semibold">
                Your Personalized Music
              </h2>
              {emotion.description && (
                <p className="text-sm text-gray-600 mt-1">
                  {emotion.description}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={handleNextPlaylist}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-purple-100 text-purple-600 rounded-lg hover:bg-purple-200 transition-colors disabled:opacity-50"
          >
            <RefreshCw size={20} />
            Try Different Songs
          </button>
        </div>
        
        {loading ? (
          <div className="h-[380px] flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
          </div>
        ) : (
          <iframe
            src={`https://open.spotify.com/embed/playlist/${playlistId}`}
            width="100%"
            height="380"
            frameBorder="0"
            allow="encrypted-media"
            className="rounded-lg"
          />
        )}
      </div>
    </div>
  );
}