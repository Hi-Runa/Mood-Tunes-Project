export interface SpotifyCredentials {
  clientId: string;
  clientSecret: string;
}

export interface EmotionResult {
  emotion: string;
  confidence: number;
  description?: string;
}

export interface SpotifyTrack {
  id: string;
  name: string;
  artists: string[];
  albumArt: string;
  previewUrl: string | null;
}

export interface User {
  email: string;
  username: string;
}