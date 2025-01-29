/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        This file is responsible for fetching the spotify API token using the 
        clientID and clientSecret Hiruna created and uses the token that spotify authorizes and provides  
*/

import { SpotifyCredentials, EmotionResult } from '../types';

const credentials: SpotifyCredentials = {
  clientId: '9e48df172d37493eb42ffbe9c061131d',
  clientSecret: '12035761b48243689460e6631b332c2c'
};

let accessToken: string | null = null;

async function getAccessToken(): Promise<string> {
  if (accessToken) return accessToken;

  const response = await fetch('https://accounts.spotify.com/api/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      Authorization: 'Basic ' + btoa(credentials.clientId + ':' + credentials.clientSecret),
    },
    body: 'grant_type=client_credentials',
  });

  const data = await response.json();
  accessToken = data.access_token;
  return accessToken;
}

// Map emotions to Spotify playlists
const EMOTION_PLAYLISTS = {
  'Angry': '37i9dQZF1EIhuCNl2WSFYd',  // Rage Beats
  'Disgust': '37i9dQZF1EIgNZCaOGb0Mi', // Dark & Stormy
  'Fear': '37i9dQZF1EIgqVxMzROxZ5',   // Anxiety Relief
  'Happy': '37i9dQZF1EIgG2NEOhqsD7',  // Happy Hits
  'Neutral': '37i9dQZF1EIcczFDmqxd5R', // Chill Vibes
  'Sad': '37i9dQZF1EIdChYeHNDfK5',    // Sad Songs
  'Surprise': '37i9dQZF1EIePuVyHKBQqp' // Feel Good Piano
};

export async function getContextualPlaylist(emotion: EmotionResult): Promise<string> {
  // Check if we have a preset playlist for the emotion
  if (EMOTION_PLAYLISTS[emotion.emotion as keyof typeof EMOTION_PLAYLISTS]) {
    return EMOTION_PLAYLISTS[emotion.emotion as keyof typeof EMOTION_PLAYLISTS];
  }

  // Fallback to search if no preset playlist
  const token = await getAccessToken();
  const searchQuery = `${emotion.emotion} mood`;

  const response = await fetch(
    `https://api.spotify.com/v1/search?q=${encodeURIComponent(searchQuery)}&type=playlist&limit=50`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );

  const data = await response.json();
  
  if (data.playlists && data.playlists.items.length > 0) {
    const randomIndex = Math.floor(Math.random() * Math.min(10, data.playlists.items.length));
    return data.playlists.items[randomIndex].id;
  }
  
  return EMOTION_PLAYLISTS['Neutral']; // Fallback to neutral playlist
}