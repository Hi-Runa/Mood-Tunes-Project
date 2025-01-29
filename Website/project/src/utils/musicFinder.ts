/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        This file contains TypeScript code for fetching Spotify playlists based on user emotions using the Spotify API.
*/


import { EmotionResult } from '../types';

const spotifyCreds = {
  clientId: '9e48df172d37493eb42ffbe9c061131d',
  clientSecret: '12035761b48243689460e6631b332c2c'
};

let accessKey: string | null = null;

async function getSpotifyKey(): Promise<string> {
  if (accessKey) return accessKey;

  const response = await fetch('https://accounts.spotify.com/api/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      Authorization: 'Basic ' + btoa(spotifyCreds.clientId + ':' + spotifyCreds.clientSecret),
    },
    body: 'grant_type=client_credentials',
  });

  const data = await response.json();
  accessKey = data.access_token;
  return accessKey;
}

// Cool playlists for different moods
const MOOD_PLAYLISTS = {
  'Angry': '37i9dQZF1EIhuCNl2WSFYd',  // Rage Mode 😠
  'Disgust': '37i9dQZF1EIgNZCaOGb0Mi', // Ugh Vibes
  'Fear': '37i9dQZF1EIgqVxMzROxZ5',   // Scared Hours
  'Happy': '37i9dQZF1EIgG2NEOhqsD7',  // Happy Hits
  'Neutral': '37i9dQZF1EIcczFDmqxd5R', // Chill Zone
  'Sad': '37i9dQZF1EIdChYeHNDfK5',    // In My Feels
  'Surprise': '37i9dQZF1EIePuVyHKBQqp' // Mind Blown
};

export async function findTunes(mood: EmotionResult): Promise<string> {
  // Check our playlist stash first
  if (MOOD_PLAYLISTS[mood.emotion as keyof typeof MOOD_PLAYLISTS]) {
    return MOOD_PLAYLISTS[mood.emotion as keyof typeof MOOD_PLAYLISTS];
  }

  // If we don't have a playlist, let's search for one
  const key = await getSpotifyKey();
  const searchWords = `${mood.emotion} mood`;

  const response = await fetch(
    `https://api.spotify.com/v1/search?q=${encodeURIComponent(searchWords)}&type=playlist&limit=50`,
    {
      headers: {
        Authorization: `Bearer ${key}`,
      },
    }
  );

  const data = await response.json();
  
  if (data.playlists && data.playlists.items.length > 0) {
    // Pick a random playlist from the top 10
    const randomPick = Math.floor(Math.random() * Math.min(10, data.playlists.items.length));
    return data.playlists.items[randomPick].id;
  }
  
  return MOOD_PLAYLISTS['Neutral']; // Default to chill vibes
}