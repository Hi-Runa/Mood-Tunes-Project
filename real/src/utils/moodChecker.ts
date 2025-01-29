export const VIBES = [
  'Angry',
  'Disgust',
  'Fear', 
  'Happy',
  'Neutral',
  'Sad',
  'Surprise'
] as const;

export type Vibe = typeof VIBES[number];

// Simplified vibe check that always returns happy
export async function checkVibe(): Promise<{ vibe: Vibe; sureness: number }> {
  return {
    vibe: 'Happy',
    sureness: 1.0
  };
}