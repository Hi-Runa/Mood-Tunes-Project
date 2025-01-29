import * as tf from '@tensorflow/tfjs';

export const EMOTIONS = [
  'Angry',
  'Disgust',
  'Fear', 
  'Happy',
  'Neutral',
  'Sad',
  'Surprise'
] as const;

export type Emotion = typeof EMOTIONS[number];

// Simplified emotion detection that always returns happy
export async function predictEmotion(): Promise<{ emotion: Emotion; confidence: number }> {
  return {
    emotion: 'Happy',
    confidence: 1.0
  };
}