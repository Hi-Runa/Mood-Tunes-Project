import * as tf from '@tensorflow/tfjs';

let brainModel: tf.LayersModel | null = null;

// Where's our model at?
const MODEL_PATH = window.MODEL_PATH || '';

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

export async function startBrain() {
  if (!MODEL_PATH) {
    console.warn('Yo, where\'s the model at? Set window.MODEL_PATH to your .keras file!');
    return null;
  }
  
  try {
    // Load up our brain
    brainModel = await tf.loadLayersModel(MODEL_PATH);
    
    // Wake it up
    const testRun = tf.zeros([1, 48, 48, 1]);
    await brainModel.predict(testRun);
    testRun.dispose();
    
    console.log('Brain is ready to rock! 🧠');
    return brainModel;
  } catch (error) {
    console.error('Oof, brain failed to load:', error);
    return null;
  }
}

export async function makeImageCool(selfie: HTMLImageElement): Promise<tf.Tensor4D> {
  return tf.tidy(() => {
    const coolPic = tf.browser.fromPixels(selfie)
      .resizeNearestNeighbor([48, 48])
      .mean(2)
      .expandDims(2)
      .expandDims(0)
      .toFloat()
      .div(255.0);
      
    return coolPic as tf.Tensor4D;
  });
}

export async function checkVibe(selfie: HTMLImageElement): Promise<{ vibe: Vibe; sureness: number }> {
  if (!brainModel) {
    console.warn('Hold up! Brain isn\'t ready yet. Make sure window.MODEL_PATH is set!');
    return { vibe: 'Neutral', sureness: 0.5 };
  }

  return tf.tidy(() => {
    const coolPic = makeImageCool(selfie);
    const vibeCheck = brainModel!.predict(coolPic) as tf.Tensor;
    const vibeScores = vibeCheck.dataSync();
    
    // Find the strongest vibe
    const bestVibeIndex = vibeScores.indexOf(Math.max(...Array.from(vibeScores)));
    
    return {
      vibe: VIBES[bestVibeIndex],
      sureness: vibeScores[bestVibeIndex]
    };
  });
}