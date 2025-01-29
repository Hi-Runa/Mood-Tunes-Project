import * as tf from '@tensorflow/tfjs';

// Initialize TensorFlow.js
let model: tf.LayersModel | null = null;

// Get the model path from the environment or use a default path
const MODEL_PATH = window.MODEL_PATH || '';

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

export async function loadModel() {
  if (!MODEL_PATH) {
    console.warn('Model path not set. Please set window.MODEL_PATH to your .keras model location.');
    return null;
  }
  
  try {
    // Load and warm up the model
    model = await tf.loadLayersModel(MODEL_PATH);
    
    // Warm up the model with a dummy tensor
    const dummyTensor = tf.zeros([1, 48, 48, 1]);
    await model.predict(dummyTensor);
    dummyTensor.dispose();
    
    console.log('Model loaded and warmed up successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    return null;
  }
}

export async function preprocessImage(image: HTMLImageElement): Promise<tf.Tensor4D> {
  return tf.tidy(() => {
    // Convert image to tensor and preprocess
    const tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([48, 48])
      .mean(2)
      .expandDims(2)
      .expandDims(0)
      .toFloat()
      .div(255.0);
      
    return tensor as tf.Tensor4D;
  });
}

export async function predictEmotion(image: HTMLImageElement): Promise<{ emotion: Emotion; confidence: number }> {
  if (!model) {
    console.warn('Model not loaded. Please ensure window.MODEL_PATH is set correctly.');
    return { emotion: 'Neutral', confidence: 0.5 };
  }

  return tf.tidy(() => {
    const tensor = preprocessImage(image);
    const predictions = model!.predict(tensor) as tf.Tensor;
    const probabilities = predictions.dataSync();
    
    // Get the index of the highest probability
    const maxProbabilityIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
    
    return {
      emotion: EMOTIONS[maxProbabilityIndex],
      confidence: probabilities[maxProbabilityIndex]
    };
  });
}