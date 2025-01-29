/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        This file is the component that detects user emotions through webcam or text input.
*/

import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Type } from 'lucide-react';
import { EmotionResult } from '../types';
import { loadModel, predictEmotion, EMOTIONS } from '../utils/emotionDetection';

interface Props {
  onEmotionDetected: (emotion: EmotionResult) => void;
}

export function EmotionDetector({ onEmotionDetected }: Props) {
  const [mode, setMode] = useState<'camera' | 'text' | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const webcamRef = useRef<Webcam>(null);
  const [textInput, setTextInput] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    async function initModel() {
      try {
        const model = await loadModel();
        setModelLoaded(!!model);
      } 
      catch (error) {
        console.error('Error loading model:', error);
      }
    }
    initModel();
  }, []);

  const analyzeContext = useCallback((text: string) => {
    // Comprehensive emotion patterns with associated words and weights
    const emotionPatterns = [
      {
        emotion: 'Happy',
        patterns: [
          { words: /\b(happy|joy|excited|delighted|great|awesome|amazing|love|wonderful|blessed|cheerful|content|ecstatic|elated|fantastic|glad|grateful|jubilant|overjoyed|pleased|thrilled|upbeat|blissful|bright|radiant|sunny|optimistic|positive|hopeful|enthusiastic|energetic|vibrant|lively|playful|carefree|peaceful|satisfied|fulfilled|accomplished|proud|confident|inspired|motivated|refreshed|renewed|alive|glowing|beaming)\b/i, weight: 1 },
          { words: /\b(smile|laugh|fun|party|celebrate|dance|sing|enjoy|win|success|achievement|breakthrough|milestone|victory|triumph|blessing|miracle|perfect|excellent|superb|outstanding|brilliant|incredible|extraordinary|remarkable|spectacular|magnificent|marvelous|splendid)\b/i, weight: 0.8 },
          { words: /😊|😃|😄|🎉|❤️|🥳|😁|😆|😍|🥰|💖|✨|🌟|☀️|🎈|🎊|🎯|🏆|👑|💫|🌈|🦋|🎵|💝|🌺|🌸|⭐️/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Sad',
        patterns: [
          { words: /\b(sad|down|depressed|unhappy|miserable|heartbroken|gloomy|melancholy|sorrowful|grief|despair|devastated|hopeless|discouraged|defeated|disappointed|rejected|abandoned|alone|isolated|empty|numb|broken|crushed|shattered|worthless|insignificant|inadequate|failure|lost|confused|uncertain|unfulfilled|regretful|guilty|ashamed|hurt|wounded|damaged|scarred)\b/i, weight: 1 },
          { words: /\b(crying|tears|lonely|miss|pain|suffering|agony|anguish|torment|struggle|hardship|difficulty|trouble|problem|crisis|tragedy|trauma|darkness|shadow|void|abyss|pit|burden|weight|pressure|stress|tension|anxiety|worry|concern|fear|dread|desperation|helplessness)\b/i, weight: 0.8 },
          { words: /😢|😭|💔|😞|😔|😪|😥|😰|😿|🥀|☔️|⛈️|🌧️|🖤|😕|🤕|😫|😖|🥺|💧|📉|🕯️/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Angry',
        patterns: [
          { words: /\b(angry|mad|furious|rage|hate|annoyed|irritated|frustrated|enraged|outraged|livid|hostile|aggressive|violent|fierce|intense|bitter|resentful|vengeful|spiteful|contempt|disgusted|repulsed|revolted|offended|insulted|disrespected|betrayed|cheated|deceived|manipulated|used|abused|violated|wronged|mistreated|hurt)\b/i, weight: 1 },
          { words: /\b(upset|agitated|disturbed|provoked|triggered|heated|boiling|burning|exploding|erupting|seething|fuming|steaming|raging|storming|thundering|roaring|screaming|shouting|yelling|cursing|swearing|threatening|attacking|fighting|battling|struggling|conflicting|clashing|opposing|resisting)\b/i, weight: 0.8 },
          { words: /😠|😡|💢|🤬|😤|👿|💥|⚡️|🔥|💪|👊|✊|🗯️|⚔️|🛡️|🏹|🎯|🚫|⛔️|🆘|⚠️|❌/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Fear',
        patterns: [
          { words: /\b(scared|afraid|fearful|anxious|worried|nervous|terrified|horrified|petrified|panicked|alarmed|startled|shocked|stunned|paralyzed|frozen|trembling|shaking|quivering|quaking|dreading|fearing|anticipating|expecting|suspecting|doubting|uncertain|insecure|vulnerable|exposed|threatened|endangered|imperiled|jeopardized|risked)\b/i, weight: 1 },
          { words: /\b(stress|panic|terror|dread|fright|horror|alarm|danger|threat|risk|hazard|emergency|crisis|disaster|catastrophe|calamity|tragedy|nightmare|phobia|trauma|anxiety|worry|concern|apprehension|hesitation|reluctance|resistance|avoidance|escape|flight|retreat|withdrawal)\b/i, weight: 0.8 },
          { words: /😨|😰|😱|😖|😣|😩|🥶|😵|🙀|💀|☠️|👻|🕷️|🕸️|🌑|🌚|🏃|🚪|🔒|🚨|🆘|⚠️/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Surprise',
        patterns: [
          { words: /\b(surprised|shocked|amazed|astonished|astounded|stunned|startled|taken aback|speechless|overwhelmed|overcome|thunderstruck|flabbergasted|dumbfounded|bewildered|confused|perplexed|puzzled|baffled|mystified|wondering|questioning|doubting|disbelieving|incredulous|skeptical|suspicious|uncertain|unsure|hesitant)\b/i, weight: 1 },
          { words: /\b(unexpected|unbelievable|incredible|extraordinary|remarkable|outstanding|exceptional|unusual|unique|special|different|strange|odd|peculiar|mysterious|magical|miraculous|wonderful|marvelous|fantastic|amazing|awesome|spectacular|stunning|striking|impressive|dramatic|sensational)\b/i, weight: 0.8 },
          { words: /😮|😲|😱|🤯|😵|😳|🤪|😜|🤨|🧐|❓|❔|💫|✨|🎯|🎲|🎪|🎭|🎨|🎬|🎼|🎵/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Disgust',
        patterns: [
          { words: /\b(disgust|gross|repulsed|repelled|revolted|sickened|nauseated|queasy|uncomfortable|unpleasant|disagreeable|objectionable|offensive|repugnant|repulsive|distasteful|unsavory|unpalatable|foul|nasty|dirty|filthy|contaminated|polluted|tainted|spoiled|rotten|decayed|decomposed|putrid|fetid)\b/i, weight: 1 },
          { words: /\b(sick|nauseous|ill|unwell|diseased|infected|contagious|toxic|poisonous|venomous|harmful|dangerous|hazardous|risky|threatening|menacing|sinister|evil|wicked|vile|corrupt|depraved|immoral|unethical|wrong|bad|terrible|horrible|awful|dreadful|appalling|shocking)\b/i, weight: 0.8 },
          { words: /🤢|🤮|🤧|😖|🥴|😫|🦠|☣️|☢️|⚠️|🚫|⛔️|💀|☠️|🕷️|🕸️|🗑️|🧹|🧼|🧽|🚽|💩/g, weight: 0.9 }
        ]
      },
      {
        emotion: 'Neutral',
        patterns: [
          { words: /\b(okay|fine|alright|normal|regular|usual|typical|standard|ordinary|common|average|moderate|balanced|stable|steady|constant|consistent|unchanged|unaffected|indifferent|neutral|impartial|objective|fair|reasonable|sensible|practical|logical|rational|sound|valid)\b/i, weight: 1 },
          { words: /\b(calm|quiet|peaceful|tranquil|serene|relaxed|composed|collected|controlled|reserved|restrained|contained|measured|deliberate|thoughtful|contemplative|reflective|meditative|mindful|aware|conscious|present|centered|grounded|balanced|harmonious)\b/i, weight: 0.8 },
          { words: /😐|😶|😑|🤔|💭|💡|⚖️|🎯|📊|📈|📉|🔄|⏳|⌛️|🕰️|⏰|📅|📆|📋|📝|✍️|💼/g, weight: 0.9 }
        ]
      }
    ];

    // Sentiment analysis for text without explicit emotion words
    const sentimentPatterns = {
      positive: /\b(good|nice|great|cool|lit|fire|vibe|chill|relaxed|peaceful|energetic|motivated|productive|accomplished|proud|blessed|fantastic|wonderful|amazing|awesome|excellent|brilliant|outstanding|remarkable|incredible|extraordinary|spectacular|magnificent|marvelous|splendid|superb|terrific|fabulous|phenomenal)\b/i,
      negative: /\b(bad|not good|meh|tired|exhausted|bored|stuck|overwhelmed|stressed|frustrated|terrible|horrible|awful|dreadful|poor|disappointing|unsatisfactory|unpleasant|unfortunate|unfavorable|inadequate|insufficient|mediocre|subpar|inferior|deficient|lacking|missing|wanting|failing)\b/i,
      intensity: /\b(very|really|so|super|extremely|totally|absolutely|completely|entirely|utterly|thoroughly|deeply|profoundly|intensely|tremendously|immensely|incredibly|extraordinarily|exceptionally|remarkably|notably|particularly|especially|significantly|substantially|considerably|markedly)\b/i
    };

    // Calculate emotion scores
    const scores = emotionPatterns.map(({ emotion, patterns }) => {
      let score = 0;
      patterns.forEach(({ words, weight }) => {
        const matches = (text.match(words) || []).length;
        score += matches * weight;
      });
      return { emotion, score };
    });

    // If we found explicit emotions, use the highest scoring one
    const maxScore = Math.max(...scores.map(s => s.score));
    if (maxScore > 0) {
      const topEmotion = scores.find(s => s.score === maxScore)!;
      return {
        emotion: topEmotion.emotion,
        confidence: Math.min(0.9, 0.5 + (maxScore * 0.1)),
        description: `Music for when you're feeling ${topEmotion.emotion.toLowerCase()}`
      };
    }

    // If no explicit emotions, analyze sentiment
    const positiveMatches = (text.match(sentimentPatterns.positive) || []).length;
    const negativeMatches = (text.match(sentimentPatterns.negative) || []).length;
    const intensityMatches = (text.match(sentimentPatterns.intensity) || []).length;

    if (positiveMatches > negativeMatches) {
      return {
        emotion: 'Happy',
        confidence: Math.min(0.8, 0.5 + (positiveMatches * 0.1) + (intensityMatches * 0.05)),
        description: 'Music to match your positive vibes'
      };
    } 
    else if (negativeMatches > positiveMatches) {
      return {
        emotion: 'Sad',
        confidence: Math.min(0.8, 0.5 + (negativeMatches * 0.1) + (intensityMatches * 0.05)),
        description: 'Music to help lift your mood'
      };
    }

    // Default to neutral if we can't determine sentiment
    return {
      emotion: 'Neutral',
      confidence: 0.5,
      description: 'Music to match your current mood'
    };
  }, []);

  const captureImage = useCallback(async () => {
    if (!webcamRef.current || !modelLoaded) return;
    
    setIsLoading(true);
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      const img = new Image();
      img.src = imageSrc;
      await img.decode();

      const { emotion, confidence } = await predictEmotion(img);
      
      onEmotionDetected({
        emotion,
        confidence,
        description: `Music for when you're feeling ${emotion.toLowerCase()}`
      });
    } 
    catch (error) {
      console.error('Error detecting emotion:', error);
    } 
    finally {
      setIsLoading(false);
    }
  }, [modelLoaded, onEmotionDetected]);

  const detectEmotionFromText = useCallback(() => {
    if (!textInput.trim()) return;
    const analysis = analyzeContext(textInput);
    onEmotionDetected(analysis);
  }, [textInput, analyzeContext, onEmotionDetected]);

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      {!mode ? (
        <div className="flex gap-4 justify-center">
          <button
            onClick={() => setMode('camera')}
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
      ) : mode === 'camera' ? (
        <div className="space-y-4">
          <div className="relative">
            <Webcam
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="w-full rounded-lg shadow-lg"
              mirrored={true}
            />
            {!modelLoaded && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
                <p className="text-white">Loading emotion detection model...</p>
              </div>
            )}
          </div>
          <button
            onClick={captureImage}
            disabled={isLoading || !modelLoaded}
            className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400"
          >
            {isLoading ? 'Detecting...' : 'Detect Mood'}
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
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="How are you feeling? (e.g., 'I'm feeling energetic and ready to take on the day!')"
            className="w-full h-32 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={detectEmotionFromText}
            disabled={!textInput.trim()}
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