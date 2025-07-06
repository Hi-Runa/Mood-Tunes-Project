# Mood Tunes

**Mood Tunes** is a web-based application that helps users discover music based on their emotional state. Whether you're feeling tired, happy, sad, or stressed, Mood Tunes finds the perfect songs to match your mood.

---

## How It Works

Mood Tunes offers two ways for users to input their mood:

### 1. Text-Based Mood Detection
Users can describe how they feel in natural language (e.g., "I'm having a long day").  
The system processes the input to interpret the underlying emotion — for example, "tired" — and recommends uplifting music accordingly.

### 2. Facial Emotion Recognition
Users can scan their face using their device’s webcam.  
A Python `cv2` script captures the image and passes it to a custom-built TensorFlow emotion recognition model.  
The model classifies the face into one of seven emotional categories (e.g., happy, tired, sad, angry).

---

## Music Recommendation

Once the user's emotion is identified (via text or face), Mood Tunes uses the Spotify API to find a curated list of songs that match that mood. These recommendations are displayed to the user in a clean and accessible interface.

---

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Computer Vision: OpenCV (`cv2`)
- Machine Learning: TensorFlow (Custom Emotion Classification Model)
- Music API: Spotify Web API

---

## Features

- Detect mood via natural language text
- Detect emotion through webcam facial analysis
- Real-time music recommendations
- Custom-trained emotion recognition model
- Clean and simple user interface
