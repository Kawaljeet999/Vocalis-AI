from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import librosa # Audio Processing
import soundfile as sf # Raeding and Writing Audio Files
import io
import os
import joblib
import time
from pathlib import Path
import shutil
import warnings
import tempfile # For temporary file handling
import wave # For WAV file handling
import struct # For structuring binary data

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

# 🔗 Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🧠 Load models
print("📦 Loading models...")
emotion_model = tf.keras.models.load_model("models/emotion_classifier.keras")
valid_model = joblib.load("models/lightgbm_valid_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ Models loaded successfully.")

# 🏷️ Emotion classes
emotion_labels = ['Happy', 'Sad', 'Fear', 'Angry', 'Neutral']

# 💬 Emotion to phrase mapping
emotion_phrases = {
    "Angry": "I'm really frustrated right now!",
    "Fear": "I'm scared and nervous!",
    "Happy": "I'm so excited and joyful!",
    "Neutral": "I'm feeling okay, nothing special.",
    "Sad": "I'm feeling down and lonely...",
}

# 📁 Audio directory
REAL_TIME_AUDIO_DIR = Path("Real_time_audio")
REAL_TIME_AUDIO_DIR.mkdir(exist_ok=True)

# 📦 Clear real-time audio directory - Enhanced version that removes ALL files
def clear_audio_directory():
    """Remove ALL files in the real-time audio directory"""
    try:
        # This will remove all files of any type in the directory
        for file_path in REAL_TIME_AUDIO_DIR.glob("*.*"):
            try:
                file_path.unlink()
                print(f"Removed old file: {file_path.name}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    except Exception as e:
        print(f"Error clearing directory: {e}")

# Function to check if data has valid WAV header (silently)
def has_valid_wav_header(data): # Validates if binary data has a proper WAV header (starts with 'RIFF' and contains 'WAVE')
    """Check if the data has a valid WAV header"""
    if len(data) < 12:  # Need at least 12 bytes for RIFF header
        return False
    
    # Check for RIFF header
    if data[0:4] != b'RIFF':
        return False
    
    # Check for WAVE format
    if data[8:12] != b'WAVE':
        return False
    
    return True

# Function to fix or create a WAV header (silently)
# Attempts to repair or create a valid WAV header for audio data
# Returns the fixed data or None if it can't be repaired

def ensure_valid_wav(data, sample_rate=16000, channels=1, bits_per_sample=16):
    """
    Ensures the data has a valid WAV header, repairing or creating one if needed.
    Returns fixed data or None if not repairable.
    """
    # First check if it already has a valid header
    if has_valid_wav_header(data):
        return data
    
    # If no valid header, silently create one
    try:
        # Create a BytesIO object to hold the new WAV file
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(bits_per_sample // 8)
                wav_file.setframerate(sample_rate)
                
                # If input looks like raw audio data, write it directly
                if len(data) > 44:  # 44 bytes is minimal WAV header size
                    # Assume the data is raw PCM
                    wav_file.writeframes(data)
                else:
                    return None
            
            # Get the complete WAV file with proper header
            return wav_io.getvalue()
    except Exception as e:
        return None


# Steps include:

# Converting stereo to mono
# Resampling to 16kHz
# Normalizing amplitude
# Trimming silence
# Checking if audio is too quiet
# Extracting MFCC features
# Padding or truncating to uniform length (100 frames)
# Transposing the result
# Direct audio processing from NumPy array approach
def process_raw_audio(samples, sample_rate=16000, n_mfcc=40, max_pad_length=100):
    """
    Process raw audio samples that are already loaded into a NumPy array.
    """
    try:
        # Ensure mono
        if len(samples.shape) > 1 and samples.shape[1] > 1:
            samples = np.mean(samples, axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Normalize
        samples = librosa.util.normalize(samples)
        
        # Trim silence
        samples, _ = librosa.effects.trim(samples, top_db=20)
        
        # Check silence
        if np.max(np.abs(samples)) < 0.01:
            return None
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
        
        # Pad or truncate
        if mfcc.shape[1] < max_pad_length:
            pad_width = max_pad_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_length]
        
        mfcc = mfcc.T  # Shape: (100, 40)
        return mfcc
    except Exception as e:
        return None

# This function tries multiple approaches to process audio data:

# First tries to read the data as a WAV file (after fixing the header if needed)
# If that fails, it writes the data to a temporary PCM file and tries to read it as raw 16-bit audio
# In both cases, it calls process_raw_audio to extract MFCC features
# Returns the MFCC features, the audio samples, and the sample rate
# New quiet audio preprocessing function - no warnings
def preprocess_audio(file_data, target_sr=16000, n_mfcc=40, max_pad_length=100):
    """
    Enhanced audio preprocessing with silent operation.
    """
    try:
        # Method 1: Try direct WAV reading with a silent header check/fix
        fixed_data = ensure_valid_wav(file_data)
        if fixed_data:
            try:
                with io.BytesIO(fixed_data) as audio_io:
                    with wave.open(audio_io, 'rb') as wav:
                        # Get WAV parameters
                        n_channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        framerate = wav.getframerate()
                        n_frames = wav.getnframes()
                        
                        # Read all frames
                        frames = wav.readframes(n_frames)
                        
                        # Convert to numpy array based on sample width
                        if sample_width == 1:  # 8-bit samples
                            samples = np.frombuffer(frames, dtype=np.uint8)
                            samples = samples.astype(np.float32) / 128.0 - 1.0
                        elif sample_width == 2:  # 16-bit samples
                            samples = np.frombuffer(frames, dtype=np.int16)
                            samples = samples.astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit samples
                            samples = np.frombuffer(frames, dtype=np.int32)
                            samples = samples.astype(np.float32) / 2147483648.0
                        else:
                            return None, None, None
                        
                        # Handle stereo
                        if n_channels == 2:
                            samples = samples.reshape(-1, 2)
                            samples = np.mean(samples, axis=1)
                                                
                        # Process the raw audio
                        mfcc = process_raw_audio(samples, framerate, n_mfcc, max_pad_length)
                        if mfcc is not None:
                            return mfcc, samples, framerate
            except Exception:
                pass
        
        # Method 2: Try a different approach with tempfile and PCM conversion (silently)
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_file:
            temp_pcm = temp_file.name
            temp_file.write(file_data)
        
        try:
            # Try loading raw PCM as 16-bit 16kHz mono
            samples = np.memmap(temp_pcm, dtype=np.int16, mode='r')
            samples = samples.astype(np.float32) / 32768.0
            
            # Process the raw audio
            mfcc = process_raw_audio(samples, 16000, n_mfcc, max_pad_length)
            if mfcc is not None:
                return mfcc, samples, 16000
        except Exception:
            pass
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_pcm)
            except:
                pass
        
        return None, None, None
        
    except Exception:
        return None, None, None

# 🗣️ Emotion Phrase
def generate_phrase_from_emotion(emotion: str) -> str: # Returns a text phrase corresponding to the detected emotion
    return emotion_phrases.get(emotion, "")

# 🏠 Frontend route
@app.get("/")
def read_root(): # Serves the main HTML page when users visit the root URL
    return FileResponse("static/index.html")

# 🚀 Prediction route
@app.post("/predict/")
async def predict_emotion(
    file: UploadFile = File(...),
    animal: str = Form(...),
    is_recorded: str = Form(...)
):
    try:
        # ⚠️ Clear the real-time audio directory FIRST
        # This ensures the directory only contains files for the current analysis
        print("🧹 Cleaning real-time audio directory...")
        clear_audio_directory()
        print("✅ Directory cleaned successfully")

        # Read the uploaded file
        file_data = await file.read()
        if len(file_data) == 0:
            return JSONResponse(
                status_code=400, 
                content={"error": "Empty file received"}
            )
            
        print(f"\n📥 File received: {file.filename} ({len(file_data)} bytes) | Animal: {animal} | Is recorded: {is_recorded}")

        # Save the original file
        save_path = REAL_TIME_AUDIO_DIR / f"{animal}_{int(time.time())}.raw"
        with save_path.open("wb") as buffer:
            buffer.write(file_data)
        print(f"💾 Saved data to {save_path}")

        # Step 1: Preprocess
        print("🔍 Beginning audio preprocessing...")
        mfcc, audio, sr = preprocess_audio(file_data)
        if mfcc is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio is silent, corrupted, or cannot be processed."}
            )

        print("✅ Audio preprocessing successful")

        # Step 3: Validate using model
        flat_features = mfcc.reshape(1, -1)
        scaled = scaler.transform(flat_features)
        validation_pred = valid_model.predict(scaled)[0]

        if validation_pred == 0:
            print("🚫 Validation model rejected the audio as UNKNOWN or invalid.")
            
            # Clean up files after rejection
            clear_audio_directory()
            print("🧹 Cleaned up files after rejection")
            
            return JSONResponse(
                content={
                    "emotion": "Unknown or Invalid",
                    "confidence": 0.0
                },
                status_code=200
            )

        print("✅ Audio validation passed!")

        # Step 4: Predict emotion
        emotion_input = scaled.reshape(1, 100, 40)
        emotion_probs = emotion_model.predict(emotion_input)[0]
        predicted_index = np.argmax(emotion_probs)
        confidence = float(np.max(emotion_probs))
        predicted_emotion = emotion_labels[predicted_index]
        phrase = generate_phrase_from_emotion(predicted_emotion)

        print(f"🎯 Emotion: {predicted_emotion} | Confidence: {round(confidence * 100, 2)}%")
        
        # Optionally, if you want to clean up after prediction as well
        # clear_audio_directory()
        # print("🧹 Cleaned up files after successful prediction")

        return JSONResponse(
            content={
                "emotion": predicted_emotion,
                "confidence": round(confidence, 2),
                "all_confidences": {
                    label: round(float(prob), 3)
                    for label, prob in zip(emotion_labels, emotion_probs)
                },
                "phrase": phrase
            }
        )

    except Exception as e:
        print("❌ Unexpected error:", e)
        
        # Make sure to clean up even if there's an error
        clear_audio_directory()
        print("🧹 Cleaned up files after error")
        
        return JSONResponse(content={"error": f"Server error: {str(e)}"}, status_code=500)