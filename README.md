
# 🐾 Vocalis-AI

**Vocalis-AI** is an AI-based web application for validating and predicting the emotion of animal sounds using advanced machine learning techniques and real-time frontend interaction. It predicts the **emotion** with **confidence score** and returns a **human-like phrase** based on the predicted emotion. It also supports **real-time audio analysis** for instant feedback.

---

## 🔍 Model Clarification

- **Frontend**: HTML, CSS, JavaScript used for UI and audio upload.
- **Audio Input Options**:
  - Upload an audio file.
  - Record audio in real-time.
- **Animal Match Validation**:
  - User selects the target animal from a dropdown.
  - Audio file name must match the selected animal.
  - If mismatched (e.g. user selects "cat" but uploads a dog audio named "cat"), the audio is rejected.
  - Validation handled using a Light BGM model that checks the audio-animal match.
- **Allowed File Formats**: Only .wav and .mp3    files are supported.
- **Feature Extraction**: MFCC via Librosa.
- **Emotion Classification Pipeline**:
  - KMeans is used to cluster audio data by emotion.
  - KNN is used to evaluate closeness of test audio to an emotion cluster.
  - LSTM model is used for final emotion classification.
- **Output**:
  - Predicts emotion with a confidence score.
  - Provides a human-like phrase corresponding to that emotion.
- **Real-time Audio Handling**:
  - Real-time audio mode is included.
  - When both upload and real-time are used, the **latest** input is used for prediction.
- **Frontend**:
  - Dark mode toggle included.

---


## 📊 Model Performance

| Metric              | Score |
|---------------------|-------|
| Training Accuracy   | 97%   |
| Training F1 Score   | 95%   |
| Validation Accuracy | 87%   |
| Validation F1 Score | 87%   |

---


## 🎙️ Audio Input Modes

- **Upload Mode**:  
  User selects an animal and uploads audio (`cat_*.wav` for Cat). A **LightGBM** model ensures the audio matches the selected animal. Mismatches are rejected.

- **Real-Time Mode**:  
  User records audio in-browser. Recording is sent to the server and processed like uploads.

---

## 🔁 Prediction Workflow

- **Last-in Wins**:  
  Only the most recent input (upload or recording) is processed to avoid prediction conflicts.

- **Validation**:  
  LightGBM checks if audio features match the selected animal. If not, result is: `"emotion": "Unknown or Invalid"`.

- **Preprocessing**:  
  Audio is:
  - Converted to mono  
  - Resampled to 16kHz  
  - Trimmed, normalized, and converted to MFCC features

- **Emotion Detection Pipeline**:
  - **KMeans** clusters known emotion profiles  
  - **KNN** compares test audio to emotion clusters  
  - **LSTM** classifies final emotion

- **Output**:
  - Returns emotion label + confidence score  
  - Human-like phrase is generated (e.g., *"I'm feeling down and lonely..."*)  
  - Confidence for all classes included

---

## ⚙️ Steps to Run the Project

### Option 1: Clone the repository

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/Vocalis-AI.git
cd Vocalis-AI

# 2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the FastAPI server
python -m uvicorn main:app --reload
```

### Option 2: Fork the repository

You can also **fork** the repository to your GitHub account and follow the same installation steps in your local environment.

---

## 🖥️ Access the App After Setup Locally

- Main App: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---


## 🌐 Hosting

The application is live and accessible at:

👉 🔗 [Try Vocalis AI Live](https://vocalis-ai-r9co.onrender.com/)

>> ⚠️ **Note**  
>> **The live link may take a few seconds to start due to server spin-up time on free hosting.**  
>> **Please be patient. For the best experience, consider running the app locally, which offers faster performance.**

---


## 📦 Extra Info

- Ensure filenames follow format: `animal_*.wav`.
- Audio is preprocessed (resampled, trimmed).
- Frontend and backend communicate in real time.

---

## 🤝 Contribute

Pull requests and suggestions are welcome. Fork the repo and help improve!

---

## 📄 License

MIT License – see `LICENSE` file.
