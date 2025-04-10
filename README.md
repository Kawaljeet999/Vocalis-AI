# Vocalis-AI

Vocalis-AI is an AI-based web application for validating and predicting the emotion of animal sounds using advanced machine learning techniques and real-time frontend interaction. It predicts the **emotion** with **confidence score** and returns a **human-like phrase** based on the predicted emotion. It also supports **real-time audio analysis** for instant feedback.

---

## 🔍 Model Clarification

- Frontend: HTML, CSS, JavaScript used for UI and audio upload.
- Animal audio file is uploaded by the user.
- User selects the target animal from a dropdown.
- Audio file name must match the selected animal.
- If mismatched (e.g. user selects "cat" but uploads a dog audio named "cat"), the audio is rejected.
- Validation handled using a Light BGM model that checks the audio-animal match.
- If valid, classification proceeds to emotion detection pipeline.
- MFCC feature extraction is done using Librosa.
- KMeans is used to cluster audio data by emotion.
- KNN is used to evaluate closeness of test audio to an emotion cluster.
- LSTM model is used for final emotion classification.
- Training accuracy: 97%, F1 Score: 95%
- Validation accuracy: 87%, F1 Score: 87%
- Real-time audio mode is included.
- Frontend supports dark mode toggle.
- Predicts emotion with a confidence score and provides a human-like phrase corresponding to that emotion.
- Supports real-time audio analysis.

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

### Option 2: Fork the repository (without cloning)

You can also **fork** the repository to your GitHub account and follow the same installation steps in your local environment.

---

## 🖥️ Access the App

After running the server, go to:
- Main App: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 How to Use

- Locate and open the `index.html` file in a browser.
- Select the animal from the dropdown.
- Upload an animal audio file.
- If audio is valid, emotion is predicted and displayed.
- If unknown or invalid, audio is rejected.
- Optionally, use real-time recording mode for live predictions.
- Use the dark mode toggle for UI theme preference.

---

