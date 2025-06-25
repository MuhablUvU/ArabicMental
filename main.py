from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pickle
from typing import List
import os

# Set TensorFlow log level (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

# Initialize FastAPI app
app = FastAPI(
    title="Arabic NLP Model API",
    description="API for predicting Arabic text classes",
    version="1.0.0"
)

# Constants
MODEL_PATH = "arabic_nlp_optimized.keras"
TOKENIZER_PATH = "tokenizer_optimized.pickle"
ENCODER_PATH = "label_encoder_optimized.pickle"
MAX_LEN = 150

# Request and response models
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: List[dict]

# Load model, tokenizer, and encoder
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    # Get all class names from encoder
    if hasattr(encoder, "classes_"):
        CLASS_NAMES = list(encoder.classes_)
    else:
        CLASS_NAMES = []
    print("✅ Model, tokenizer, and encoder loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model, tokenizer, or encoder: {e}")
    raise

def preprocess_text(text: str) -> np.ndarray:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

def get_predictions(text: str):
    padded = preprocess_text(text)
    preds = model.predict(padded, verbose=0)
    if preds.shape[1] == 1:  # Binary
        confidence = float(preds[0][0])
        label_idx = int(confidence > 0.5)
        predicted_class = encoder.inverse_transform([label_idx])[0]
        top_predictions = [
            {"class": encoder.inverse_transform([0])[0], "confidence": float(1-confidence)*100},
            {"class": encoder.inverse_transform([1])[0], "confidence": float(confidence)*100}
        ]
    else:  # Multi-class
        label_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        predicted_class = encoder.inverse_transform([label_idx])[0]
        # Top predictions sorted
        top_indices = np.argsort(preds[0])[::-1]
        top_predictions = [
            {"class": encoder.inverse_transform([idx])[0], "confidence": float(preds[0][idx])*100}
            for idx in top_indices
        ]
    return predicted_class, confidence*100, top_predictions

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.text or len(request.text.strip()) < 2:
        raise HTTPException(status_code=400, detail="Text is too short.")
    try:
        predicted_class, confidence, top_predictions = get_predictions(request.text)
        return PredictResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_predictions=top_predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Arabic NLP Model API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)