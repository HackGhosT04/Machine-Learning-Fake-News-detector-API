import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

# ----- NLTK setup -----
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# ----- FastAPI setup -----
MODEL_PATH = 'fake_news_detector.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

app = FastAPI(title="Fake News Predictor API", description="Classifies news as real or fake.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load model and vectorizer at startup
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: Dict[str, float]
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    processed = preprocess_text(request.text)
    vec_input = vectorizer.transform([processed])
    pred = model.predict(vec_input)[0]
    probs = model.predict_proba(vec_input)[0]
    label_to_key = {0: "real", 1: "fake"}
    confidence = max(probs)
    
    return PredictionResponse(
        prediction=pred,
        probabilities={label_to_key[i]: float(prob) for i, prob in enumerate(probs)},
        confidence=float(confidence)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
