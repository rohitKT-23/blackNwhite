from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Load Fake News Detection Model
MODEL = "jy46604790/Fake-News-Bert-Detect"
classifier = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

# Define request body schema
class NewsInput(BaseModel):
    text: str

# Health check endpoint
@app.get("/ping")
def ping():
    return {"message": "Fake news detection service is running!"}

# Predict fake/real news
@app.post("/predict-text")
def predict_news(news: NewsInput):
    result = classifier(news.text[:500])[0]  # Only first 500 words (model limit)
    return {
        "label": "Fake News" if result["label"] == "LABEL_0" else "Real News",
        "confidence": result["score"]
    }

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.31.1.46", port=8001)
