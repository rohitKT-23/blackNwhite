from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the image classification model
MODEL = "prithivMLmods/AI-vs-Deepfake-vs-Real"
classifier = pipeline("image-classification", model=MODEL)

# Define request body schema
class ImageInput(BaseModel):
    image: UploadFile = File(...)

# Health check endpoint
@app.get("/ping")
def ping():
    return {"message": "AI vs Deepfake vs Real image classification service is running!"}

# Predict image category
@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    # Read the image file
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Perform inference
    result = classifier(image)[0]  # Get the top result
    return {
        "label": result["label"],
        "confidence": result["score"]
    }

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)