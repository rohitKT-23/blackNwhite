from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import io
import json

# Initialize FastAPI
app = FastAPI()

# Load the deepfake detection model
MODEL = "not-lain/deepfake"
classifier = pipeline(model=MODEL, trust_remote_code=True)

@app.get("/ping")
def ping():
    return {"message": "Deepfake detection service is running!"}

@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    # Read the image file as bytes
    image_data = await image.read()
    image_bytes = io.BytesIO(image_data)  # Convert to BytesIO

    # Perform inference using classifier.predict()
    result = classifier.predict(image_bytes)

    # Inspect the result (for debugging purposes)
    print("Raw result from classifier:", result)

    # Extract only the relevant part (confidences)
    if isinstance(result, dict) and "confidences" in result:
        # Return only the confidences part
        return {"result": result["confidences"]}
    else:
        # Handle unexpected output format
        return {"error": "Unexpected output format from classifier."}

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="54.163.173.200", port=8002)
