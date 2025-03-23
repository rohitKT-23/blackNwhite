from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
import io

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
    # Read image bytes
    image_data = await image.read()
    image_bytes = io.BytesIO(image_data)

    # Open with PIL and convert to RGB to ensure 3 channels
    pil_image = Image.open(image_bytes).convert("RGB")

    # Convert PIL image back to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')  # or 'JPEG' depending on your needs
    img_byte_arr.seek(0)  # Move to the beginning of the byte stream

    # Call classifier on the byte stream
    result = classifier.predict(img_byte_arr)

    # Debug print
    print("Raw result from classifier:", result)

    # Extract only the confidences
    if isinstance(result, dict) and "confidences" in result:
        return {"result": result["confidences"]}
    else:
        return {"error": "Unexpected output format from classifier."}
# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)