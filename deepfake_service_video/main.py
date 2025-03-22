from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import shutil, os
from utils import load_model, predict_video

app = FastAPI()

# Load both models with correct paths
model1 = load_model("genconvit_ed_inference.pth")  # Pass the correct path here
model2 = load_model("genconvit_vae_inference.pth")  # Pass the correct path here

@app.post("/predict-video")
async def predict(
    file: UploadFile = File(...),
    model_choice: str = Query("model1", enum=["model1", "model2"])
):
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"

    # Save the uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Select the model based on the user's choice
    model = model1 if model_choice == "model1" else model2

    result = predict_video(model, temp_path)

    os.remove(temp_path)  # Clean up the temporary file
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="54.163.173.200", port=8003, reload=True)
