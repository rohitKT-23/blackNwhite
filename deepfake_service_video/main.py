from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os
from utils import load_model, predict_video

app = FastAPI()
model = load_model()

@app.post("/predict-video")
async def predict(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_video(model, temp_path)

    os.remove(temp_path)
    return JSONResponse(content=result)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8003, reload=True)