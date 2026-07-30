from fastapi import FastAPI, UploadFile, File
from inference import predict_deepfake
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# implementing cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_deepfake(image_bytes)
    return result