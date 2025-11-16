from fastapi import FastAPI, UploadFile, File
from deployment_preprocessing import preprocess_pil_image
from model_tf import predict_emotion_tf

app = FastAPI(
    title="Emotion Detection API (TF)",
    version="1.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    face = preprocess_pil_image(image_bytes)
    
    if face is None:
        return {"error": "No face detected"}

    emotion, confidence, all_emotions = predict_emotion_tf(face)

    return {
        "emotion": emotion,
        "confidence": round(confidence, 3),
        "all_emotions": all_emotions
    }

@app.get("/")
def home():
    return {"message": "TensorFlow Emotion Detection API Running"}
