from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI(title="Plant Disease Detection API")

# -----------------------------
# Load Model
# -----------------------------
try:
    model = tf.keras.models.load_model("plantAID_final.keras")
except Exception as e:
    raise RuntimeError(f"Model failed to load: {e}")

# -----------------------------
# Class Labels
# -----------------------------
class_names = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
    "Sugarcane___Bacterial_Blight",
    "Sugarcane___Healthy",
    "Sugarcane___Red_Rot"
]

# -----------------------------
# Schemas
# -----------------------------
class Prediction(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    input_shape: List[int]
    total_classes: int

class ErrorResponse(BaseModel):
    error: str

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image: Image.Image):
    try:
        input_shape = model.input_shape[1:3]
        image = image.resize(input_shape)

        image = np.array(image)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/", response_model=HealthResponse)
def home():
    return {"status": "API is running"}

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}
@app.get("/model-info")
def model_info():
    try:
        return {
            "model_name": "Plant Disease Classifier",
            "input_shape": str(model.input_shape),
            "total_classes": len(class_names)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )

        # Read image
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )

        # Preprocess
        input_data = preprocess_image(image)

        # Predict
        predictions = model.predict(input_data)[0]

        # Top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]

        results = []
        for i in top_indices:
            results.append({
                "class_name": class_names[i],
                "confidence": float(predictions[i])
            })

        return {"predictions": results}

    except HTTPException as http_err:
        raise http_err  # re-raise known errors

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )