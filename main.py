from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("plantAID_final.keras")

# -----------------------------
# Class Labels (from your JSON)
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
# Image Preprocessing Function
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((160, 160))   # ⚠️ must match training size
    image = np.array(image)

    if image.shape[-1] == 4:  # remove alpha channel if present
        image = image[:, :, :3]

    image = image / 255.0  # normalize if used in training
    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Plant Disease Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_data = preprocess_image(image)

        # Prediction
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        predicted_class = class_names[predicted_class_index]

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}