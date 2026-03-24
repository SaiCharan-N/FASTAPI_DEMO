# 🌱 Plant Disease Detection API (FastAPI)

This project is a FastAPI-based Machine Learning API that classifies plant diseases from uploaded images using a trained Keras model.

---

## 🚀 Features

* Upload plant leaf images
* Predict disease class
* Returns prediction with confidence score
* Supports multiple crops (Corn, Potato, Rice, Wheat, Sugarcane)

---

## 🛠️ Tech Stack

* FastAPI
* TensorFlow / Keras
* NumPy
* Pillow (Image Processing)

---

## 📂 Project Structure

```
.
├── main.py
├── class_indices.json
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

1. Install dependencies:

```
pip install fastapi uvicorn tensorflow pillow numpy
```

2. Start the server:

```
uvicorn main:app --reload
```

3. Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📸 API Usage

### POST `/predict`

* Upload an image file
* Returns predicted class and confidence

### Example Response:

```
{
  "class": "Rice___Leaf_Blast",
  "confidence": 0.93
}
```

---

## ⚠️ Notes

* Model file (`.keras`) is not included in the repo
* Ensure correct input image size (160x160 as per model training)

---

## 👨‍💻 Author

Sai Charan
