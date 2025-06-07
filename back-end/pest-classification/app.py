# Import library yang dibutuhkan
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image, UnidentifiedImageError
import io

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Mengaktifkan CORS agar API dapat diakses dari frontend (berbeda origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke model dan label
MODEL_PATH = 'saved_model/model_hama.h5'
LABELS_PATH = 'saved_model/class_labels.json'

# Load model klasifikasi
model = load_model(MODEL_PATH)

# Load daftar label kelas dari file JSON
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

# Endpoint untuk melakukan prediksi gambar
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Menerima file gambar, melakukan praproses, 
    lalu mengembalikan hasil klasifikasi.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    try:
        # Membaca isi file gambar dari request
        contents = await file.read()

        # Validasi format gambar
        try:
            img = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="File bukan gambar yang valid")

        # Pra-pemrosesan gambar
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        confidence_percent = round(confidence * 100, 2)

        # Jika confidence terlalu rendah
        if confidence < 0.93:
            raise HTTPException(status_code=400, detail="Gambar yang diunggah tampaknya tidak menunjukkan keberadaan hama tanaman seperti serangga. Sistem tidak dapat melakukan identifikasi hama berdasarkan gambar ini.")

        class_index = int(np.argmax(prediction))
        class_name = class_labels[class_index]

        # Kembalikan hasil prediksi
        return {
            "data": {
            "prediction": class_name,
            "confidence": f"{confidence_percent}%"
            }
        }

    except HTTPException as http_err:
        # Biarkan HTTPException dikembalikan seperti aslinya
        raise http_err

    except Exception as e:
        # Tangani error tak terduga
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
