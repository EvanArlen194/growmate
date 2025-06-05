from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model
try:
    model = load_model("saved_model/model_crop_recom.h5", compile=False)
except Exception as e:
    model = None
    load_model_error = str(e)
else:
    load_model_error = None

# Label bahasa Inggris ke Bahasa Indonesia
label_translation = {
    'apple': 'Apel',
    'banana': 'Pisang',
    'blackgram': 'Kacang Hitam',
    'chickpea': 'Kacang Arab',
    'coconut': 'Kelapa',
    'coffee': 'Kopi',
    'cotton': 'Kapas',
    'grapes': 'Anggur',
    'jute': 'Goni',
    'kidneybeans': 'Kacang Merah',
    'lentil': 'Kacang Lentil',
    'maize': 'Jagung',
    'mango': 'Mangga',
    'mothbeans': 'Kacang Ngengat',
    'mungbean': 'Kacang Hijau',
    'muskmelon': 'Blewah',
    'orange': 'Jeruk',
    'papaya': 'Pepaya',
    'pigeonpeas': 'Kacang Gude',
    'pomegranate': 'Delima',
    'rice': 'Padi',
    'watermelon': 'Semangka'
}

@app.post("/predict/recom")
async def predict_recom(
    N: int = Form(...),
    P: int = Form(...),
    K: int = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model gagal dimuat", "detail": load_model_error}
        )

    # Validasi nilai
    if any(v < 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
        return JSONResponse(
            status_code=400,
            content={"error": "Nilai tidak boleh negatif"}
        )

    if not (0 <= ph <= 14):
        return JSONResponse(
            status_code=400,
            content={"error": "pH harus antara 0 hingga 14"}
        )

    if not (0 <= humidity <= 100):
        return JSONResponse(
            status_code=400,
            content={"error": "Kelembaban harus antara 0 hingga 100"}
        )

    try:
        input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_array)[0]
        label_index = int(np.argmax(prediction))
        predicted_crop_en = list(label_translation.keys())[label_index]
        predicted_crop_id = label_translation[predicted_crop_en]
        confidence = float(prediction[label_index]) * 100
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Gagal melakukan prediksi", "detail": str(e)}
        )

    return JSONResponse(
        status_code=200,
        content={
            "data": {
                "recom_prediction": predicted_crop_id,
                "confidence": f"{confidence:.2f}%"
            }
        }
    )