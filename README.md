# 🌱 GrowMate

**Smart Agricultural Solution powered by AI & Machine Learning**

GrowMate adalah platform web cerdas yang dirancang untuk membantu petani dan pecinta tanaman dalam mengelola kesehatan tanaman mereka. Dengan memanfaatkan teknologi **Convolutional Neural Network (CNN)** untuk klasifikasi penyakit dan hama tanaman, serta **Predictive Analytics** untuk rekomendasi tanaman berdasarkan analisis tanah.

🔗 **Website Resmi**: [https://growmate-app.web.app](https://growmate-app.web.app/)

---

## ✨ Fitur Utama

- 🔬 **Deteksi Penyakit Tanaman**: Identifikasi penyakit tanaman secara akurat menggunakan teknologi CNN  
- 🐛 **Klasifikasi Hama**: Deteksi dan klasifikasi hama tanaman dengan model deep learning  
- 🌾 **Rekomendasi Tanaman Cerdas**: Saran tanaman terbaik berdasarkan analisis kondisi tanah menggunakan predictive analytics  
- 📱 **Interface User-Friendly**: Antarmuka web yang mudah digunakan dan responsif  

---

## 🚀 Teknologi yang Digunakan

### Frontend
- **Tools**: HTML, CSS, JavaScript, Bootstrap, Node.js  
- **Styling**: Responsive design untuk berbagai perangkat  

### Backend & AI/ML
- **Deep Learning**: Convolutional Neural Network (CNN) untuk klasifikasi gambar  
- **Machine Learning**: Predictive Analytics untuk analisis tanah  
- **API Framework**: FastAPI REST API services  
- **Python Libraries**: TensorFlow, Scikit-learn, NumPy, Pandas  

### Arsitektur
- **Microservices**: Setiap fitur AI berjalan sebagai service terpisah  
- **RESTful API**: Komunikasi antar service menggunakan HTTP API  
- **Scalable Design**: Mudah untuk dikembangkan dan di-deploy  

---

## 🔗 Endpoints API (Public)

- 🔬 **Klasifikasi Penyakit Tanaman**  
  `https://plant-disease-classification-api.evanarlen.my.id/`

- 🐛 **Klasifikasi Hama Tanaman**  
  `https://pest-classification-api.evanarlen.my.id/`

- 🌾 **Rekomendasi Tanaman Berdasarkan Tanah**  
  `https://crop-recommendation-api.evanarlen.my.id/`

---

### Clone Repository

```bash
git clone https://github.com/EvanArlen194/growmate.git
cd growmate
```

## Frontend

### Instalasi Dependencies

```bash
cd front-end
npm install
```

### Running (Development)

```bash
npm run start-dev
```

### Build

```bash
npm run build
```

### Running (Production)

```bash
npm run serve
```

## Backend

### 1. Klasifikasi Penyakit

#### Setup Virtual Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
venv/Scripts/activate
```

#### Instalasi dan Menjalankan

```bash
# Pindah ke direktori klasifikasi penyakit
cd back-end/plant-disease-classification/app

# Install dependencies
pip install -r requirements.txt

# Menjalankan API
python main.py
```

### 2. Klasifikasi Hama

#### Setup dan Menjalankan

```bash
# Pindah ke direktori klasifikasi hama
cd back-end/klasifikasi_hama

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment (pastikan menggunakan Bash)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Menjalankan API
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

### 3. Rekomendasi Tanaman

#### Setup dan Menjalankan

```bash
# Pindah ke direktori rekomendasi tanaman
cd back-end/pest-classification

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Menjalankan API
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

## Struktur Direktori

## 📁 Struktur Direktori

```
growmate/
├── front-end/
│   ├── package.json
│   └── src/
├── back-end/
│   ├── crop-recommendation/              # 🌾 Rekomendasi Tanaman
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── pest-classification/                 # 🐛 Klasifikasi Hama Tanaman
│   │   ├── app.py
│   │   └── requirements.txt
│   └── plant-disease-classification/     # 🔬 Klasifikasi Penyakit Tanaman
│       └── app/
│           ├── main.py
│           └── requirements.txt
└── README.md
```

## 📋 Endpoints API

- **🔬 Klasifikasi Penyakit**: `http://localhost:9000` - Service CNN untuk deteksi penyakit tanaman
- **🐛 Klasifikasi Hama**: `http://localhost:5000` - Service CNN untuk identifikasi hama tanaman  
- **🌾 Rekomendasi Tanaman**: `http://localhost:8080` - Service ML untuk analisis tanah dan rekomendasi tanaman

## ⚠️ Catatan Penting

1. Pastikan Python 3.9+ sudah terinstall di sistem Anda
2. Untuk klasifikasi hama, pastikan menggunakan Bash terminal saat mengaktifkan virtual environment
3. Setiap service backend berjalan pada port yang berbeda untuk menghindari konflik
4. Pastikan semua dependencies terinstall dengan benar sebelum menjalankan aplikasi


## 💬 Support & Community

Jika Anda mengalami masalah atau memiliki pertanyaan:
- 🐛 **Bug Reports**: Buat issue di repository ini
- 💡 **Feature Requests**: Diskusikan di section Issues
- 📧 **Contact**: growmate.help@gmail.com

---

**⭐ Jika proyek ini membantu Anda, jangan lupa berikan star di GitHub!**