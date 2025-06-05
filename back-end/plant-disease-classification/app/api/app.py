"""
Plant Disease Classification API

This FastAPI application provides an API for classifying plant diseases using a Keras model.
It includes endpoints for health checks and image prediction with proper error handling.
Enhanced with plant/leaf detection to validate that uploaded images contain plant leaves.
"""

import os
import io
import traceback
import logging
from typing import List, Tuple, Dict, Any, Optional
import mimetypes
import imghdr
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "keras_model"
)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
DEFAULT_INPUT_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

INDONESIAN_TRANSLATIONS = {
    "Apple___Apple_scab": "Apel - Kudis Apel",
    "Apple___Black_rot": "Apel - Busuk Hitam",
    "Apple___Cedar_apple_rust": "Apel - Karat Cedar",
    "Apple___healthy": "Apel - Sehat",
    "Blueberry___healthy": "Blueberry - Sehat",
    "Cherry_(including_sour)___Powdery_mildew": "Ceri - Embun Tepung",
    "Cherry_(including_sour)___healthy": "Ceri - Sehat",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Jagung - Bercak Daun Cercospora",
    "Corn_(maize)___Common_rust_": "Jagung - Karat Biasa",
    "Corn_(maize)___Northern_Leaf_Blight": "Jagung - Hawar Daun Utara",
    "Corn_(maize)___healthy": "Jagung - Sehat",
    "Grape___Black_rot": "Anggur - Busuk Hitam",
    "Grape___Esca_(Black_Measles)": "Anggur - Esca (Campak Hitam)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Anggur - Hawar Daun (Bercak Isariopsis)",
    "Grape___healthy": "Anggur - Sehat",
    "Orange___Haunglongbing_(Citrus_greening)": "Jeruk - Huanglongbing (Penghijauan Sitrus)",
    "Peach___Bacterial_spot": "Persik - Bercak Bakteri",
    "Peach___healthy": "Persik - Sehat",
    "Pepper,_bell___Bacterial_spot": "Paprika - Bercak Bakteri",
    "Pepper,_bell___healthy": "Paprika - Sehat",
    "Potato___Early_blight": "Kentang - Hawar Awal",
    "Potato___Late_blight": "Kentang - Hawar Akhir",
    "Potato___healthy": "Kentang - Sehat",
    "Raspberry___healthy": "Raspberry - Sehat",
    "Soybean___healthy": "Kedelai - Sehat",
    "Squash___Powdery_mildew": "Labu - Embun Tepung",
    "Strawberry___Leaf_scorch": "Stroberi - Gosong Daun",
    "Strawberry___healthy": "Stroberi - Sehat",
    "Tomato___Bacterial_spot": "Tomat - Bercak Bakteri",
    "Tomato___Early_blight": "Tomat - Hawar Awal",
    "Tomato___Late_blight": "Tomat - Hawar Akhir",
    "Tomato___Leaf_Mold": "Tomat - Jamur Daun",
    "Tomato___Septoria_leaf_spot": "Tomat - Bercak Daun Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomat - Tungau Laba-laba",
    "Tomato___Target_Spot": "Tomat - Bercak Target",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomat - Virus Keriting Daun Kuning",
    "Tomato___Tomato_mosaic_virus": "Tomat - Virus Mosaik Tomat",
    "Tomato___healthy": "Tomat - Sehat"
}

app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using a Keras model with plant/leaf detection."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
input_size = DEFAULT_INPUT_SIZE

MAX_IMAGE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {'image/jpeg', 'image/png', 'image/jpg'}

class PlantDetector:
    """Service for detecting if an image contains plant leaves"""
    
    @staticmethod
    def analyze_color_features(img: Image.Image) -> Dict[str, float]:
        """
        Analyze color features to detect plant characteristics
        
        Args:
            img: PIL Image to analyze
            
        Returns:
            Dictionary with color feature scores
        """
        img_array = np.array(img)
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        lower_green1 = np.array([40, 40, 40])
        upper_green1 = np.array([80, 255, 255])
        
        lower_green2 = np.array([25, 40, 40])
        upper_green2 = np.array([40, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask1, mask2)
        
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_percentage = green_pixels / total_pixels
        
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        
        value = hsv[:, :, 2]
        avg_value = np.mean(value) / 255.0
        
        return {
            'green_percentage': float(green_percentage),
            'avg_saturation': float(avg_saturation),
            'avg_value': float(avg_value)
        }
    
    @staticmethod
    def analyze_texture_features(img: Image.Image) -> Dict[str, float]:
        """
        Analyze texture features that are common in plant leaves
        
        Args:
            img: PIL Image to analyze
            
        Returns:
            Dictionary with texture feature scores
        """
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(gradient_magnitude) / 255.0
        
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        texture_variance = np.var(texture_response) / (255.0 ** 2)
        
        return {
            'edge_density': float(edge_density),
            'texture_variance': float(texture_variance)
        }
    
    @staticmethod
    def detect_plant_leaf(img: Image.Image, threshold: float = 0.6) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if the image contains a plant leaf based on multiple features
        
        Args:
            img: PIL Image to analyze
            threshold: Confidence threshold for plant detection
            
        Returns:
            Tuple of (is_plant_leaf, confidence_score, analysis_details)
        """
        try:
            analysis_size = (224, 224)
            img_resized = img.resize(analysis_size)
            
            color_features = PlantDetector.analyze_color_features(img_resized)
            
            texture_features = PlantDetector.analyze_texture_features(img_resized)
            
            green_score = min(color_features['green_percentage'] * 2.5, 1.0)
            
            saturation_score = color_features['avg_saturation']
            
            texture_score = min(texture_features['edge_density'] * 3.0, 1.0)
            
            brightness_score = min(color_features['avg_value'] * 1.5, 1.0)
            
            plant_confidence = (
                green_score * 0.4 +
                saturation_score * 0.25 +
                texture_score * 0.25 +
                brightness_score * 0.1
            )
            
            if color_features['avg_saturation'] < 0.1 and color_features['green_percentage'] < 0.05:
                plant_confidence *= 0.3
            
            if color_features['green_percentage'] < 0.02:
                plant_confidence *= 0.5
            
            is_plant = plant_confidence >= threshold
            
            analysis_details = {
                'color_features': color_features,
                'texture_features': texture_features,
                'scores': {
                    'green_score': green_score,
                    'saturation_score': saturation_score,
                    'texture_score': texture_score,
                    'brightness_score': brightness_score
                },
                'plant_confidence': plant_confidence,
                'threshold_used': threshold
            }
            
            logger.info(f"Plant detection - Confidence: {plant_confidence:.3f}, Is plant: {is_plant}")
            
            return is_plant, plant_confidence, analysis_details
            
        except Exception as e:
            logger.error(f"Error in plant detection: {str(e)}")
            return True, 0.5, {'error': str(e)}

class ModelService:
    """Service for model operations"""
    
    @staticmethod
    def load_model() -> Optional[tf.keras.Model]:
        """
        Load the Keras model from the specified path.
        
        Returns:
            Loaded Keras model or None if loading fails
        """
        global model, input_size
        
        try:
            if model is None:
                logger.info(f"Loading model from: {MODEL_PATH}")
                model = load_model(MODEL_PATH, compile=False)
                logger.info("Model loaded successfully")
                model.summary()
                
                input_size = ModelService.detect_input_size(model)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def detect_input_size(model: tf.keras.Model) -> Tuple[int, int]:
        """
        Detect the expected input size from the model.
        
        Args:
            model: Loaded Keras model
            
        Returns:
            Tuple of (height, width) for the model's expected input
        """
        try:
            input_shape = model.input_shape
            if input_shape and len(input_shape) == 4:
                height, width = input_shape[1], input_shape[2]
                if height is not None and width is not None:
                    logger.info(f"Detected model input size: ({height}, {width})")
                    return (height, width)
        except Exception as e:
            logger.warning(f"Could not detect model input size: {str(e)}")
        
        logger.info(f"Using default input size: {DEFAULT_INPUT_SIZE}")
        return DEFAULT_INPUT_SIZE

class ImageValidator:
    """Image validation service"""
    
    @staticmethod
    def validate_image_size(file_size: int) -> None:
        """
        Validate image file size
        
        Args:
            file_size: Size of the file in bytes
            
        Raises:
            ValueError: If file size exceeds the maximum allowed size
        """
        if file_size > MAX_IMAGE_SIZE:
            max_size_mb = MAX_IMAGE_SIZE / (1024 * 1024)
            raise ValueError(f"Ukuran gambar melebihi batas maksimum {max_size_mb}MB")
    
    @staticmethod
    def validate_mime_type(content_type: str) -> None:
        """
        Validate image MIME type
        
        Args:
            content_type: MIME type of the file
            
        Raises:
            ValueError: If file has an unsupported MIME type
        """
        if not content_type.startswith('image/'):
            raise ValueError("File yang diunggah bukan gambar")
        
        if content_type not in SUPPORTED_FORMATS:
            supported_formats = ', '.join(fmt.replace('image/', '') for fmt in SUPPORTED_FORMATS)
            raise ValueError(f"Format gambar tidak didukung. Format yang didukung: {supported_formats}")
    
    @staticmethod
    def validate_image_file(file_bytes: bytes) -> str:
        """
        Validate image file contents
        
        Args:
            file_bytes: Image file bytes
            
        Returns:
            Image format detected
            
        Raises:
            ValueError: If file is not a valid image
        """
        if not file_bytes:
            raise ValueError("File yang diunggah kosong")
        
        image_format = imghdr.what(None, file_bytes)
        if not image_format:
            raise ValueError("File yang diunggah bukan gambar yang valid")
        
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
                return image_format
        except Exception as e:
            raise ValueError(f"File gambar tidak valid: {str(e)}")
    
    @staticmethod
    def validate_plant_content(img: Image.Image, strict_mode: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that the image contains plant/leaf content
        
        Args:
            img: PIL Image to validate
            strict_mode: Whether to use strict validation threshold
            
        Returns:
            Tuple of (is_valid, analysis_details)
            
        Raises:
            ValueError: If image doesn't contain plant content
        """
        threshold = 0.7 if strict_mode else 0.5
        is_plant, confidence, details = PlantDetector.detect_plant_leaf(img, threshold)
        
        if not is_plant:
            error_msg = (
                f"Gambar yang diunggah tampaknya tidak berisi daun tanaman. "
                f"Silakan unggah gambar yang menunjukkan daun tanaman untuk klasifikasi penyakit. "
            )
            raise ValueError(error_msg)
        
        return is_plant, details

class ImageProcessor:
    """Service for image processing operations"""
    
    @staticmethod
    def read_image(file_bytes: bytes) -> Image.Image:
        """
        Read image file bytes and convert to PIL Image.
        
        Args:
            file_bytes: Bytes of the image file
            
        Returns:
            PIL Image object in RGB format
        
        Raises:
            ValueError: If image cannot be read or processed
        """
        try:
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error(f"Image format not recognized: {str(e)}")
            raise ValueError(f"Image format not recognized: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            raise ValueError(f"Error reading image: {str(e)}")
    
    @staticmethod
    def preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Preprocess image for model prediction using multiple methods.
        
        Args:
            img: PIL Image to preprocess
            target_size: Target size (height, width) for resizing
            
        Returns:
            List of preprocessed image arrays using different methods
        
        Raises:
            ValueError: If image cannot be preprocessed
        """
        try:
            img = img.resize(target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            normalized = img_array / 255.0
            mobilenet_preprocessed = preprocess_input(img_array.copy())
            
            return [normalized, mobilenet_preprocessed]
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")

class PredictionService:
    """Service for prediction operations"""
    
    @staticmethod
    def format_class_name_english(class_name: str) -> str:
        """
        Format class name in English for reference.
        
        Args:
            class_name: Original class name with underscores
            
        Returns:
            Formatted class name in English
        """
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{plant} - {condition}"
        return class_name.replace('_', ' ')
    
    @staticmethod
    def format_class_name(class_name: str) -> str:
        """
        Format class name by translating to Indonesian and improving readability.
        
        Args:
            class_name: Original class name with underscores
            
        Returns:
            Formatted class name in Indonesian
        """
        if class_name in INDONESIAN_TRANSLATIONS:
            return INDONESIAN_TRANSLATIONS[class_name]
        
        parts = class_name.split('___')
        if len(parts) == 2:
            plant, condition = parts
            plant = plant.replace('_', ' ')
            condition = condition.replace('_', ' ')
            return f"{plant} - {condition}"
        return class_name.replace('_', ' ')
    
    @staticmethod
    def predict(img_array: np.ndarray, method_idx: int) -> Dict[str, Any]:
        """
        Make a prediction using the model.
        
        Args:
            img_array: Preprocessed image array
            method_idx: Index of preprocessing method used
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If prediction fails
        """
        loaded_model = ModelService.load_model()
        if loaded_model is None:
            raise ValueError("Model failed to load")
        
        try:
            logger.info(f"Making prediction with method #{method_idx+1}, shape: {img_array.shape}")
            
            preds = loaded_model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            
            raw_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
            formatted_class = PredictionService.format_class_name(raw_class)
            
            confidence = float(np.max(preds[0]))
            confidence_percentage = f"{confidence * 100:.2f}%"
            
            logger.info(f"Prediction successful with method #{method_idx+1}")
            
            return {
                "prediction": formatted_class, 
                "raw_class": raw_class,
                "confidence": confidence_percentage,
                "preprocessing_method": method_idx+1,
                "prediction_english": PredictionService.format_class_name_english(raw_class)
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise ValueError(f"Error making prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    try:
        ModelService.load_model()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "message": "Validation error"}),
    )

@app.get("/")
def root() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with API status message
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        return {"message": "PERINGATAN: Model gagal dimuat. API tidak berfungsi penuh."}
    return {"message": "API Klasifikasi Penyakit Tanaman berjalan dengan deteksi tanaman yang diaktifkan."}

@app.post("/predict-disease")
async def predict(
    file: UploadFile = File(...),
    validate_image: bool = Query(True, description="Whether to perform comprehensive image validation"),
    validate_plant: bool = Query(True, description="Whether to validate that image contains plant leaves"),
    strict_plant_detection: bool = Query(True, description="Whether to use strict plant detection threshold")
) -> JSONResponse:
    """
    Predicts plant diseases from the uploaded image.

    Args:
        file: The uploaded image file.
        validate_image: Whether to perform comprehensive image validation.
        validate_plant: Whether to validate that the image contains plant leaves.
        strict_plant_detection: Whether to use a strict threshold for plant detection.

    Returns:
        JSONResponse with the prediction results.

    Raises:
        HTTPException: For various error conditions.
    """
    loaded_model = ModelService.load_model()
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model gagal dimuat. Silakan periksa log server."
        )
    
    try:
        ImageValidator.validate_mime_type(file.content_type)
        
        file_bytes = await file.read()
        
        ImageValidator.validate_image_size(len(file_bytes))
        
        if validate_image:
            image_format = ImageValidator.validate_image_file(file_bytes)
            logger.info(f"Image validated successfully, format: {image_format}")
        
        img = ImageProcessor.read_image(file_bytes)
        
        plant_analysis = None
        if validate_plant:
            is_plant, plant_analysis = ImageValidator.validate_plant_content(img, strict_plant_detection)
            logger.info(f"Plant validation passed with confidence: {plant_analysis.get('plant_confidence', 'N/A')}")
        
        preprocessed_images = ImageProcessor.preprocess_image(img, input_size)
        
        last_error = None
        for i, img_array in enumerate(preprocessed_images):
            try:
                prediction_result = PredictionService.predict(img_array, i)
                
                if plant_analysis:
                    plant_confidence = plant_analysis.get('plant_confidence', 0)
                    prediction_result['plant_detection'] = {
                        'confidence': f"{plant_confidence * 100:.2f}%",
                        'validated': True
                    }
                
                return JSONResponse(content=prediction_result)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Preprocessing method #{i+1} failed: {last_error}")
                continue
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Semua metode preprocessing gagal. Error terakhir: {last_error}"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Prediksi gagal: {str(e)}"
        )