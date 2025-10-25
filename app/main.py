from fastapi import FastAPI
from pydantic import BaseModel
from app.utils.extract_phone import extract_phone_number
import os
import joblib
from app.utils.preprocess import preprocess_text
import json
import logging
from app.ai.retrain_manager import append_confident_ad, trigger_retrain_if_needed
from fastapi.middleware.cors import CORSMiddleware
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI(title="Aimak 996 Classifier API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],      
)

MODEL_PATH = "data/models/classifier.pkl"
VECTORIZER_PATH = "data/models/vectorizer.pkl"
METADATA_PATH = "data/models/metadata.json"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

# Load metadata to map predicted label index -> category id -> ru_name
index_to_id: dict[int, int] | None = None
id_to_name: dict[int, str] | None = None
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # JSON keys become strings; convert back to int
        raw_index_to_id = meta.get("index_to_id", {}) or {}
        index_to_id = {int(k): int(v) for k, v in raw_index_to_id.items()}
        categories = meta.get("categories", []) or []
        id_to_name = {int(c.get("id")): str(c.get("ru_name")) for c in categories if "id" in c}
    except Exception:
        index_to_id = None
        id_to_name = None

class AdInput(BaseModel):
    text: str


class AdPrediction(BaseModel):
    category: int | None
    category_name: str | None
    phone: str | None
    confidence: float | None


class PredictRequest(BaseModel):
    description: str

@app.post("/predict", response_model=AdPrediction)
def predict_ad(data: AdInput):
    logger.info(f"Received prediction request with text: {data.text[:100]}...")
    
    # 1️⃣ Extract phone number
    phone = extract_phone_number(data.text)
    logger.info(f"Extracted phone: {phone}")
    
    # 2️⃣ Preprocess text
    processed = preprocess_text(data.text)
    logger.info(f"Preprocessed text: {processed[:100]}...")
    
    # 3️⃣ Predict category
    category_id: int | None = None
    categories_name: Union[str, None] = None
    confidence = None
    
    if model and vectorizer:
        try:
            logger.info("Transforming text with vectorizer...")
            X = vectorizer.transform([processed])
            logger.info("Making prediction...")
            
            # Get probability estimates
            probas = model.predict_proba(X)[0]
            label_idx = model.predict(X)[0]
            confidence = float(probas.max())  # Get the highest probability
            logger.info(f"Raw prediction (label index): {label_idx}, Confidence: {confidence:.4f}")
            
            if index_to_id and id_to_name is not None:
                logger.info(f"Mapping label index {label_idx} to category...")
                cat_id = index_to_id.get(int(label_idx))
                logger.info(f"Mapped to category ID: {cat_id}")
                
                if cat_id is not None:
                    category_id = int(cat_id)
                    category_name = id_to_name.get(int(cat_id))
                    logger.info(f"Mapped to category name: {category_name}")
            else:
                logger.warning("No category mapping available")
                # Fallback: expose label index as category id
                try:
                    category_id = int(label_idx)
                except Exception:
                    category_id = None
                category_name = None
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return {"category": None, "category_name": None, "phone": phone, "confidence": None}
    else:
        logger.error("Model or vectorizer not loaded")
        return {"category": None, "category_name": None, "phone": phone, "confidence": None}
    
    logger.info(f"Returning prediction: category_id={category_id}, category_name={category_name}, phone={phone}, confidence={confidence:.4f}")
    return {
        "category": category_id,
        "category_name": category_name,
        "phone": phone,
        "confidence": confidence
    }


class TrainingSample(BaseModel):
    description: str
    category_id: int
    source: Union[str, None] = None

    

@app.post("/learn")
def learn_from_new_sample(sample: TrainingSample):
    """
    Принимает новое confident-объявление, сохраняет в буфер и
    при достижении порога запускает переобучение модели.
    """
    try:
        # 1️⃣ Добавляем объявление в буфер CSV
        count = append_confident_ad(
            description=sample.description,
            category_id=sample.category_id,
        )

        # 2️⃣ Проверяем порог и при необходимости запускаем переобучение
        trigger_retrain_if_needed()

        # 3️⃣ Возвращаем ответ
        return {
            "status": "ok",
            "added": count,
            "retrain_triggered": count >= 10,
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}







# --- Health-check ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Classifier API is running 🚀"}