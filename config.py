import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "app.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_ML_PATH = os.path.join(BASE_DIR, "models", "stroke_ml_model.pkl")
    MODEL_CT_PATH = os.path.join(BASE_DIR, "models", "stroke_cnn_model_v2.keras")
    MODEL_MRI_PATH= os.path.join(BASE_DIR, "models", "stroke_cnn_model.h5")

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

    RAG_API_URL = os.environ.get(
        "RAG_API_URL", "https://research-assistant-pa3v.onrender.com/api/v1"
    )
    RAG_USER_EMAIL = os.environ.get(
        "RAG_USER_EMAIL", "mdashraf228144@gmail.com"
    )
