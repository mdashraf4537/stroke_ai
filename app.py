import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image

import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from rag_client import query_clinical_guidelines, ask_clinical_qa


# ----------------------------
# App + DB setup
# ----------------------------
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# ----------------------------
# IMPORTANT: edit to match your ML model columns
# ----------------------------
NUM_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# IMPORTANT: your folder order is correct (alphabetical)
MRI_CLASS_NAMES = ["Haemorrhagic", "Ischemic", "Normal"]
CT_CLASS_NAMES = ["Hemorrhage", "Ischemia", "Normal"]

MRI_IMG_SIZE = (224, 224)
CT_IMG_SIZE = (224, 224)


# ----------------------------
# Database Models
# ----------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    pred_type = db.Column(db.String(30), nullable=False)  # "numerical" | "image_mri" | "image_ct" | "image"(old)

    # numerical
    input_json = db.Column(db.Text, nullable=True)
    proba = db.Column(db.Float, nullable=True)
    label = db.Column(db.String(50), nullable=True)

    # image
    image_filename = db.Column(db.String(255), nullable=True)
    gradcam_filename = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="predictions")


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ----------------------------
# Utilities
# ----------------------------
def now_year():
    return datetime.now().year


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in app.config["ALLOWED_EXTENSIONS"]


_ml_model_cache = None
_mri_model_cache = None
_ct_model_cache = None


def load_ml_model():
    global _ml_model_cache
    if _ml_model_cache is not None:
        return _ml_model_cache
    path = app.config["MODEL_ML_PATH"]
    if not os.path.exists(path):
        return None
    _ml_model_cache = joblib.load(path)
    return _ml_model_cache


def load_mri_model():
    global _mri_model_cache
    if _mri_model_cache is not None: return _mri_model_cache
    path = app.config["MODEL_MRI_PATH"]
    if not os.path.exists(path): return None
    _mri_model_cache = tf.keras.models.load_model(path, compile=False)
    return _mri_model_cache

def load_ct_model():
    global _ct_model_cache
    if _ct_model_cache is not None: return _ct_model_cache
    path = app.config["MODEL_CT_PATH"]
    if not os.path.exists(path): return None
    _ct_model_cache = tf.keras.models.load_model(path, compile=False)
    return _ct_model_cache


# ----------------------------
# IMAGE PREPROCESS (FIX: avoid double-normalization)
# ----------------------------
def _model_has_rescaling(model):
    """Detect Rescaling(1./255) layer even inside nested models."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Rescaling):
                    return True
    return False


def prepare_image_input(model, img_rgb,img_size):
    """
    If model already has Rescaling layer (common in your training code),
    do NOT divide by 255 again.
    """
    img_resized = cv2.resize(img_rgb, img_size).astype(np.float32)

    if _model_has_rescaling(model):
        x = img_resized          # keep 0..255
    else:
        x = img_resized / 255.0  # scale to 0..1

    return np.expand_dims(x, axis=0)


# -------- Grad-CAM helpers (FIXED) --------
def _iter_layers_reverse(m):
    """Recursively yield layers in reverse order (supports nested models)."""
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_reverse(layer)
        yield layer


def find_last_conv_layer_obj(model):
    """Return the last Conv2D layer object (works for nested transfer learning models)."""
    for layer in _iter_layers_reverse(model):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def _unwrap_output(x):
    """If keras returns list/tuple outputs, unwrap the first tensor."""
    return x[0] if isinstance(x, (list, tuple)) else x


def make_gradcam_overlay(img_rgb, model, img_size,pred_index=None):
    """
    img_rgb: numpy uint8 (H,W,3)
    returns: overlay_rgb (224,224,3), pred_index
    """
    last_conv_layer = find_last_conv_layer_obj(model)
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM (model must contain Conv2D layers).")

    # Use correct input structure
    model_inputs = model.inputs
    model_outputs = model.outputs
    pred_tensor = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs

    grad_model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[last_conv_layer.output, pred_tensor]
    )

    # Prepare input exactly as prediction
    inp = prepare_image_input(model, img_rgb,img_size)
    inp_for_model = [inp] if isinstance(model_inputs, (list, tuple)) else inp

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp_for_model)
        preds = _unwrap_output(preds)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Create overlay on resized base image
    base = cv2.resize(img_rgb, img_size)
    heatmap = cv2.resize(heatmap, img_size)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.60,
        heatmap_color, 0.40, 0
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay, pred_index


# ----------------------------
# Routes
# ----------------------------
@app.context_processor
def inject_globals():
    def static_file(filename: str) -> str:
        file_path = os.path.join(app.static_folder, filename)
        version = int(os.path.getmtime(file_path)) if os.path.exists(file_path) else None
        return url_for("static", filename=filename, v=version)

    return {"now_year": now_year(), "static_file": static_file}


@app.route("/")
def home():
    return render_template("home.html", title="Home")


@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not full_name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))

        user = User(
            full_name=full_name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("auth_register.html", title="Register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("home"))

    return render_template("auth_login.html", title="Login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "success")
    return redirect(url_for("home"))


@app.route("/predict/numerical", methods=["GET", "POST"])
@login_required
def predict_numerical():
    model = load_ml_model()
    if model is None:
        flash("Numerical model not found. Paste 'stroke_ml_model.pkl' into models/ folder.", "warning")

    result = None

    if request.method == "POST":
        if model is None:
            flash("Cannot predict: Numerical model missing.", "error")
            return redirect(url_for("predict_numerical"))

        payload = {}
        for f in NUM_FEATURES:
            payload[f] = request.form.get(f, None)

        # Cast numeric fields
        for f in ["age", "avg_glucose_level", "bmi"]:
            v = payload.get(f, None)
            if v is None or str(v).strip() == "":
                payload[f] = np.nan
            else:
                payload[f] = float(v)

        for f in ["hypertension", "heart_disease"]:
            payload[f] = int(payload.get(f, 0))

        X = pd.DataFrame([payload], columns=NUM_FEATURES)

        try:
            proba = float(model.predict_proba(X)[:, 1][0])
            pred = int(proba >= 0.5)
            label = "Stroke Risk" if pred == 1 else "No Stroke Risk"

            rec = PredictionHistory(
                user_id=current_user.id,
                pred_type="numerical",
                input_json=str(payload),
                proba=proba,
                label=label
            )
            db.session.add(rec)
            db.session.commit()

            result = {
                "proba": proba,
                "label": label,
                "created_at": rec.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            flash("Prediction saved in History.", "success")
        except Exception as e:
            flash(f"Prediction failed: {e}", "error")

    return render_template("predict_numerical.html", title="Numerical Prediction", result=result)


def _run_image_prediction(modality):
    is_mri = (modality == "mri")
    model = load_mri_model() if is_mri else load_ct_model()
    class_names = MRI_CLASS_NAMES if is_mri else CT_CLASS_NAMES
    img_size = MRI_IMG_SIZE if is_mri else CT_IMG_SIZE
    label = "MRI" if is_mri else "CT"

    if model is None:
        flash(f"{label} model not found in models/ folder.", "warning")
    img_result = None
    if request.method == "POST":
        if model is None:
            flash(f"Cannot predict: {label} model missing.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        if "image_file" not in request.files:
            flash("No file uploaded.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        f = request.files["image_file"]
        if f.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        if not allowed_file(f.filename):
            flash("Only PNG/JPG/JPEG/DCM allowed.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        filename = secure_filename(f.filename)
        if not filename or "." not in filename:
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "png"
            filename = f"scan_{uuid.uuid4().hex[:8]}.{ext}"
        unique = f"{modality}_{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
        f.save(upload_path)
        try:
            from image_preprocessor import (
                preprocess_real_world_scan,
                prepare_model_tensor,
                check_scan_quality,
                validate_brain_scan,
            )
            # For CT scans apply CLAHE; for MRI keep native contrast as in training
            preprocessed_rgb, raw_rgb = preprocess_real_world_scan(
                upload_path, target_size=img_size, apply_contrast_enhancement=(not is_mri)
            )

            # Strict Brain CT / MRI Scan Validation
            validation = validate_brain_scan(
                upload_path,
                raw_rgb,
                expected_modality=modality,
                original_filename=f.filename,
            )
            if not validation["is_valid"]:
                try:
                    os.remove(upload_path)
                except OSError:
                    pass
                flash(validation["error"], "error")
                return redirect(url_for(f"predict_image_{modality}"))

            quality_info = check_scan_quality(preprocessed_rgb)
            for warning_msg in quality_info.get("warnings", []):
                flash(f"Quality Warning: {warning_msg}", "warning")

            display_filename = unique
            if unique.lower().endswith(".dcm"):
                display_filename = f"{unique}.png"
                display_path = os.path.join(app.config["UPLOAD_FOLDER"], display_filename)
                Image.fromarray(preprocessed_rgb).save(display_path)

            x = prepare_model_tensor(model, preprocessed_rgb, target_size=img_size)
            preds_raw = model.predict(x, verbose=0)
            preds_raw = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            preds = preds_raw[0]
            pred_idx = int(np.argmax(preds))
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
            overlay, _ = make_gradcam_overlay(preprocessed_rgb, model, img_size, pred_idx)
            gradcam_name = f"gradcam_{modality}_{uuid.uuid4().hex}.png"
            gradcam_path = os.path.join(app.config["OUTPUT_FOLDER"], gradcam_name)
            Image.fromarray(overlay).save(gradcam_path)
            rec = PredictionHistory(
                user_id=current_user.id,
                pred_type=f"image_{modality}",
                label=pred_label,
                proba=float(preds[pred_idx]),
                image_filename=display_filename,
                gradcam_filename=gradcam_name
            )
            db.session.add(rec)
            db.session.commit()
            probas = [(class_names[i] if i < len(class_names) else f"Class {i}", float(p)) for i, p in enumerate(preds.tolist())]

            img_result = {
                "pred_label": pred_label,
                "upload_filename": display_filename,
                "gradcam_filename": gradcam_name,
                "probas": probas,
            }
            flash(f"{label} prediction saved in History.", "success")
        except Exception as e:
            flash(f"{label} prediction failed: {e}", "error")

    return render_template(f"predict_image_{modality}.html", title=f"{label} Prediction", img_result=img_result)
    

@app.route("/predict/image", methods=["GET"])
@login_required
def predict_image():
    return render_template("predict_image_select.html", title="Image Prediction")

@app.route("/predict/image-mri", methods=["GET", "POST"])
@login_required
def predict_image_mri():
    return _run_image_prediction("mri")

@app.route("/predict/image-ct", methods=["GET", "POST"])
@login_required
def predict_image_ct():
    return _run_image_prediction("ct")


@app.route("/history")
@login_required
def history():
    rows = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.created_at.desc()).all()
    return render_template("history.html", title="History", rows=rows)


@app.route("/report")
@login_required
def report():
    rows = PredictionHistory.query.filter_by(user_id=current_user.id).all()

    total = len(rows)
    numerical = sum(1 for r in rows if r.pred_type == "numerical")
    mri = sum(1 for r in rows if r.pred_type == "image_mri")
    ct = sum(1 for r in rows if r.pred_type == "image_ct")
    old = sum(1 for r in rows if r.pred_type == "image")
    image = mri + ct + old

    label_counts = {}
    for r in rows:
        if r.label:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
         # ... label_counts same ...
    stats = {"total": total, "numerical": numerical, "image": image, "mri": mri, "ct": ct}
    chart = {
        "type_data": {"labels": ["Numerical", "MRI", "CT"], "values": [numerical, mri, ct]},
        "label_data": {"labels": list(label_counts.keys()) or ["No Data"], "values": list(label_counts.values()) or [0]},
    }

    return render_template("report.html", title="Report", stats=stats, chart=chart)


@app.route("/api/clinical-qa", methods=["POST"])
@login_required
def api_clinical_qa():
    """Interactive Clinical QA endpoint routing to AHA/ASA Knowledge Base RAG + Google/Web Search RAG."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    patient_context = (data.get("patient_context") or "").strip()
    search_mode = (data.get("search_mode") or "auto").strip()

    if not question:
        return jsonify({"error": "Please enter a clinical question."}), 400

    result = ask_clinical_qa(
        question=question,
        patient_context=patient_context,
        search_mode=search_mode,
    )
    return jsonify(result), 200


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    app.run(debug=True)
