# Stroke Prediction Flask App (SQLite + ML + CNN + Grad-CAM)

This project provides:
- User auth (Register/Login/Logout) using **Flask-Login**
- SQLite database (stored in `instance/app.db`)
- **Numerical stroke prediction** using a saved **.pkl** scikit-learn pipeline/model
- **Image stroke prediction** using a saved **.keras / .h5** TensorFlow model
- **Grad-CAM heatmap** generation saved under `static/outputs/`
- Prediction **History** + simple **Report** dashboard

## 1) Setup (Windows / Linux / macOS)

```bash
cd stroke_flask_app
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## 2) Add your trained models

Place your files here:

- `models/stroke_ml_model.pkl`
- `models/stroke_cnn_model.keras`  (recommended)
  - optional: `models/stroke_cnn_model.h5` (backup)

If models are missing, the app will still run but will show a warning on prediction pages.

## 3) Adjust numerical input columns (IMPORTANT)

In `app.py`, edit `NUM_FEATURES` to match the exact columns used when you trained your `.pkl` model.

Default columns are based on the common Kaggle stroke tabular dataset:
`gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status`

## 4) Adjust image class names (IMPORTANT)

In `app.py`, edit `IMAGE_CLASS_NAMES` to match your training folder order/names.

Default:
`["Haemorrhagic", "Ischemic", "Normal"]`

## 5) Real-World Image Preprocessing & DICOM Support

The app now integrates `image_preprocessor.py` for real-world clinical and photo scans:
- **DICOM Support (.dcm)**: Ingests raw hospital DICOM files and applies the standard radiologist Brain Window ($\text{Center}=40\,\text{HU}, \text{Width}=80\,\text{HU}$).
- **Head & Brain Localization**: Automatically crops out black scanner margins and external patient annotations.
- **CLAHE Contrast Enhancement**: Enhances tissue density differentiation, making subtle ischemic hypodensities visible.
- **Letterbox Resizing**: Preserves exact anatomical aspect ratios without distortion.

## 6) Model Fine-Tuning Pipeline

To retrain or fine-tune models on your dataset:
```bash
python train_finetune.py "<path_to_images>" "models/stroke_cnn_model.keras" "MobileNetV2"
```
Or open and execute `Fine_Tuning_and_RealWorld_Inference.ipynb` in Google Colab / Kaggle with GPU.

---
Made for your project: **Stroke Disease and Its Stages Using Deep Learning**
