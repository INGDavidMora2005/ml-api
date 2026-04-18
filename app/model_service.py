import io
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from app.feature_extractor import extract_color_histogram

MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def train_model(zip_bytes: bytes, classifier_name: str) -> dict:
    X, y = [], []
    classes_found = set()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for entry in zf.namelist():
            path = Path(entry)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            parts = path.parts
            if len(parts) < 2:
                continue
            class_name = parts[-2]
            classes_found.add(class_name)
            try:
                features = extract_color_histogram(zf.read(entry))
                X.append(features)
                y.append(class_name)
            except Exception:
                continue

    if len(X) == 0:
        raise ValueError("No se encontraron imágenes válidas en el .zip.")
    if len(classes_found) < 2:
        raise ValueError(f"Se necesitan al menos 2 clases. Encontradas: {classes_found}")

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Eliminado multi_class (deprecado en scikit-learn 1.6+)
    # LogisticRegression ya maneja multiclase automáticamente con solver lbfgs
    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    model_path = MODELS_DIR / f"{classifier_name}.pkl"
    joblib.dump(model, model_path)

    return {
        "classifier_name": classifier_name,
        "accuracy": round(float(accuracy), 4),
        "classes": sorted(list(classes_found)),
        "total_images": len(X),
    }


def classify_image(image_bytes: bytes, classifier_name: str) -> dict:
    model_path = MODELS_DIR / f"{classifier_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo '{classifier_name}' no encontrado. Entrénalo primero.")

    model = joblib.load(model_path)
    features = extract_color_histogram(image_bytes).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = float(max(model.predict_proba(features)[0]))

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "classifier_name": classifier_name,
    }


def list_models() -> list:
    return [p.stem for p in MODELS_DIR.glob("*.pkl")]