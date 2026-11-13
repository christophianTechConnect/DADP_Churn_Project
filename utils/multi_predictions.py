"""Upload CSV dan inference batch; tidak menggunakan dataset historis."""
import base64
import binascii
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .model_utils import load_metadata, load_model, predict, validate_input

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "top_models_registry.json"
# Ketiga model memakai fitur, kategori, threshold, dan risk bins yang sama → satu file metadata.
METADATA_FILE = MODELS_DIR / "model_metadata.json"
# Dipakai jika top_models_registry.json tidak ada/rusak: hanya model final asli.
FALLBACK_MODELS = [{"rank": 1, "key": "gradient_boosting", "name": "Gradient Boosting",
                    "file": "churn_model.joblib"}]


def _build_registry():
    """Daftar model untuk dropdown, diurutkan sesuai rank di models/top_models_registry.json.

    Menambah model: taruh file .joblib di models/ lalu tambahkan satu entri di file JSON tersebut.
    """
    try:
        models = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        models = FALLBACK_MODELS
    registry = {}
    for m in sorted(models, key=lambda m: m.get("rank", 99)):
        auc = (m.get("cv_roc_auc") or {}).get("mean")
        registry[m["key"]] = {
            "label": f"#{m['rank']} {m['name']}" + (f" · CV ROC-AUC {auc:.3f}" if auc else ""),
            "name": m["name"],
            "description": m.get("description", ""),
            "holdout_metrics": m.get("holdout_metrics"),
            "model": MODELS_DIR / m["file"],
            "metadata": METADATA_FILE,
        }
    return registry


MODEL_REGISTRY = _build_registry()


@lru_cache(maxsize=8)
def get_model_bundle(model_id):
    if model_id not in MODEL_REGISTRY:
        raise ValueError("Pilih model yang tersedia.")
    config = MODEL_REGISTRY[model_id]
    if not config["model"].is_file() or not config["metadata"].is_file():
        raise ValueError("File model atau metadata belum tersedia di folder models.")
    meta = load_metadata(config["metadata"])
    if config.get("holdout_metrics"):  # metadata dipakai bersama; metrik harus milik model ini
        meta = {**meta, "model_name": config["name"], "holdout_metrics": config["holdout_metrics"]}
    return load_model(config["model"]), meta


def predict_upload(contents, filename, model_id):
    """Mengembalikan hasil dan daftar error untuk satu file CSV upload."""
    if not filename or Path(filename).suffix.lower() != ".csv":
        return None, ["Format file harus CSV (.csv)."]
    try:
        payload = base64.b64decode(contents.split(",", 1)[1], validate=True)
        data = pd.read_csv(io.StringIO(payload.decode("utf-8-sig")))
    except (ValueError, IndexError, binascii.Error, UnicodeError, pd.errors.ParserError):
        return None, ["CSV tidak dapat dibaca. Gunakan CSV UTF-8 dengan pemisah koma dan header kolom."]
    if data.empty:
        return None, ["File tidak memiliki baris pelanggan."]
    model, meta = get_model_bundle(model_id)
    missing = [c for c in meta["feature_order"] if c not in data.columns]
    if missing:
        return None, [f"Kolom wajib tidak ada: {missing}"]
    errors = []
    for col in meta["numeric_features"]:
        converted = pd.to_numeric(data[col], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted)
        if invalid.any():
            errors.append(f"Kolom '{col}' harus berisi angka terhingga dan tidak boleh kosong ({int(invalid.sum())} baris).")
        data[col] = converted
    for col in meta["categorical_features"]:
        if data[col].isna().any():
            errors.append(f"Kolom '{col}' tidak boleh kosong.")
    errors.extend(validate_input(data, meta))
    if errors:
        return None, errors
    # Kolom output dari upload akan ditimpa oleh hasil model yang dipilih.
    return predict(model, data, meta), []
