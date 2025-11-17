import pandas as pd
import joblib
import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score
)
from dvclive import Live

# --- 1. CONFIGURACIÓN DE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.append(str(SRC_DIR))

# --- 2. IMPORTAR EL PIPELINE DE PREPROCESAMIENTO ---
try:
    # Importamos el objeto NO ENTRENADO
    from preprocessing_pipeline import full_preprocess_pipe
except ImportError:
    print("Error: No se pudo encontrar 'preprocessing_pipeline.py' en 'src/'.")
    sys.exit(1)

# --- 3. FUNCIONES AUXILIARES ---
def add_target_next_day(df, source_col="Frio (Kw)", target_col="target_frio"):
    df = df.copy()
    df[target_col] = df[source_col].shift(-1)
    df = df.dropna(subset=[target_col])
    return df

def drop_na_and_align(X, y):
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y

# --- 4. CONSTANTES DE ARTEFACTOS ---
BEST_HYPERPARAMS = {
    "alpha": 7.616538614054318  # Descubierto en modelado.ipynb
}
MODEL_VERSION = "v1.0.0"

# Artefacto 1: Modelo
FINAL_MODEL_NAME = "ridge_final"
FINAL_MODEL_PATH = MODELS_DIR / f"{FINAL_MODEL_NAME}_{MODEL_VERSION}.pkl"

# Artefacto 2: Pipeline
PIPELINE_NAME = "pipeline"
PIPELINE_PATH = MODELS_DIR / f"{PIPELINE_NAME}_{MODEL_VERSION}.pkl"

# Archivo de Registro
REGISTRY_FILE = MODELS_DIR / "model_registry.json"

# --- 5. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
def main():
    """
    Script completo de entrenamiento y registro:
    1. Valida el modelo (Train/Test) para obtener métricas.
    2. Guarda y registra el PIPELINE entrenado.
    3. Re-entrena el modelo (Train+Test) para producción.
    4. Guarda, rastrea y registra el MODELO final.
    """
    
    with Live(save_dvc_exp=True) as live:
        
        print("--- Iniciando el script de entrenamiento final ---")
        
        # --- Cargar Datos ---
        print(f"Cargando datos desde {PROCESSED_DIR}...")
        train_df = pd.read_csv(PROCESSED_DIR / "dataset_train_v2.csv", index_col=0)
        test_df  = pd.read_csv(PROCESSED_DIR / "dataset_test_v2.csv", index_col=0)

        # --- Crear Target (Pre-pipeline) ---
        train_df = add_target_next_day(train_df)
        test_df  = add_target_next_day(test_df)

        X_train = train_df.drop(columns=["target_frio"])
        y_train = train_df["target_frio"]
        X_test = test_df.drop(columns=["target_frio"])
        y_test = test_df["target_frio"]

        # --- Aplicar Pipeline (Fit en Train, Transform en ambos) ---
        print("Aplicando y entrenando pipeline de preprocesamiento...")
        # Usamos el 'full_preprocess_pipe' importado
        X_train_ready = full_preprocess_pipe.fit_transform(X_train, y_train)
        X_test_ready = full_preprocess_pipe.transform(X_test)
        
        # --- ¡NUEVO PASO: GUARDAR EL PIPELINE ENTRENADO! ---
        print(f"Guardando pipeline entrenado en: {PIPELINE_PATH}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(full_preprocess_pipe, PIPELINE_PATH)
        live.log_artifact(
            str(PIPELINE_PATH),
            type="pipeline",
            name=PIPELINE_NAME
        )
        
        try:
            feature_names = full_preprocess_pipe.get_feature_names_out()
            X_train_ready = pd.DataFrame(X_train_ready, index=X_train.index, columns=feature_names)
            X_test_ready = pd.DataFrame(X_test_ready, index=X_test.index, columns=feature_names)
        except Exception as e:
            if not isinstance(X_train_ready, pd.DataFrame):
                 X_train_ready = pd.DataFrame(X_train_ready, index=X_train.index)
                 X_test_ready = pd.DataFrame(X_test_ready, index=X_test.index)

        # --- Limpieza Post-pipeline (NaNs de Lags) ---
        X_train_ready, y_train = drop_na_and_align(X_train_ready, y_train)
        X_test_ready, y_test = drop_na_and_align(X_test_ready, y_test)

        # --- 6. PASO DE VALIDACIÓN (Para obtener métricas) ---
        print("Validando modelo para obtener métricas...")
        model_for_validation = Ridge(**BEST_HYPERPARAMS)
        model_for_validation.fit(X_train_ready, y_train)
        preds_test = model_for_validation.predict(X_test_ready)
        
        final_model_metrics = {
            "mae": mean_absolute_error(y_test, preds_test),
            "rmse": root_mean_squared_error(y_test, preds_test),
            "r2": r2_score(y_test, preds_test)
        }
        print(f"Métricas de Validación (Test): {final_model_metrics}")
        live.log_metric("test_mae", round(final_model_metrics["mae"], 4))

        # --- 7. PASO DE RE-ENTRENAMIENTO (Modelo final) ---
        print("Re-entrenando modelo con datos completos (Train+Test)...")
        X_full = pd.concat([X_train_ready, X_test_ready])
        y_full = pd.concat([y_train, y_test])
        
        final_model = Ridge(**BEST_HYPERPARAMS)
        final_model.fit(X_full, y_full)

        # --- 8. GUARDAR Y RASTREAR MODELO FINAL ---
        print(f"Guardando modelo final en: {FINAL_MODEL_PATH}")
        joblib.dump(final_model, FINAL_MODEL_PATH)
        
        print("Rastreando artefacto y parámetros con DVC...")
        live.log_params(BEST_HYPERPARAMS)
        live.log_artifact(
            str(FINAL_MODEL_PATH),
            type="model",
            name=FINAL_MODEL_NAME
        )

        # --- 9. ESCRITURA DEL REGISTRO DE MODELOS ---
        print(f"Actualizando el registro de modelos: {REGISTRY_FILE}")
        
        registry = {}
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                try: registry = json.load(f)
                except json.JSONDecodeError: registry = {}

        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        except Exception:
            commit_hash = "NA"

        # --- Entrada 1: El Modelo Final ---
        model_key = f"{FINAL_MODEL_NAME}_{MODEL_VERSION}"
        registry[model_key] = {
            "model_name": FINAL_MODEL_NAME,
            "version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat(),
            "path": str(FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "hiperparametros": BEST_HYPERPARAMS,
            "metricas_validacion": final_model_metrics,
            "git_commit_hash": commit_hash
        }
        
        # --- Entrada 2: El Pipeline Entrenado ---
        pipeline_key = f"{PIPELINE_NAME}_{MODEL_VERSION}"
        registry[pipeline_key] = {
            "model_name": PIPELINE_NAME,
            "version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat(),
            "path": str(PIPELINE_PATH.relative_to(PROJECT_ROOT)),
            "metricas_validacion": None, # El pipeline no tiene métricas
            "git_commit_hash": commit_hash
        }
        
        # Escribir el archivo JSON
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=4)
        
        print(f"Registro actualizado con: {model_key} y {pipeline_key}")
        print("\n--- ¡Entrenamiento completado con éxito! ---")

# --- 10. EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    main()