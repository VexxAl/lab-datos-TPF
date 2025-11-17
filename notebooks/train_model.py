import pandas as pd
import joblib
import sys
import os
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from dvclive import Live  # <--- IMPORTANTE

# --- 1. CONFIGURACIÓN DE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.append(str(SRC_DIR))

# --- 2. IMPORTAR EL PIPELINE DE PREPROCESAMIENTO ---
try:
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

# --- 4. CONSTANTES DEL MODELO GANADOR ---
BEST_HYPERPARAMS = {
    "alpha": 7.616538614054318
}
MODEL_VERSION = "v1.0.0"
FINAL_MODEL_NAME = "ridge_final"
FINAL_MODEL_PATH = MODELS_DIR / f"{FINAL_MODEL_NAME}_{MODEL_VERSION}.pkl"


# --- 5. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
def main():
    """Función principal para entrenar y guardar el modelo final."""
    
    # --- INICIAR DVCLIVE ---
    # No es un "experimento" (no corre Optuna),
    # pero sí rastrea el artefacto final.
    with Live(save_dvc_exp=True) as live:
        
        print("--- Iniciando el script de entrenamiento final ---")
        
        # --- Cargar Datos ---
        print(f"Cargando datos desde {PROCESSED_DIR}...")
        try:
            train_df = pd.read_csv(PROCESSED_DIR / "dataset_train_v2.csv", index_col=0)
            test_df  = pd.read_csv(PROCESSED_DIR / "dataset_test_v2.csv", index_col=0)
        except FileNotFoundError:
            print(f"Error: No se encontraron los archivos CSV en {PROCESSED_DIR}.")
            return

        # --- Crear Target (Pre-pipeline) ---
        train_df = add_target_next_day(train_df)
        test_df  = add_target_next_day(test_df)

        # --- Combinar Datos (Re-entrenamiento Fase 3.4) ---
        print("Combinando datos de train y test...")
        full_df = pd.concat([train_df, test_df])
        
        X_full = full_df.drop(columns=["target_frio"])
        y_full = full_df["target_frio"]

        # --- Aplicar Pipeline de Preprocesamiento ---
        print("Aplicando pipeline de preprocesamiento (fit/transform)...")
        X_full_ready = full_preprocess_pipe.fit_transform(X_full, y_full)
        
        try:
            feature_names = full_preprocess_pipe.get_feature_names_out()
            X_full_ready = pd.DataFrame(X_full_ready, index=X_full.index, columns=feature_names)
        except Exception as e:
            if not isinstance(X_full_ready, pd.DataFrame):
                 X_full_ready = pd.DataFrame(X_full_ready, index=X_full.index)

        # --- Limpieza Post-pipeline (NaNs de Lags) ---
        X_full_ready, y_full = drop_na_and_align(X_full_ready, y_full)
        print(f"Datos listos para entrenar. Shape final: {X_full_ready.shape}")

        # --- Entrenar Modelo Final ---
        print(f"Entrenando modelo Ridge (Alpha={BEST_HYPERPARAMS['alpha']})...")
        final_model = Ridge(**BEST_HYPERPARAMS)
        final_model.fit(X_full_ready, y_full)

        # --- Guardar Modelo ---
        print(f"Guardando modelo final en: {FINAL_MODEL_PATH}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(final_model, FINAL_MODEL_PATH)
        
        # --- RASTREAR CON DVCLIVE ---
        print("Rastreando artefacto y parámetros con DVC...")
        live.log_params(BEST_HYPERPARAMS)
        live.log_artifact(
            str(FINAL_MODEL_PATH),
            type="model",
            name=FINAL_MODEL_NAME
        )
        live.log_param("model_name", FINAL_MODEL_NAME)
        live.log_param("model_version", MODEL_VERSION)

        print("\n--- ¡Entrenamiento completado con éxito! ---")


# --- 6. EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    main()