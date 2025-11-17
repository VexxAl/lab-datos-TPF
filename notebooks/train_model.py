import pandas as pd
import joblib
import sys
import os
from pathlib import Path
from sklearn.linear_model import Ridge

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
    """Crea la columna objetivo 'target_frio' usando el día D+1."""
    df = df.copy()
    df[target_col] = df[source_col].shift(-1)
    # Eliminamos la última fila que se queda sin target
    df = df.dropna(subset=[target_col])
    return df

def drop_na_and_align(X, y):
    """Elimina NaNs (de lag features, etc.) y alinea X con y."""
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


# --- 4. CONSTANTES DEL MEJOR MODELO ---
BEST_HYPERPARAMS = {
    "alpha": 7.616538614054318
}
MODEL_VERSION = "v1.0.0"
FINAL_MODEL_NAME = "ridge_final" # El modelo re-entrenado
FINAL_MODEL_PATH = MODELS_DIR / f"{FINAL_MODEL_NAME}_{MODEL_VERSION}.pkl"


# --- 5. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
def main():
    """Función principal para entrenar y guardar el modelo final."""
    
    print("--- Iniciando el script de entrenamiento final ---")
    
    # --- Cargar Datos ---
    print(f"Cargando datos desde {PROCESSED_DIR}...")
    try:
        train_df = pd.read_csv(PROCESSED_DIR / "dataset_train_v2.csv", index_col=0)
        test_df  = pd.read_csv(PROCESSED_DIR / "dataset_test_v2.csv", index_col=0)
    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos CSV en {PROCESSED_DIR}.")
        print("Asegúrate de haber corrido el pipeline de preprocesamiento.")
        return

    # --- Crear Target (Pre-pipeline) ---
    train_df = add_target_next_day(train_df)
    test_df  = add_target_next_day(test_df)

    # --- Combinar Datos (Re-entrenamiento Fase 3.4) ---
    print("Combinando datos de train y test para el re-entrenamiento...")
    full_df = pd.concat([train_df, test_df])
    
    X_full = full_df.drop(columns=["target_frio"])
    y_full = full_df["target_frio"]

    # --- Aplicar Pipeline de Preprocesamiento ---
    print("Aplicando pipeline de preprocesamiento (fit/transform)...")
    
    # Fit y transform en TODOS los datos
    X_full_ready = full_preprocess_pipe.fit_transform(X_full, y_full)
    
    # El pipeline devuelve un array de numpy, lo convertimos a DataFrame
    # para mantener la lógica de 'drop_na_and_align'
    try:
        feature_names = full_preprocess_pipe.get_feature_names_out()
        X_full_ready = pd.DataFrame(X_full_ready, index=X_full.index, columns=feature_names)
    except Exception as e:
        print(f"Warning: No se pudieron obtener nombres de features del pipeline. {e}")
        # Si falla, asumimos que el pipeline ya devuelve un DataFrame
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
    
    print("\n--- ¡Entrenamiento completado con éxito! ---")
    print(f"Modelo: {FINAL_MODEL_PATH.name}")


# --- 6. EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    main()