import pandas as pd
import joblib
import sys
import json
import os
from pathlib import Path

# --- 1. CONFIGURACIÓN DE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Añadir 'src' al path de Python para importar módulos
sys.path.append(str(SRC_DIR))

try:
    from aux_functions import load_all_excel_data
    from preprocessing_pipeline import full_preprocess_pipe
except ImportError as e:
    print(f"Error importando módulos desde 'src/'. Asegúrate que los archivos existan. {e}")
    sys.exit(1)

# --- 2. FUNCIÓN DE CARGA DE MODELO ---
def load_model(model_name: str = "ridge_final_v1.0.0"):
    """
    Carga el modelo y preprocesadores desde el registro.
    """
    print(f"Cargando modelo: {model_name}...")
    registry_path = MODELS_DIR / "model_registry.json"
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró 'model_registry.json' en {MODELS_DIR}")
        sys.exit(1)
        
    if model_name not in registry:
        print(f"Error: El modelo '{model_name}' no está en el registro.")
        print(f"Modelos disponibles: {list(registry.keys())}")
        sys.exit(1)

    # Cargar el modelo usando la ruta (relativa) guardada en el JSON
    model_path = PROJECT_ROOT / registry[model_name]['path']
    
    try:
        model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo .pkl en {model_path}")
        sys.exit(1)

# --- 3. FUNCIÓN DE PREPROCESAMIENTO ---
def load_and_preprocess(filepath: str):
    """
    Carga el archivo Excel, lo agrega por día, crea features
    y aplica el pipeline de preprocesamiento.
    """
    print(f"Preprocesando archivo: {filepath}...")
    
    # --- Parte 1: Carga de Excel ---
    # (Usando la función que ya tenías en 'aux_functions.py')
    try:
        # 'load_all_excel_data' debe tomar un filepath y devolver un df unificado
        raw_df = load_all_excel_data(filepath) 
    except Exception as e:
        print(f"Error al cargar el archivo Excel con 'aux_functions': {e}")
        # Plan B: Implementación simple si la función falla o no existe
        if 'raw_df' not in locals():
            print("Intentando cargar solo la primera hoja del Excel...")
            raw_df = pd.read_excel(filepath)
            # (Aquí faltaría la unificación de hojas, es un riesgo)

    # --- Parte 2: Agregación Diaria (Paso 1 de la consigna) ---
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raw_df['Fecha'] = pd.to_datetime(raw_df['Fecha'])
        raw_df = raw_df.set_index('Fecha')

    print("Agregando totalizadores horarios a valores diarios (última hora)...")
    # Resample a 'Día', y tomar el último (máximo) valor
    daily_df = raw_df.resample('D').max() 
    # (Usar .max() es equivalente a tomar el valor de las 23:59)
    daily_df = daily_df.dropna(how='all') # Eliminar días sin datos

    # --- Parte 3: Feature Engineering (Replicando Fase 2) ---
    print("Creando features de ingeniería (lags, tiempo)...")
    
    # Variables de tiempo [cite: 396-397]
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['month'] = daily_df.index.month
    
    # Variables de Lag (ej. lag de 1, 7 días) [cite: 395]
    # (Esto debe replicar las features con las que se entrenó el modelo)
    # NOTA: Creamos lags de 'Frio (kW)' ANTES de dropearlo.
    target_col = "Frio (kW)" 
    if target_col in daily_df.columns:
        daily_df['lag_1_frio'] = daily_df[target_col].shift(1)
        daily_df['lag_7_frio'] = daily_df[target_col].shift(7)
    
    # (Aquí irían todas las demás features que creaste en 'preprocessing.ipynb')
    
    # --- Parte 4: Aplicar Pipeline (Scaling, etc.) ---
    # 'full_preprocess_pipe' se importó desde 'preprocessing_pipeline.py'
    
    # El pipeline espera las columnas con las que fue entrenado
    # Nos aseguramos de que el df tenga esas columnas (aunque sea con NaN)
    try:
        pipe_features = full_preprocess_pipe.feature_names_in_
        # Creamos un df con NaNs para las columnas que falten
        processed_df = pd.DataFrame(index=daily_df.index, columns=pipe_features)
        processed_df.update(daily_df)
        
        print(f"Aplicando pipeline de preprocesamiento (escalado, etc.)...")
        # .transform() porque el pipe ya está fiteado
        X_processed_np = full_preprocess_pipe.transform(processed_df)
        
        # Convertir de vuelta a DataFrame
        X_processed = pd.DataFrame(X_processed_np, index=processed_df.index, columns=pipe_features)
        
        # --- Parte 5: Limpieza final ---
        # El modelo no puede predecir sobre NaNs (creados por los lags)
        # Guardamos el índice original
        original_dates = X_processed.index
        X_processed = X_processed.dropna()
        final_dates = X_processed.index

        print(f"Datos preprocesados listos. Shape: {X_processed.shape}")
        
        # Devolvemos los datos procesados y las fechas
        return X_processed, final_dates

    except Exception as e:
        print(f"Error al aplicar el pipeline 'full_preprocess_pipe': {e}")
        print("Asegúrate de que 'preprocessing_pipeline.py' lo contenga.")
        sys.exit(1)

# --- 4. FUNCIÓN DE PREDICCIÓN (de la consigna) ---
def predict_consumption(filepath: str):
    """
    Función principal de predicción.
    """
    model = load_model(model_name="ridge_final_v1.0.0") # Carga el modelo final
    
    # 'X_processed' está listo para el .predict()
    # 'dates' es el índice de Datetime
    X_processed, dates = load_and_preprocess(filepath)
    
    if X_processed.empty:
        print("No se pudieron procesar datos, no se generarán predicciones.")
        return pd.DataFrame(columns=['fecha', 'prediccion_frio_kw'])

    print("Generando predicciones...")
    predictions = model.predict(X_processed)
    
    # Formatear la salida según la consigna [cite: 487-496]
    results_df = pd.DataFrame({
        'fecha': dates.strftime('%Y-%m-%d'),
        'prediccion_frio_kw': predictions
    })
    
    return results_df

# --- 5. BLOQUE DE EJECUCIÓN (de la consigna) ---
if __name__ == "__main__":
    
    # Verificar que se pasó un argumento
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar la ruta al archivo Excel.")
        print("Uso: python src/predict.py <ruta_al_excel>")
        sys.exit(1)
        
    input_filepath = sys.argv[1] # [cite: 501]
    
    if not os.path.exists(input_filepath):
        print(f"Error: No se encuentra el archivo en: {input_filepath}")
        sys.exit(1)

    # Ejecutar el pipeline de predicción
    results = predict_consumption(input_filepath)
    
    # Guardar los resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_csv = RESULTS_DIR / "predicciones.csv"
    
    results.to_csv(output_csv, index=False)
    
    print(f"\n¡Éxito!")
    print(f"Predicciones generadas en: {output_csv}") # [cite: 503]