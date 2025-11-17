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

sys.path.append(str(SRC_DIR))

# --- 2. FUNCIÓN DE CARGA DE ARTEFACTOS (GENERALIZADA) ---
def load_artifact(artifact_name: str):
    """
    Carga un artefacto (modelo o pipeline) desde el registro.
    Asume que el 'model_registry.json' contiene la clave del artefacto.
    """
    print(f"Cargando artefacto: {artifact_name}...")
    registry_path = MODELS_DIR / "model_registry.json"
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró 'model_registry.json' en {MODELS_DIR}")
        sys.exit(1)
        
    if artifact_name not in registry:
        print(f"Error: El artefacto '{artifact_name}' no está en el registro.")
        sys.exit(1)

    artifact_path = PROJECT_ROOT / registry[artifact_name]['path']
    
    try:
        artifact = joblib.load(artifact_path)
        print(f"Artefacto cargado desde: {artifact_path}")
        return artifact
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo .pkl en {artifact_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error desconocido al cargar {artifact_path}: {e}")
        sys.exit(1)

# --- 3. FUNCIÓN INTERNA PARA PROCESAR UNA SOLA HOJA ---
def process_sheet_to_daily(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Helper. Carga UNA hoja, aplica lógica de totalizador y la devuelve diaria.
    """
    print(f"  Procesando hoja: {sheet_name}...")
    try:
        df_hourly = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
        
        if 'DIA' not in df_hourly.columns or 'HORA' not in df_hourly.columns:
            print(f"  Info: Hoja {sheet_name} omitida (no contiene 'DIA' o 'HORA').")
            return pd.DataFrame()
        
        df_hourly['timestamp'] = pd.to_datetime(
            pd.to_datetime(df_hourly['DIA']).dt.strftime('%Y-%m-%d') + ' ' + df_hourly['HORA'].astype(str),
            errors='coerce' 
        )
        
        # Si 'coerce' generó NaT (Not a Time), los eliminamos
        if df_hourly['timestamp'].isnull().any():
            print(f"  Warning: Se encontraron fechas/horas inválidas en {sheet_name}. Esas filas serán omitidas.")
            df_hourly = df_hourly.dropna(subset=['timestamp'])

        if df_hourly.empty:
             print(f"  Warning: No quedaron datos en {sheet_name} después de limpiar fechas.")
             return pd.DataFrame()

        df_hourly_sorted = df_hourly.sort_values(by='timestamp')
        df_daily = df_hourly_sorted.groupby(df_hourly_sorted['DIA']).last().reset_index()
        df_daily = df_daily.rename(columns={'timestamp': 'Fecha'})
        df_daily['Fecha'] = pd.to_datetime(df_daily['Fecha'])
        df_daily = df_daily.set_index('Fecha')
        
        columnas_a_eliminar = [col for col in df_daily.columns if str(col).startswith('Unnamed')] + ['Id','DIA', 'HORA']
        df_daily = df_daily.drop(columns=columnas_a_eliminar, errors='ignore')
        
        return df_daily

    except Exception as e:
        print(f"  Warning: No se pudo procesar la hoja {sheet_name}. Saltando. Error: {e}")
        return pd.DataFrame()


# --- 4. FUNCIÓN DE PREPROCESAMIENTO (CORREGIDA) ---
def load_and_preprocess(filepath: str, pipeline):
    """
    Carga un *único* archivo Excel de forma eficiente (hoja por hoja)
    y aplica el pipeline ENTRENADO.
    """
    print(f"Iniciando preprocesamiento para: {filepath}")

    HOJAS_REQUERIDAS = [
        'Consolidado EE',        
        'Totalizadores Energia',   
        'Consolidado Produccion',
        'Consolidado Agua',
        'Consolidado GasVapor',
        'Consolidado Aire',
        'Totalizadores Glicol',
        'Totalizadores CO2'
    ]
    
    try:
        print("Inspeccionando hojas del Excel...")
        xls = pd.ExcelFile(filepath)
        sheets_en_archivo = xls.sheet_names

        # --- Validación ---
        if 'Consolidado EE' not in sheets_en_archivo:
            print("Error: El archivo Excel no tiene la hoja 'Consolidado EE'.")
            sys.exit(1)

        hojas_faltantes = [h for h in HOJAS_REQUERIDAS if h not in sheets_en_archivo]
        if hojas_faltantes:
            print(f"Error: Al archivo Excel le faltan hojas que el modelo necesita:")
            for hoja in hojas_faltantes:
                print(f"  - {hoja}")
            print("El script no puede continuar.")
            sys.exit(1)

        # --- Carga y Fusión (Controlada) ---
        df_daily_final = process_sheet_to_daily(xls, 'Consolidado EE')
        
        if df_daily_final.empty:
            print("Error: La hoja 'Consolidado EE' no pudo ser procesada.")
            sys.exit(1)

        for sheet_name in HOJAS_REQUERIDAS:
            if sheet_name == 'Consolidado EE':
                continue 

            df_daily_feature = process_sheet_to_daily(xls, sheet_name)
            
            if df_daily_feature.empty:
                print(f"  Warning: La hoja {sheet_name} no generó datos. Se omitirá.")
                continue 
            
            df_daily_final = pd.merge(
                df_daily_final,
                df_daily_feature,
                left_index=True, 
                right_index=True,
                how='left', # 'left' merge es correcto si 'Consolidado EE' es la base
                suffixes=('', f'_{sheet_name}')
            )
        
        print("Todas las hojas REQUERIDAS han sido procesadas y unificadas a nivel diario.")
        daily_df = df_daily_final 

    except Exception as e:
        print(f"Error durante la carga y fusión de hojas: {e}")
        sys.exit(1)

    
    # --- Parte 2: Aplicar Pipeline de Scikit-Learn ---
    try:
        print("Aplicando pipeline ENTRENADO (imputación, features, scaling)...")
        
        X_processed_np = pipeline.transform(daily_df)
        
        # Reconstruir el DataFrame con las columnas transformadas
        try:
            # El pipeline de sklearn permite acceder a los pasos por nombre
            # Pedimos los nombres al último paso ("scaler")
            processed_feature_names = pipeline.named_steps['scaler'].get_feature_names_out()
        except Exception as e:
            print(f"Warning: No se pudo usar .get_feature_names_out() del scaler. {e}")
            # Fallback (asume que los nombres no cambiaron en el scaler)
            try:
                processed_feature_names = pipeline.named_steps['feature_selector'].get_feature_names_out()
            except:
                 processed_feature_names = [f"feature_{i}" for i in range(X_processed_np.shape[1])]


        X_processed = pd.DataFrame(
            X_processed_np, 
            index=daily_df.index, # <-- Usamos el índice original
            columns=processed_feature_names
        )
        
        # --- Parte 4: Limpieza Final (NaNs de Lags) ---
        # El paso 'TimeSeriesFeatureEngineerIndex' crea NaNs al principio
        # del DataFrame (ej. lag_1 del primer día).
        # Esos NaNs deben ser eliminados *después* de procesar.
        X_processed = X_processed.dropna()
        final_dates = X_processed.index 

        print(f"Datos preprocesados listos. Shape final: {X_processed.shape}")

        return X_processed, final_dates
    except Exception as e:
        print(f"Error fatal al aplicar el pipeline: {e}")
        print("Asegúrate de que el pipeline cargado sea el correcto y esté entrenado.")
        sys.exit(1)

# --- 5. FUNCIÓN DE PREDICCIÓN ---
def predict_consumption(filepath: str):
    """
    Función principal de predicción.
    """
    # --- Nombres de Artefactos  ---

    MODEL_NAME_IN_REGISTRY = "ridge_final_v1.0.0"
    PIPELINE_NAME_IN_REGISTRY = "pipeline_v1.0.0" 

    # 1. Cargar artefactos entrenados
    model = load_artifact(artifact_name=MODEL_NAME_IN_REGISTRY) 
    pipeline = load_artifact(artifact_name=PIPELINE_NAME_IN_REGISTRY) 
    
    # 2. Cargar y preprocesar datos "nuevos" usando el pipeline entrenado
    X_processed, dates = load_and_preprocess(filepath, pipeline)
    
    if X_processed.empty:
        print("No se pudieron procesar datos (posiblemente por falta de historia para lags).")
        return pd.DataFrame(columns=['fecha', 'prediccion_frio_kw'])

    print("Generando predicciones...")
    
    # 3. Predecir
    # El 'X_processed' ya tiene solo las columnas correctas
    # gracias al 'feature_selector' dentro del pipeline
    predictions = model.predict(X_processed)
    
    results_df = pd.DataFrame({
        'fecha': dates.strftime('%Y-%m-%d'),
        'prediccion_frio_kw': predictions
    })
    
    return results_df

# --- 6. BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar la ruta al archivo Excel de entrada.")
        print(f"Uso: python {sys.argv[0]} <ruta_al_excel>")
        sys.exit(1)
        
    input_filepath = sys.argv[1]
    
    if not os.path.exists(input_filepath):
        print(f"Error: No se encuentra el archivo en: {input_filepath}")
        sys.exit(1)

    # Iniciar la ejecución
    results = predict_consumption(input_filepath)
    
    # Guardar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_csv = RESULTS_DIR / "predicciones.csv"
    
    results.to_csv(output_csv, index=False)
    
    print(f"\n¡Éxito!")
    print(f"Predicciones generadas en: {output_csv}")