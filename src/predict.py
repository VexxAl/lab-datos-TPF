import pandas as pd
import joblib
import sys
import json
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# --- 1. CONFIGURACIÓN DE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Agregamos SRC al path para que joblib encuentre las clases
sys.path.append(str(SRC_DIR))

# IMPORTANTE: Importamos el módulo donde están definidas las clases del pipeline.
# Sin esto, joblib fallará al intentar reconstruir el objeto.
try:
    import preprocessing_pipeline
except ImportError:
    print("Warning: No se pudo importar 'preprocessing_pipeline'. Asegúrate de ejecutar desde la raíz o que src esté en el path.")

# --- 2. FUNCIÓN DE CARGA DE ARTEFACTOS ---
def load_artifact(artifact_name: str):
    """
    Carga un artefacto (modelo o pipeline) desde el registro.
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

    # Ajustamos el path relativo al sistema actual
    rel_path = registry[artifact_name]['path']
    # Fix para Windows/Linux path separator si el json viene de otro OS
    rel_path = rel_path.replace('\\', '/')
    artifact_path = PROJECT_ROOT / rel_path
    
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
        
        if df_hourly['timestamp'].isnull().any():
            df_hourly = df_hourly.dropna(subset=['timestamp'])

        if df_hourly.empty:
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
        print(f"  Warning: No se pudo procesar la hoja {sheet_name}. Error: {e}")
        return pd.DataFrame()

# --- 4. FUNCIÓN DE PREPROCESAMIENTO ---
def load_and_preprocess(filepath: str, pipeline):
    print(f"Iniciando preprocesamiento para: {filepath}")

    HOJAS_REQUERIDAS = [
        'Consolidado EE', 'Totalizadores Energia', 'Consolidado Produccion',
        'Consolidado Agua', 'Consolidado GasVapor', 'Consolidado Aire',
        'Totalizadores Glicol', 'Totalizadores CO2'
    ]
    
    try:
        xls = pd.ExcelFile(filepath)
        sheets_en_archivo = xls.sheet_names

        if 'Consolidado EE' not in sheets_en_archivo:
            print("Error: El archivo Excel no tiene la hoja 'Consolidado EE'.")
            sys.exit(1)

        df_daily_final = process_sheet_to_daily(xls, 'Consolidado EE')
        
        if df_daily_final.empty:
            print("Error: La hoja 'Consolidado EE' no pudo ser procesada.")
            sys.exit(1)

        for sheet_name in HOJAS_REQUERIDAS:
            if sheet_name == 'Consolidado EE': continue 
            if sheet_name not in sheets_en_archivo: continue

            df_daily_feature = process_sheet_to_daily(xls, sheet_name)
            
            if not df_daily_feature.empty:
                df_daily_final = pd.merge(
                    df_daily_final, df_daily_feature,
                    left_index=True, right_index=True,
                    how='left', suffixes=('', f'_{sheet_name}')
                )
        
        daily_df = df_daily_final 
        print(f"Hojas unificadas. Shape crudo: {daily_df.shape}")

        # --- Extracción de Ground Truth para Validación (si existe) ---
        # Intentamos extraer el valor real futuro para calcular MAE después.
        # El modelo predice t+1 basado en t.
        # Así que el Target Real de la fila t es el valor de "Frio (Kw)" en t+1.
        y_ground_truth = None
        if "Frio (Kw)" in daily_df.columns:
            print("Columna 'Frio (Kw)' encontrada: Se calcularán métricas de validación (MAE).")
            # Shift -1 porque la fila de HOY predice MAÑANA
            y_ground_truth = daily_df["Frio (Kw)"].shift(-1)
        else:
            print("Columna 'Frio (Kw)' NO encontrada: Modo inferencia pura (sin cálculo de error).")

    except Exception as e:
        print(f"Error durante la carga de hojas: {e}")
        sys.exit(1)
    
    # --- Aplicar Pipeline ---
    try:
        print("Aplicando pipeline ENTRENADO...")
        X_processed_np = pipeline.transform(daily_df)
        
        try:
            processed_feature_names = pipeline.named_steps['scaler'].get_feature_names_out()
        except:
            processed_feature_names = [f"feat_{i}" for i in range(X_processed_np.shape[1])]

        X_processed = pd.DataFrame(
            X_processed_np, 
            index=daily_df.index,
            columns=processed_feature_names
        )
        
        # --- Limpieza de NaNs generados por Lags ---
        # Al crear lags, las primeras filas quedan con NaN. Debemos alinearlas con el ground truth.
        
        # Indices válidos tras el pipeline (el pipeline puede no dropear na, pero el modelo fallará con na)
        # Generalmente el Imputer rellena, pero los lags del principio quedan vacíos si no hay historia.
        # Asumimos que el imputer manejó lo que pudo, pero eliminamos remanentes si el modelo no soporta NaNs.
        
        # IMPORTANTE: Antes de eliminar filas en X, debemos alinear y_ground_truth si existe.
        if y_ground_truth is not None:
            # Unimos temporalmente para dropear juntos
            temp_df = X_processed.copy()
            temp_df["__target__"] = y_ground_truth
            
            # Eliminamos filas donde X tenga NaNs (por falta de historia para lags)
            # OJO: No eliminamos si el target es NaN (eso es el día futuro a predecir)
            # Solo eliminamos si faltan FEATURES.
            rows_with_nan_features = X_processed.isna().any(axis=1)
            X_processed = X_processed[~rows_with_nan_features]
            
            # Recuperamos el target alineado
            y_ground_truth = temp_df.loc[X_processed.index, "__target__"]
            
        else:
            X_processed = X_processed.dropna()

        final_dates = X_processed.index 
        print(f"Datos listos para inferencia. Muestras: {len(X_processed)}")

        return X_processed, final_dates, y_ground_truth

    except Exception as e:
        print(f"Error fatal al aplicar pipeline: {e}")
        sys.exit(1)

# --- 5. FUNCIÓN DE PREDICCIÓN ---
def predict_consumption(filepath: str):
    MODEL_NAME = "ridge_final_v1.0.0"
    PIPELINE_NAME = "pipeline_v1.0.0" 

    model = load_artifact(MODEL_NAME) 
    pipeline = load_artifact(PIPELINE_NAME) 
    
    # X_processed: Features listos
    # dates: Fechas de los datos (Día T)
    # y_true: Consumo real del Día T+1 (si existe en el excel)
    X_processed, dates, y_true = load_and_preprocess(filepath, pipeline)
    
    if X_processed.empty:
        print("No hay datos suficientes para generar predicciones.")
        return

    print("Generando predicciones...")
    predictions = model.predict(X_processed)
    
    # Armar DataFrame de resultados
    # La fecha de los datos es T. La predicción es para T+1.
    results_df = pd.DataFrame({
        'Fecha_Datos': dates,
        'Fecha_Prediccion': dates + pd.Timedelta(days=1), # Explicitamos que es para mañana
        'Prediccion_Frio_Kw': predictions
    })

    # --- CÁLCULO DE MAE (Si aplica) ---
    if y_true is not None:
        results_df['Valor_Real_Frio_Kw'] = y_true.values
        results_df['Error_Absoluto'] = (results_df['Prediccion_Frio_Kw'] - results_df['Valor_Real_Frio_Kw']).abs()
        
        # Calculamos MAE solo sobre las filas que tienen Valor Real (excluimos la última predicción del futuro desconocido)
        valid_validation = results_df.dropna(subset=['Valor_Real_Frio_Kw'])
        
        if not valid_validation.empty:
            mae = mean_absolute_error(valid_validation['Valor_Real_Frio_Kw'], valid_validation['Prediccion_Frio_Kw'])
            print("\n" + "="*40)
            print(f"RESULTADO DE VALIDACIÓN")
            print(f"MAE (Mean Absolute Error): {mae:.4f} Kw")
            print(f"Filas validadas: {len(valid_validation)}")
            print("="*40 + "\n")
        else:
            print("\nNota: Se encontraron datos históricos, pero no suficientes pares consecutivos para calcular MAE.")
            
    return results_df

# --- 6. BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Falta la ruta al archivo Excel.")
        print(f"Uso: python src/predict.py datos_nuevos.xlsx")
        sys.exit(1)
        
    input_filepath = sys.argv[1]
    if not os.path.exists(input_filepath):
        print(f"Error: No existe el archivo {input_filepath}")
        sys.exit(1)

    results = predict_consumption(input_filepath)
    
    if results is not None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_csv = RESULTS_DIR / "predicciones_con_mae.csv"
        results.to_csv(output_csv, index=False)
        print(f"Resultados guardados en: {output_csv}")