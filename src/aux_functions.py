# Funciones auxiliares para el proyecto

# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Configuración básica de logging para el módulo (Se activará cuando se importe y se llame a las funciones)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------

""" 
Sección 1: ANALISIS EXPLORATORIO DE DATOS
    - Funcion para cargar y procesar datos desde archivos Excel.
    - Funcion para unir múltiples hojas en un solo dataset.
    - Funcion para calcular checksums de los datasets.
"""

import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any



def process_excel_sheet(sheet_name: str, excel_files: List[str], data_raw_dir: Path) -> pd.DataFrame:
    """
    Procesa UNA hoja específica de una lista de archivos Excel.
    Aplica la lógica de "totalizadores" (último valor del día).

    Args:
        sheet_name (str): Nombre de la hoja a procesar (ej. 'Consolidado EE').
        excel_files (List[str]): Lista de nombres de archivos Excel (ej. 'Totalizadores...2020_2021.xlsx').
        data_raw_dir (Path): Directorio donde se encuentran los archivos Excel crudos.

    Returns:
        pd.DataFrame: Un DataFrame indexado por fecha con los valores del último
                      registro de cada día para esa hoja.
    """
    
    all_dfs = []
    logging.info(f"--- Iniciando procesamiento para Hoja: '{sheet_name}' ---")
    
    for file_name in excel_files:
        file_path = data_raw_dir / file_name
        if not file_path.exists():
            logging.warning(f"No se encontró el archivo: {file_path}")
            continue
        
        try:
            logging.info(f"Leyendo: {file_name}...")
            # Usamos pd.read_excel directamente sobre el .xlsx
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            all_dfs.append(df)
        except Exception as e:
            # Captura errores comunes como "hoja no encontrada"
            logging.error(f"Error en {file_name} (Hoja: {sheet_name}): {e}")

    if not all_dfs:
        logging.error(f"No se pudieron cargar datos para la hoja '{sheet_name}'. Se omitirá.")
        return pd.DataFrame() # Retorna DF vacío si no hay datos

    # 1. Concatenar todos los DataFrames horarios (de todos los años para esta hoja)
    df_hourly = pd.concat(all_dfs, ignore_index=True)

    # --- 2. Aplicar Filtrado Diario (Totalizadores) ---
    try:
        # Maneja la concatenación de DIA y HORA
        df_hourly['timestamp'] = pd.to_datetime(
            pd.to_datetime(df_hourly['DIA']).dt.strftime('%Y-%m-%d') + ' ' + df_hourly['HORA']
        )
    except Exception:
        # Fallback si 'HORA' no existe o falla (aunque en nuestro caso debería)
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['DIA'])
    
    df_hourly_sorted = df_hourly.sort_values(by='timestamp')
    
    # 3. Agrupar por fecha (del 'DIA') y tomar el ÚLTIMO registro.
    df_daily = df_hourly_sorted.groupby(df_hourly_sorted['DIA']).last().reset_index()
    
    # 4. Limpieza y seteo de índice
    df_daily = df_daily.rename(columns={'timestamp': 'Fecha'})
    df_daily['Fecha'] = pd.to_datetime(df_daily['Fecha'])
    df_daily = df_daily.set_index('Fecha')

    # 5. Eliminar columnas de metadatos o sin nombre
    columnas_a_eliminar = [col for col in df_daily.columns if str(col).startswith('Unnamed')] + ['Id','DIA', 'HORA']
    df_daily = df_daily.drop(columns=columnas_a_eliminar, errors='ignore')
    
    logging.info(f"Procesamiento para '{sheet_name}' completado. {df_daily.shape[0]} días unificados.")
    
    return df_daily


def create_dataset_from_xlsx(sheet_names: List[str], excel_files: List[str], data_raw_dir: Path, output_csv_path: Path) -> pd.DataFrame:
    """
    Orquesta la creación del dataset v0.1.
    Procesa múltiples hojas (usando process_excel_sheet), las une y
    guarda el resultado en un archivo CSV.

    Args:
        sheet_names (List[str]): Lista de todas las hojas a procesar.
        excel_files (List[str]): Lista de todos los archivos Excel fuente.
        data_raw_dir (Path): Directorio de datos crudos (ej. 'data/raw').
        output_csv_path (Path): Path completo del archivo CSV de salida (ej. 'data/processed/dataset_v01.csv').

    Returns:
        pd.DataFrame: El DataFrame final unificado.
    """
    
    dataframes_diarios: Dict[str, pd.DataFrame] = {}

    # --- 1. Procesar cada hoja y almacenar el resultado ---
    for sheet in sheet_names:
        df_sheet = process_excel_sheet(sheet, excel_files, data_raw_dir)
        if not df_sheet.empty:
            dataframes_diarios[sheet] = df_sheet

    # --- 2. Unificación y Guardado ---
    if not dataframes_diarios or 'Consolidado EE' not in dataframes_diarios:
        logging.error("No se pudo generar el dataset final. 'Consolidado EE' es obligatorio.")
        return pd.DataFrame()

    logging.info(f"Se procesaron {len(dataframes_diarios)} hojas. Uniendo...")

    # Tomar el primer dataframe (Consolidado EE) como base
    df_final = dataframes_diarios.pop('Consolidado EE')

    # Hacer merge iterativo con el resto de dataframes
    for nombre_hoja, df_feature in dataframes_diarios.items():
        logging.info(f"Uniendo con: {nombre_hoja}...")

        # Hacemos merge por el índice (Fecha)
        df_final = pd.merge(
            df_final,
            df_feature,
            left_index=True,
            right_index=True,
            how='left',
            # Añadir sufijos solo si hay colisiones (evita errores de columnas duplicadas)
            suffixes=('', f'_{nombre_hoja}')
        )

    # --- 3. Guardar en CSV ---
    try:
        # Asegurarse de que el directorio de salida exista
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_csv_path)
        logging.info(f"¡ÉXITO! Dataset v01 guardado en: {output_csv_path}")
        logging.info(f"Forma del dataset: {df_final.shape}")
    except Exception as e:
        logging.error(f"No se pudo guardar el archivo CSV en {output_csv_path}: {e}")
        return pd.DataFrame() # Retorna DF vacío si falla el guardado

    return df_final


def generar_checksum(file_path: Path) -> str:
    """
    Genera un checksum SHA256 para un archivo.
    Esto es crucial para el versionado de datos.

    Args:
        file_path (Path): Ruta al archivo a hashear.

    Returns:
        str: El hash SHA256 del contenido del archivo.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        logging.error(f"No se pudo generar checksum. Archivo no encontrado: {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error generando checksum para {file_path}: {e}")
        return ""


def track_data_artifact(output_file_path: Path, description: str, source_files_list: List[Path], parameters: Dict[str, Any], base_dir: Path, checksums_file_path: Path, lineage_file_path: Path) -> None:
    """
    Automatiza el versionado de un artefacto de datos (dataset, etc.).

    Genera el checksum del archivo de salida y actualiza
    los archivos 'checksums.json' y 'data_lineage.json'.
   

    Args:
        output_file_path (Path): Path al archivo generado (ej. dataset_v01.csv).
        description (str): Descripción para el data lineage.
        source_files_list (List[Path]): Lista de Paths a los archivos fuente.
        parameters (Dict[str, Any]): Dict de parámetros (ej. {'sheets_used': [...]}).
        base_dir (Path): Path raíz del proyecto (para crear paths relativos).
        checksums_file_path (Path): Path al archivo checksums.json.
        lineage_file_path (Path): Path al archivo data_lineage.json.
    """
    
    logging.info(f"--- Iniciando tracking de artefacto para: {output_file_path.name} ---")
    
    # 1. Generar Checksum
    checksum = generar_checksum(output_file_path)
    if not checksum:
        logging.error(f"No se pudo generar checksum. Abortando tracking.")
        return
    logging.info(f"Checksum (SHA256): {checksum}")

    # 2. Definir la 'key' (path relativo) para los JSON
    # Usamos .as_posix() para asegurar '/' como separador, estándar en DVC/Git
    try:
        relative_path_key = output_file_path.relative_to(base_dir).as_posix()
    except ValueError:
        logging.error(f"Error: {output_file_path} no está dentro de {base_dir}. Usando path absoluto.")
        relative_path_key = output_file_path.as_posix()

    # --- 3. Helper interno para leer JSON de forma segura ---
    def _read_json_safe(file_path: Path) -> Dict:
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Archivo {file_path.name} corrupto. Se reiniciará.")
                return {}
        return {}

    # --- 4. Actualizar checksums.json ---
    checksums_data = _read_json_safe(checksums_file_path)
    checksums_data[relative_path_key] = checksum
    
    try:
        checksums_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checksums_file_path, 'w') as f:
            json.dump(checksums_data, f, indent=4)
        logging.info(f"Checksum guardado en {checksums_file_path.name}")
    except Exception as e:
        logging.error(f"Error al escribir en {checksums_file_path}: {e}")

    # --- 5. Actualizar data_lineage.json ---
    data_lineage = _read_json_safe(lineage_file_path)
    
    # Convertir paths de origen a strings relativos posix
    source_files_relative = [
        f.relative_to(base_dir).as_posix() for f in source_files_list
    ]
    
    # Crear la nueva entrada de lineage
    lineage_entry = {
        "description": description,
        "checksum": checksum,
        "source_files": source_files_relative,
        "parameters": parameters
    }
    
    data_lineage[relative_path_key] = lineage_entry
    
    try:
        lineage_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lineage_file_path, 'w') as f:
            json.dump(data_lineage, f, indent=4)
        logging.info(f"Data lineage actualizado en {lineage_file_path.name}")
    except Exception as e:
        logging.error(f"Error al escribir en {lineage_file_path}: {e}")

    logging.info("--- Tracking de artefacto completado ---")
