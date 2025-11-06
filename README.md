# Trabajo Práctico Final - Predicción de Consumo Energético

En este repositorio se encuentra el TPF que desarrollamos para la materia Laboratorio de Datos II.

El objetivo de este trabajo es conseguir crear un modelo de ML que sea capaz de predecir el consumo eléctrico total diario (en Kw) del sistema de refrigeración de una planta cervezera en México.

Para lograrlo, seguiremos dos pasos clave de preprocesamiento:

- **Paso 1: Calcular el Consumo Total Diario**
    Es crucial entender que las variables de nuestro dataset son totalizadores horarios. Esto significa que el valor de cada hora representa el consumo acumulado desde el inicio del día hasta ese momento.
    Para obtener el consumo total de cada día, se deben filtrar los datos para quedarse únicamente con el valor registrado en la última hora (ej., 23:00 o 23:59).

- **Paso 2: Construir la Variable a Predecir**
    Una vez que tengamos los datos agregados por día, construiremos la variable objetivo. Dado que el objetivo es predecir el consumo eléctrico del día siguiente, es necesario alinear las características de un día $D$ con el consumo del día $D+1$.

## 🚀 Estado Actual del Proyecto

El proyecto sigue una metodología MLOps por fases. Actualmente:

- [X] Fase 0: Configuración y Versionado (Git, DVC, Entorno)
- [X] Fase 1: EDA y Refactorización de Ingesta (Merge de `feature/eda` completado)
- [ ] Fase 2: Preprocesamiento y Feature Engineering (En progreso)
- [ ] Fase 3: Modelado y Optimización
- [ ] Fase 4: Pipeline de Predicción

## ⚙️ Configuración y Reproducibilidad

Sigue estos pasos para replicar el entorno y obtener los datos.

### 1. Clonar el Repositorio

  ```bash
  git clone https://github.com/VexxAl/lab-datos-TPF
  cd lab-datos-tpf
  ```

### 2. Crear Entorno Virtual

Recomendamos usar `uv` para una gestión de entorno y paquetes ultra-rápida.

  ```bash
  # Crear el entorno virtual
  $ uv venv
  
  # Activar el entorno (Windows CMD)
  $ .venv\Scripts\activate
  
  # Activar el entorno (Linux/macOS/Git Bash)
  $ source .venv/bin/activate
  ```

### 3. Instalar Dependencias

Instala todas las librerías del proyecto (incluyendo `dvc[s3]`) desde el archivo `requirements.txt`.

  ```bash
  uv pip install -r requirements.txt
  ```

### 4. Sincronizar Datos con DVC (¡Importante!)

Este proyecto utiliza **DVC (Data Version Control)** para gestionar los datasets sin subirlos a Git, asegurando la reproducibilidad. Los archivos `.dvc` en el repositorio (como `data.dvc`) son punteros a los datos reales almacenados en nuestro S3 remoto.

Para descargar los datos, ejecuta:

  ```bash
  dvc pull
  ```

Este comando leerá el archivo `.dvc/config`, se conectará al S3 y descargará los archivos de datos correspondientes (ej. `data/processed/dataset_v01.csv`) a tu copia local.

## Criterios de Evaluación

¿Qué tuvimos en cuenta para considerar que el TPF estaba en condiciones adecuadas?

### Condición de Aprobación

- MAE < 4000 en un set de test oculto.
- El proyecto debe ser completamente reproducible:
  - `git clone`
  - `uv pip install -r requirements.txt`
  - `dvc pull`
  - ejecución de los scripts sin errores
- Todas las fases deben estar reflejadas en el historial de Git con sus respectivas ramas y Pull Requests.
- El script `predict.py` debe ejecutarse sin errores.

### Penalizaciones

- No alcanzar MAE < 4000.
- Código no reproducible o entorno mal configurado.
- Falta de documentación (README, comentarios, justificaciones).
- No seguir la estructura de control de versiones y branching solicitada.
- Fallo en la implementación del versionado de datos, tracking de experimentos o registro de modelos.
