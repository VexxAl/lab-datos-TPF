# Trabajo Práctico Final

En este repositorio se encuentra el TPF que desarrollamos para la materia Laboratorio de Datos II.

El objetivo de este trabajo es conseguir crear un modelo de ML que sea capaz de predecir el consumo eléctrico total diario (en Kw) del sistema de refrigeración de una planta cervezera en México.

Para lograrlo, seguiremos dos pasos clave de preprocesamiento:

- **Paso 1: Calcular el Consumo Total Diario**

    Es crucial entender que las variables de nuestro dataset son totalizadores horarios. Esto significa que el valor de cada hora representa el consumo acumulado desde el inicio del día hasta ese momento.

    Para obtener el consumo total de cada día, se deben filtrar los datos para quedarse únicamente con el valor registrado en la última hora (ej., 23:00 o 23:59). Este valor final representará la suma de todo el consumo de esa jornada y será el que utilicemos como la característica (X) para esa fecha específica.

- **Paso 2: Construir la Variable a Predecir**

    Una vez que tengamos los datos agregados por día (el valor de las 23:59 de cada jornada), construiremos la variable objetivo. Dado que el objetivo es predecir el consumo eléctrico del día siguiente, es necesario alinear las características de un día $D$ con el consumo del día $D+1$.

## Reproducibilidad

### ¿Cómo clonar lab-datos-TPF?

Para poder clonar el repositorio en tu equipo y trabajar con el simplemente tenés que ejecutar este comando en tu consola desde la carpeta donde quieras alojar el proyecto:

```bash
git clone https://github.com/VexxAl/lab-datos-TPF.git
```

### ¿Hay que configurar algo?

Hasta esta versión del proyecto (0.1.0) los pasos que recomendamos para configurar tu entorno y poder trabajar en el TPF son:

1. Instalar `uv` si todavía no está instalado en tu equipo. Para hacerlo podés ejucar en tu terminal:

    *Windows:*

    ```powershell
    winget install --id=astral-sh.uv  -e
    ```

    *macOS or Linux:*

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Para más información visitar la [documentación oficial de `uv`](https://docs.astral.sh/uv/)

2. Crear un entorno virtual con `uv` e inicializarlo:

    ```bash
    uv venv
    ```

    ```bash
    source .venv/Scripts/activate
    ```

3. Instalar las dependencias y requerimientos del proyecto:

    ```bash
    uv add -r requirements.txt
    ```

## Estado Actual

- [ ] Fase 0: Configuración
- [ ] Fase 1: EDA
- [ ] Fase 2: Preprocesamiento
- [ ] Fase 3: Modelado
- [ ] Fase 4: Pipeline de Predicción

## Criterios de Evaluación

¿Qué tuvimos en cuenta para considerar que el TPF estaba en condiciones adecuadas?

### Condición de Aprobación

- MAE < 4000 en un set de test oculto.

- El proyecto debe ser completamente reproducible:
  - ejecución de los scripts sin errores
  - `git clone`
  - `uv pip install -r requirements.txt`

- Todas las fases deben estar reflejadas en el historial de Git con sus respectivas ramas y Pull Requests.

- El script predict.py debe ejecutarse sin errores

### Penalizaciones

- No alcanzar MAE < 4000.
- Código no reproducible o entorno mal configurado.
- Falta de documentación (README, comentarios, justificaciones).
- No seguir la estructura de control de versiones y branching solicitada.
- Fallo en la implementación del versionado de datos, tracking de experimentos o registro de modelos.

## Desarrolladores

Este proyecto fue creado, diseñado y finalizado por:

- **Alderete, Valentín**
  - [Github](https://github.com/VexxAl)

- **Jurado, Juan Manuel**
  - [Github](https://github.com/jjuradok)
