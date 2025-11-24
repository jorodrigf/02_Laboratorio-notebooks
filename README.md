# UEES - Inteligencia Artificial - Semana 1
## Laboratorio Práctico en Google Colab

**Autor:** José Luis Rodríguez Flores 
**Universidad:** Universidad de Especialidades Espíritu Santo (UEES)  
**Asignatura:** Inteligencia Artificial  
**Fecha:** 23 Noviembre 2025

---

## Descripción del Proyecto

Este repositorio contiene 4 notebooks de Google Colab que cubren también los fundamentos de Machine Learning y Deep Learning, trabajando con 5 datasets obligatorios.

---

## Estructura del Repositorio
```
02_Laboratorio-notebooks/
│
├── README.md
├── requirements.txt
├── 01_Fundamentos_NumPy_Pandas.ipynb
├── 02_Visualizacion_Datos.ipynb
├── 03_Machine_Learning_Basico.ipynb
├── 04_Deep_Learning_Intro.ipynb
│
└── images/
    ├── notebook1/
    │   ├── distribucion_de_supervivientes.png
    │   ├── digits_fundamentos.png
    │   ├── figura_de_sepal.png
    │
    ├── notebook2/
    │   ├── distribucion_de_clases_vinos.png
    │   ├── distribucion_y_correlacion_de_precios.png
    │   ├── distribucion_de_clases_por_edades_y_supervivencia.png
    │   ├── graficos_estadisticos.png
    │   ├── escatter_interactivo.png
    │   └── digitos.png
    │   └── titanic_vs_iris.png
    │
    ├── notebook3/
    │   ├── matriz_de_confusion_digits.png
    │   ├── matriz_de_confusion_logistic_regresion.png
    │   ├── prediccines_linear_regresion.png
    │   ├── top_10_fratures_mas_importantes.png
    │   └── digits_analysis.png
    │
    └── notebook4/
        ├── accuracy_and_loss_digits.png
        ├── accuracy_and_loss_iris.png
        ├── predicciones_vs_real_boston_housings.png
        ├── comparacion_de_modelos_deep_learning.png
        ├── digits.png     
```

---

## Notebooks

### Notebook 1: Fundamentos de NumPy y Pandas
**Archivo:** `01_Fundamentos_NumPy_Pandas.ipynb`

**Contenido:**
- Manipulación de arrays con NumPy
- Operaciones matemáticas y estadísticas
- Análisis de datos con Pandas
- Exploración de 5 datasets: Titanic, Iris, Wine, Boston Housing, Digits

**Datasets usados:**
-  Titanic (supervivencia)
-  Iris (clasificación de flores)
-  Wine (clasificación de vinos)
-  Boston Housing (predicción de precios)
-  Digits (reconocimiento de dígitos)

---

### Notebook 2: Visualización de Datos
**Archivo:** `02_Visualizacion_Datos.ipynb`

**Contenido:**
- Gráficos básicos con Matplotlib
- Visualizaciones estadísticas con Seaborn
- Gráficos interactivos con Plotly
- Análisis visual de los 5 datasets

**Técnicas de visualización:**
- Gráficos de línea, dispersión, barras
- Boxplots, violinplots, heatmaps
- Matrices de correlación
- Gráficos interactivos

---

### Notebook 3: Machine Learning Básico
**Archivo:** `03_Machine_Learning_Basico.ipynb`

**Contenido:**
- Clasificación binaria (Titanic)
- Clasificación multiclase (Iris, Wine)
- Regresión (Boston Housing)
- Clasificación de imágenes (Digits)

**Modelos implementados:**
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Linear Regression / Ridge

**Resultados:**
- Comparación de algoritmos
- Métricas de evaluación
- Matrices de confusión
- Feature importance

---

### Notebook 4: Introducción a Deep Learning
**Archivo:** `04_Deep_Learning_Intro.ipynb`

**Contenido:**
- Redes neuronales con TensorFlow/Keras
- Clasificación binaria y multiclase
- Regresión con redes neuronales
- Visualización del proceso de entrenamiento

**Arquitecturas:**
- Red densa con Dropout (Titanic)
- Red simple 3 capas (Iris)
- Red con regularización (Wine)
- Red para regresión (Boston)
- Red profunda para imágenes (Digits)

**Técnicas:**
- Funciones de activación (ReLU, Sigmoid, Softmax)
- Optimización con Adam
- Dropout para regularización
- Normalización de datos

---

## Datasets Utilizados

| Dataset | Tipo | Características | Objetivo |
|---------|------|----------------|----------|
| **Titanic** | Clasificación Binaria | 6 features | Predecir supervivencia |
| **Iris** | Clasificación Multiclase | 4 features, 3 clases | Clasificar especies |
| **Wine** | Clasificación Multiclase | 13 features, 3 clases | Clasificar tipos de vino |
| **Boston Housing** | Regresión | 13 features | Predecir precios |
| **Digits** | Clasificación Imágenes | 64 píxeles, 10 clases | Reconocer dígitos 0-9 |

---

## Cómo Ejecutar los Notebooks

### Opción 1: Google Colab
1. Abre [Google Colab](https://colab.research.google.com/)
2. Ve a **Archivo → Abrir notebook → GitHub**
3. Pega la URL de este repositorio
4. Selecciona el notebook que deseas ejecutar
5. Ejecuta todas las celdas: **Entorno de ejecución → Ejecutar todas**

## Tecnologías y Librerías

- **Python 3.10+**
- **TensorFlow/Keras** - Deep Learning
- **Scikit-learn** - Machine Learning
- **NumPy** - Computación numérica
- **Pandas** - Análisis de datos
- **Matplotlib** - Visualización
- **Seaborn** - Visualización estadística
- **Plotly** - Gráficos interactivos

---

## Resultados Principales

### Machine Learning (Notebook 3)
- **Titanic:** Mejor accuracy 82% (Random Forest)
- **Iris:** Mejor accuracy 100% (múltiples modelos)
- **Wine:** Accuracy 97% (Random Forest)
- **Boston:** R² 0.85 (Random Forest)
- **Digits:** Accuracy 97% (Random Forest)

### Deep Learning (Notebook 4)
- **Titanic:** Accuracy 78%
- **Iris:** Accuracy 100%
- **Wine:** Accuracy 98%
- **Boston:** R² 0.83
- **Digits:** Accuracy 96%

---

##  Aprendizajes Clave

 Manipulación y análisis de datos con NumPy y Pandas  
 Visualización efectiva de datos  
 Implementación de algoritmos de Machine Learning  
 Construcción de redes neuronales con Keras  
 Evaluación y comparación de modelos  
 Técnicas de regularización y optimización  
 Preprocesamiento y normalización de datos  

---

## Autor

**José Luis Rodríguez Flores**  
Estudiante de Inteligencia de Negocios y Ciencia de Datos
Asignatura Inteligencia Artificial
Email: jorodrig@uees.edu.ec  
GitHub: [jorodrigf](https://github.com/jorodrigf)

---

## Licencia

Este proyecto es parte de un trabajo académico para la asignatura de Inteligencia Artificial de la maestría de Inteligencia de Negocios y Ciencia de Datos de la
la Universidad de Especialidades Espíritu Santo (UEES).
