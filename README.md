# 🏠 Dashboard de Análisis Inmobiliario con IA

**Sistema de Soporte a la Decisión** | Universidad EAFIT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-app.streamlit.app)

---

## 📋 Descripción del Problema

### Contexto de Negocio

El mercado inmobiliario de Washington State (USA) presenta una complejidad significativa debido a la diversidad de factores que influyen en el precio de las propiedades. Los inversionistas y agentes inmobiliarios necesitan herramientas analíticas que les permitan:

- **Identificar** los factores clave que determinan el valor de una propiedad
- **Predecir** tendencias de precios basándose en datos históricos
- **Tomar decisiones** informadas sobre inversiones inmobiliarias

Este dashboard integra **Ciencia de Datos** e **Inteligencia Artificial Generativa** para proporcionar análisis automatizados y recomendaciones estratégicas.

### Preguntas de Negocio

1. **¿Qué factores correlacionan más con el precio de las propiedades?**
   - Análisis de correlación entre área, habitaciones, ubicación y precio

2. **¿Existe estacionalidad en los precios de venta?**
   - Identificación de patrones temporales en el mercado

3. **¿Cuál es el impacto de tener vista al agua (waterfront) en el precio final?**
   - Cuantificación del premium por características especiales

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- API Key de Groq (para funciones de IA)

### Pasos para Clonar y Ejecutar Localmente

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/FinalProject-DSF.git
   cd FinalProject-DSF
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```

5. **Abrir en el navegador**
   - La aplicación se abrirá automáticamente en `http://localhost:8501`

---

## 🌐 Link al Despliegue

**🔗 Aplicación en Producción:** [https://final-tst.streamlit.app/]

---

## 📁 Estructura del Proyecto

```
FinalProject-DSF/
├── .streamlit/
│   └── config.toml          # Configuración del tema (colores EAFIT)
├── data/
│   ├── data_imperfecto_v2.csv  # Dataset de propiedades inmobiliarias
│   └── data.csv             # Dataset limpio (opcional)
├── app.py                   # Código principal de la aplicación
├── requirements.txt         # Dependencias del proyecto
├── README.md               # Este archivo
└── manual_usuario.pdf      # Guía para el usuario final
```

---

## 🔧 Funcionalidades

### Módulo 1: ETL (Ingesta y Procesamiento)
- ✅ Carga de archivos CSV y JSON
- ✅ Carga desde URL
- ✅ Eliminación de duplicados
- ✅ Imputación de valores nulos (Media, Mediana, Cero)
- ✅ Detección y tratamiento de outliers
- ✅ Feature Engineering automático

### Módulo 2: EDA (Visualización Dinámica)
- ✅ Filtros globales (fechas, categorías, slider numérico)
- ✅ Histogramas interactivos (Plotly)
- ✅ Boxplots dinámicos
- ✅ Matriz de correlaciones (Heatmap)
- ✅ Gráficos de evolución temporal
- ✅ Organización por pestañas (Univariado, Bivariado, Reporte)

### Módulo 3: Inteligencia Artificial
- ✅ Integración con API de Groq
- ✅ Modelo Llama-3.3-70b-versatile
- ✅ Generación de insights en lenguaje natural
- ✅ Análisis de tendencias, riesgos y oportunidades

---

## 📊 Dataset

**Fuente:** Datos de propiedades inmobiliarias de Washington State, USA

| Característica | Valor |
|---------------|-------|
| Registros | 4,600+ |
| Columnas | 18 |
| Variables Numéricas | 13 |
| Variables Categóricas | 5 |
| Variables Temporales | 1 (date) |
| Variables Booleanas | 1 (waterfront) |

### Variables Principales

| Variable | Descripción |
|----------|-------------|
| `price` | Precio de venta ($) |
| `bedrooms` | Número de habitaciones |
| `bathrooms` | Número de baños |
| `sqft_living` | Área habitable (pies²) |
| `sqft_lot` | Área del terreno (pies²) |
| `floors` | Número de pisos |
| `waterfront` | Vista al agua (0/1) |
| `view` | Calidad de la vista (0-4) |
| `condition` | Condición de la propiedad (1-5) |
| `yr_built` | Año de construcción |
| `city` | Ciudad |
| `date` | Fecha de venta |

---

## 🔑 Configuración de API (Groq)

Para usar las funciones de IA:

1. Visita [console.groq.com](https://console.groq.com)
2. Crea una cuenta gratuita
3. Genera una API Key
4. Ingresa la key en el módulo de IA del dashboard

---

## 📸 Capturas de Pantalla

### Módulo ETL
*Carga y limpieza interactiva de datos*

### Módulo EDA
*Visualizaciones dinámicas con Plotly*

### Módulo IA
*Insights generados por Llama-3*

---

## 👤 Créditos

**Autor:** Juan Pablo Rua, Pedro Saldarriaga, Juan Pablo Mejia  
**Curso:** Fundamentos de Ciencia de Datos  
**Universidad:** EAFIT  
**Periodo:** 2026-1  
**Docente:** Jorge Iván Padilla-Buriticá

### Fuentes de Datos
- Dataset de propiedades inmobiliarias de Washington State
- Procesado y adaptado para fines académicos

---

## 📄 Licencia

Este proyecto es parte del curso de Fundamentos de Ciencia de Datos de la Universidad EAFIT y está destinado únicamente para fines educativos.

---

*"La tecnología por sí sola no genera valor; es la capacidad de usarla para responder las preguntas correctas lo que define a un Científico de Datos."*
