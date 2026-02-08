"""
Sistema de Soporte a la Decisión - Dashboard Inteligente
Análisis de Mercado Inmobiliario con IA

Universidad EAFIT | Fundamentos de Ciencia de Datos
Autor: Juan Rua
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import os
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar configuración específica para Streamlit
try:
    from streamlit_config import check_dependencies, get_data_path
except ImportError:
    # Fallback si hay problemas con el import
    def check_dependencies():
        return []
    def get_data_path():
        return "data/data_imperfecto_v2.csv"

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Dashboard Inmobiliario | EAFIT",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def remove_duplicates(df):
    """Elimina filas duplicadas"""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    return df, removed

def impute_missing_values(df, method):
    """Imputa valores faltantes en columnas numéricas"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputed_count = 0
    
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if method == "Media":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Mediana":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Cero":
                df[col] = df[col].fillna(0)
            imputed_count += null_count
    
    return df, imputed_count

def detect_outliers(df, column):
    """Detecta outliers usando el método IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def treat_outliers(df, column, method):
    """Trata outliers según el método seleccionado"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == "Eliminar":
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == "Reemplazar con límites":
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    elif method == "Reemplazar con mediana":
        median = df[column].median()
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median
    
    return df

def create_calculated_features(df):
    """Crea nuevas columnas calculadas (Feature Engineering)"""
    new_features = []
    
    # Precio por pie cuadrado
    if 'price' in df.columns and 'sqft_living' in df.columns:
        df['precio_por_sqft'] = df['price'] / df['sqft_living'].replace(0, np.nan)
        new_features.append('precio_por_sqft')
    
    # Ratio de baños por habitación
    if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
        df['ratio_banos_habitaciones'] = df['bathrooms'] / df['bedrooms'].replace(0, np.nan)
        new_features.append('ratio_banos_habitaciones')
    
    # Edad de la propiedad
    if 'yr_built' in df.columns:
        current_year = datetime.now().year
        df['edad_propiedad'] = current_year - df['yr_built']
        new_features.append('edad_propiedad')
    
    # Fue renovada (booleano)
    if 'yr_renovated' in df.columns:
        df['fue_renovada'] = (df['yr_renovated'] > 0).astype(int)
        new_features.append('fue_renovada')
    
    # Extraer mes y año si hay fecha
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['mes'] = df['date'].dt.month
            df['anio'] = df['date'].dt.year
            new_features.extend(['mes', 'anio'])
        except:
            pass
    
    return df, new_features

def generate_ai_insights(df, api_key):
    """Genera insights usando la API de Groq"""
    try:
        # Preparar resumen estadístico
        stats_summary = df.describe().to_string()
        
        # Información adicional
        null_info = df.isnull().sum().to_string()
        
        # Prompt estructurado
        prompt = f"""Eres un analista de datos experto en el mercado inmobiliario. 
        Analiza los siguientes datos estadísticos de propiedades y proporciona insights valiosos.
        
        RESUMEN ESTADÍSTICO:
        {stats_summary}
        
        VALORES FALTANTES:
        {null_info}
        
        Por favor proporciona:
        1. **Tendencias principales** observadas en los datos
        2. **Riesgos potenciales** o problemas detectados
        3. **Oportunidades de negocio** basadas en los patrones
        4. **Recomendaciones estratégicas** para inversores
        
        Responde en español de forma clara y concisa, usando viñetas cuando sea apropiado."""
        
        # Llamada a la API de Groq
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-70b-versatile",
            "messages": [
                {"role": "system", "content": "Eres un analista de datos experto especializado en bienes raíces."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"], None
        else:
            return None, f"Error API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, str(e)

# ============================================================================
# SIDEBAR - NAVEGACIÓN Y CONFIGURACIÓN
# ============================================================================
with st.sidebar:
    st.image("https://www.eafit.edu.co/PublishingImages/logosimbolo-eafit.png", width=200)
    st.title("🏠 Dashboard Inmobiliario")
    st.markdown("---")
    
    # Navegación por módulos
    modulo = st.radio(
        "📌 Navegación",
        ["🔄 ETL - Carga y Limpieza", "📊 EDA - Visualizaciones", "🤖 IA - Insights Inteligentes"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Preguntas de Negocio")
    st.markdown("""
    1. ¿Qué factores correlacionan más con el precio?
    2. ¿Existe estacionalidad en los precios?  
    3. ¿Impacto de waterfront en el precio?
    """)
    
    st.markdown("---")
    st.markdown("##### 📚 EAFIT 2026-1")
    st.markdown("Fundamentos de Ciencia de Datos")

# ============================================================================
# INICIALIZACIÓN DEL ESTADO
# ============================================================================
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'new_features' not in st.session_state:
    st.session_state.new_features = []

# ============================================================================
# MÓDULO 1: ETL - INGESTA Y PROCESAMIENTO
# ============================================================================
if modulo == "🔄 ETL - Carga y Limpieza":
    st.title("🔄 Módulo ETL: Ingesta y Procesamiento de Datos")
    st.markdown("Carga, limpia y transforma tus datos de propiedades inmobiliarias.")
    
    # ----- SECCIÓN DE CARGA -----
    st.header("📁 1. Carga de Datos")
    st.markdown("**Dataset:** Propiedades inmobiliarias de Washington State, USA")
    
    df = None
    error_msg = None
    
    # Cargar automáticamente el dataset del proyecto
    try:
        data_path = get_data_path()
        df = pd.read_csv(data_path)
        st.success("✅ Dataset cargado correctamente: data_imperfecto_v2.csv")
        st.info("📊 Este dataset contiene datos reales de propiedades inmobiliarias con imperfecciones para demostrar el proceso completo de ETL")
    except Exception as e:
        error_msg = f"Error al cargar el dataset del proyecto: {str(e)}"
        st.error(f"❌ No se pudo cargar el archivo de datos. {error_msg}")
        st.stop()
    
    if df is not None:
        st.session_state.df_original = df.copy()
        
        # Mostrar información básica
        st.success(f"✅ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        
        with st.expander("👁️ Vista previa de datos crudos"):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("📋 Información del Dataset"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Filas", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columnas", df.shape[1])
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            
            st.markdown("**Tipos de datos:**")
            st.dataframe(df.dtypes.astype(str).reset_index().rename(
                columns={'index': 'Columna', 0: 'Tipo'}
            ), use_container_width=True)
        
        # ----- SECCIÓN DE LIMPIEZA -----
        st.markdown("---")
        st.header("🧹 2. Limpieza de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Checkbox para duplicados
            remove_dups = st.checkbox("Eliminar filas duplicadas", value=True)
            
            # Método de imputación
            imputation_method = st.selectbox(
                "Método de imputación para valores nulos:",
                ["Media", "Mediana", "Cero", "No imputar"]
            )
        
        with col2:
            # Tratamiento de outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outlier_col = st.selectbox(
                "Columna para detectar outliers:",
                ["Ninguna"] + numeric_cols
            )
            
            if outlier_col != "Ninguna":
                outlier_method = st.selectbox(
                    "Método de tratamiento de outliers:",
                    ["No tratar", "Eliminar", "Reemplazar con límites", "Reemplazar con mediana"]
                )
        
        # Botón de aplicar limpieza
        if st.button("🚀 Aplicar Limpieza", type="primary"):
            df_clean = df.copy()
            log_messages = []
            
            # Eliminar duplicados
            if remove_dups:
                df_clean, removed = remove_duplicates(df_clean)
                log_messages.append(f"✓ Duplicados eliminados: {removed}")
            
            # Imputar valores
            if imputation_method != "No imputar":
                df_clean, imputed = impute_missing_values(df_clean, imputation_method)
                log_messages.append(f"✓ Valores imputados ({imputation_method}): {imputed}")
            
            # Tratar outliers
            if outlier_col != "Ninguna" and outlier_method != "No tratar":
                outliers_before, _, _ = detect_outliers(df_clean, outlier_col)
                df_clean = treat_outliers(df_clean, outlier_col, outlier_method)
                outliers_after, _, _ = detect_outliers(df_clean, outlier_col)
                log_messages.append(f"✓ Outliers en {outlier_col}: {len(outliers_before)} → {len(outliers_after)}")
            
            st.session_state.df_clean = df_clean
            
            # Mostrar log
            st.success("✅ Limpieza completada")
            for msg in log_messages:
                st.info(msg)
        
        # ----- SECCIÓN DE FEATURE ENGINEERING -----
        st.markdown("---")
        st.header("⚙️ 3. Feature Engineering")
        
        if st.button("🔧 Crear Variables Calculadas", type="secondary"):
            df_to_process = st.session_state.df_clean if st.session_state.df_clean is not None else df
            df_engineered, new_features = create_calculated_features(df_to_process.copy())
            
            st.session_state.df_clean = df_engineered
            st.session_state.new_features = new_features
            
            st.success(f"✅ {len(new_features)} nuevas variables creadas:")
            for feat in new_features:
                st.write(f"  • `{feat}`")
        
        # Mostrar dataset final
        if st.session_state.df_clean is not None:
            st.markdown("---")
            st.header("📊 4. Dataset Procesado")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filas finales", f"{st.session_state.df_clean.shape[0]:,}")
            with col2:
                st.metric("Columnas finales", st.session_state.df_clean.shape[1])
            with col3:
                st.metric("Nulos restantes", st.session_state.df_clean.isnull().sum().sum())
            
            with st.expander("👁️ Vista del dataset procesado"):
                st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)

# ============================================================================
# MÓDULO 2: EDA - VISUALIZACIÓN DINÁMICA
# ============================================================================
elif modulo == "📊 EDA - Visualizaciones":
    st.title("📊 Módulo EDA: Análisis Exploratorio de Datos")
    
    # Verificar que hay datos
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_original
    
    if df is None:
        st.warning("⚠️ Por favor, primero carga datos en el módulo ETL")
        st.stop()
    
    # ----- FILTROS GLOBALES EN SIDEBAR -----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Filtros Globales")
    
    # Filtro de fecha
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            min_date = df['date'].min()
            max_date = df['date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.sidebar.date_input(
                    "Rango de fechas:",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                if len(date_range) == 2:
                    df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
        except:
            pass
    
    # Filtro de categoría (ciudad)
    if 'city' in df.columns:
        cities = ["Todas"] + sorted(df['city'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("Ciudad:", cities)
        if selected_city != "Todas":
            df = df[df['city'] == selected_city]
    
    # Filtro numérico (slider de precio)
    if 'price' in df.columns:
        price_min = float(df['price'].min()) if pd.notna(df['price'].min()) else 0
        price_max = float(df['price'].max()) if pd.notna(df['price'].max()) else 1000000
        price_range = st.sidebar.slider(
            "Rango de Precio ($):",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            format="$%,.0f"
        )
        df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
    
    st.info(f"📊 Mostrando {len(df):,} registros después de aplicar filtros")
    
    # ----- PESTAÑAS DE ANÁLISIS -----
    tab1, tab2, tab3 = st.tabs(["📈 Análisis Univariado", "🔗 Análisis Bivariado", "📑 Reporte"])
    
    # ----- TAB 1: ANÁLISIS UNIVARIADO -----
    with tab1:
        st.subheader("Análisis Univariado")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Selecciona una variable numérica:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma con Plotly
                fig_hist = px.histogram(
                    df, x=selected_col, nbins=30,
                    title=f"Distribución de {selected_col}",
                    color_discrete_sequence=["#F7B500"]
                )
                fig_hist.update_layout(
                    xaxis_title=selected_col,
                    yaxis_title="Frecuencia",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Boxplot con Plotly
                fig_box = px.box(
                    df, y=selected_col,
                    title=f"Boxplot de {selected_col}",
                    color_discrete_sequence=["#001E62"]
                )
                fig_box.update_layout(template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Estadísticas descriptivas
            with st.expander("📊 Estadísticas Descriptivas"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Media", f"{df[selected_col].mean():,.2f}")
                with col2:
                    st.metric("Mediana", f"{df[selected_col].median():,.2f}")
                with col3:
                    st.metric("Desv. Estándar", f"{df[selected_col].std():,.2f}")
                with col4:
                    st.metric("Nulos", df[selected_col].isnull().sum())
        else:
            st.warning("No hay columnas numéricas para analizar")
    
    # ----- TAB 2: ANÁLISIS BIVARIADO -----
    with tab2:
        st.subheader("Análisis Bivariado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap de correlaciones
            st.markdown("#### 🔥 Matriz de Correlaciones")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdYlBu_r",
                    title="Correlación entre Variables"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Se necesitan al menos 2 columnas numéricas")
        
        with col2:
            # Scatter plot
            st.markdown("#### 📍 Relación entre Variables")
            if len(numeric_cols) >= 2:
                x_var = st.selectbox("Variable X:", numeric_cols, index=0)
                y_var = st.selectbox("Variable Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                fig_scatter = px.scatter(
                    df, x=x_var, y=y_var,
                    color='waterfront' if 'waterfront' in df.columns else None,
                    title=f"{y_var} vs {x_var}",
                    color_discrete_sequence=["#001E62", "#F7B500"]
                )
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Evolución temporal
        st.markdown("---")
        st.markdown("#### 📅 Evolución Temporal")
        
        if 'date' in df.columns and 'price' in df.columns:
            df_time = df.dropna(subset=['date', 'price']).copy()
            df_time['date'] = pd.to_datetime(df_time['date'], errors='coerce')
            df_time = df_time.dropna(subset=['date'])
            
            if len(df_time) > 0:
                df_time = df_time.set_index('date').resample('M')['price'].mean().reset_index()
                
                fig_time = px.area(
                    df_time, x='date', y='price',
                    title="Precio Promedio Mensual",
                    color_discrete_sequence=["#F7B500"]
                )
                fig_time.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precio Promedio ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para el gráfico temporal")
        else:
            st.info("Se requiere columna 'date' y 'price' para el análisis temporal")
    
    # ----- TAB 3: REPORTE -----
    with tab3:
        st.subheader("📑 Reporte Ejecutivo")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Resumen del Dataset")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("### 📈 KPIs Principales")
            if 'price' in df.columns:
                st.metric("Precio Promedio", f"${df['price'].mean():,.0f}")
                st.metric("Precio Mediano", f"${df['price'].median():,.0f}")
            if 'sqft_living' in df.columns:
                st.metric("Área Promedio (sqft)", f"{df['sqft_living'].mean():,.0f}")
            st.metric("Total Propiedades", f"{len(df):,}")
        
        # Gráfico comparativo por ciudad
        if 'city' in df.columns and 'price' in df.columns:
            st.markdown("---")
            st.markdown("### 🏙️ Precio por Ciudad (Top 10)")
            
            city_prices = df.groupby('city')['price'].agg(['mean', 'count']).reset_index()
            city_prices = city_prices.nlargest(10, 'mean')
            
            fig_cities = px.bar(
                city_prices, x='city', y='mean',
                title="Precio Promedio por Ciudad",
                color='count',
                color_continuous_scale="Blues",
                labels={'mean': 'Precio Promedio ($)', 'city': 'Ciudad', 'count': 'Cantidad'}
            )
            fig_cities.update_layout(template="plotly_white")
            st.plotly_chart(fig_cities, use_container_width=True)
        
        # Respuestas a preguntas de negocio
        st.markdown("---")
        st.markdown("### 🎯 Respuestas a Preguntas de Negocio")
        
        with st.expander("1. ¿Qué factores correlacionan más con el precio?"):
            if 'price' in df.columns:
                correlations = df.select_dtypes(include=[np.number]).corr()['price'].drop('price').sort_values(ascending=False)
                st.write("**Top 5 factores correlacionados con el precio:**")
                for i, (col, corr) in enumerate(correlations.head(5).items(), 1):
                    st.write(f"{i}. {col}: {corr:.3f}")
        
        with st.expander("2. ¿Existe estacionalidad en los precios?"):
            if 'mes' in df.columns and 'price' in df.columns:
                monthly_avg = df.groupby('mes')['price'].mean()
                st.line_chart(monthly_avg)
                st.write(f"Mes con mayor precio promedio: {monthly_avg.idxmax()}")
                st.write(f"Mes con menor precio promedio: {monthly_avg.idxmin()}")
            else:
                st.info("Ejecuta Feature Engineering en ETL para ver análisis de estacionalidad")
        
        with st.expander("3. ¿Impacto de waterfront en el precio?"):
            if 'waterfront' in df.columns and 'price' in df.columns:
                wf_analysis = df.groupby('waterfront')['price'].agg(['mean', 'median', 'count'])
                wf_analysis.index = ['Sin vista al agua', 'Con vista al agua']
                st.dataframe(wf_analysis)
                
                if len(wf_analysis) == 2:
                    diff = wf_analysis.loc['Con vista al agua', 'mean'] - wf_analysis.loc['Sin vista al agua', 'mean']
                    pct = (diff / wf_analysis.loc['Sin vista al agua', 'mean']) * 100
                    st.success(f"📈 Las propiedades con vista al agua cuestan en promedio ${diff:,.0f} más ({pct:.1f}%)")

# ============================================================================
# MÓDULO 3: INTEGRACIÓN CON IA (GROQ)
# ============================================================================
elif modulo == "🤖 IA - Insights Inteligentes":
    st.title("🤖 Módulo IA: Insights Generados con Inteligencia Artificial")
    st.markdown("Usa el poder de los modelos de lenguaje para obtener análisis avanzados.")
    
    # Verificar que hay datos
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_original
    
    if df is None:
        st.warning("⚠️ Por favor, primero carga datos en el módulo ETL")
        st.stop()
    
    # Configuración de API
    st.markdown("### 🔐 Configuración de API")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input(
            "Ingresa tu API Key de Groq:",
            type="password",
            help="Obtén tu API key en https://console.groq.com"
        )
    with col2:
        st.markdown("")
        st.markdown("")
        st.link_button("🔑 Obtener API Key", "https://console.groq.com")
    
    # Mostrar resumen de datos actuales
    st.markdown("---")
    st.markdown("### 📊 Datos para Analizar")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Registros", f"{len(df):,}")
        st.metric("Variables", df.shape[1])
    with col2:
        st.metric("Valores Nulos", df.isnull().sum().sum())
        if 'price' in df.columns:
            st.metric("Precio Promedio", f"${df['price'].mean():,.0f}")
    
    with st.expander("📋 Resumen Estadístico (se enviará al LLM)"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Botón para generar insights
    st.markdown("---")
    st.markdown("### 🎯 Generar Análisis con IA")
    
    if st.button("🚀 Generar Insights con IA", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ Por favor, ingresa tu API Key de Groq")
        else:
            with st.spinner("🤔 Analizando datos con IA... (esto puede tomar unos segundos)"):
                insights, error = generate_ai_insights(df, api_key)
                
                if error:
                    st.error(f"❌ Error al generar insights: {error}")
                else:
                    st.success("✅ Análisis completado")
                    st.markdown("---")
                    st.markdown("### 💡 Insights Generados por IA")
                    st.markdown(insights)
                    
                    # Guardar insights en session state
                    st.session_state.last_insights = insights
    
    # Mostrar últimos insights si existen
    if 'last_insights' in st.session_state and st.session_state.last_insights:
        with st.expander("📜 Últimos Insights Generados"):
            st.markdown(st.session_state.last_insights)
    
    # Información adicional
    st.markdown("---")
    st.info("""
    💡 **¿Cómo funciona?**
    1. Se toma el resumen estadístico (`df.describe()`) de tus datos
    2. Se construye un prompt estructurado con contexto de negocio
    3. Se envía a la API de Groq (modelo Llama-3)
    4. El LLM analiza los patrones y devuelve insights accionables
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏠 Dashboard de Análisis Inmobiliario | Universidad EAFIT 2026-1</p>
        <p>Fundamentos de Ciencia de Datos | Prof. Jorge Iván Padilla-Buriticá</p>
    </div>
    """,
    unsafe_allow_html=True
)
