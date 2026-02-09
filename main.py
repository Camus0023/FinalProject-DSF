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

def generate_ai_insights(df, api_key, custom_prompt=None):
    """Genera insights usando la API de Groq"""
    try:
        # Preparar resumen estadístico
        stats_summary = df.describe().to_string() if not df.empty else "No hay datos estadísticos"
        
        # Información adicional
        null_info = df.isnull().sum().to_string() if not df.empty else "No hay información de nulos"
        
        # Usar prompt personalizado o el predeterminado
        if custom_prompt:
            prompt = f"""{custom_prompt}
            
            DATOS ESTADÍSTICOS:
            {stats_summary}
            
            INFORMACIÓN DE VALORES FALTANTES:
            {null_info}
            
            Responde en español de forma clara y práctica, enfocado en decisiones de negocio.
            """
        else:
            # Prompt estructurado predeterminado
            prompt = f"""Eres un analista inmobiliario experto. 
            Analiza los siguientes datos de propiedades y proporciona insights comerciales valiosos.
            
            RESUMEN ESTADÍSTICO:
            {stats_summary}
            
            VALORES FALTANTES:
            {null_info}
            
            Por favor proporciona un análisis orientado al negocio con:
            1. **Oportunidades de inversión** que detectas en los datos
            2. **Riesgos potenciales** para compradores e inversionistas
            3. **Tendencias del mercado** basadas en los patrones
            4. **Recomendaciones específicas** para diferentes tipos de compradores (primera vivienda, inversión, lujo)
            
            Responde en español de forma clara y práctica, usando viñetas cuando sea apropiado.
            Enfócate en insights que sean útiles para tomar decisiones de compra/venta/inversión."""
        
        # Llamada a la API de Groq
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Eres un analista inmobiliario experto especializado en asesorar decisiones de inversión."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
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
    st.markdown("### 🎯 Preguntas de Negocio Reales")
    st.markdown("""
    1. ¿Dónde invertir para obtener mejor retorno?
    2. ¿Cuál es el precio ideal de mi propiedad?  
    3. ¿Qué zonas están subvaloradas?
    4. ¿Qué tipo de propiedades son más rentables?
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
        
        # ----- SECCIÓN DE LIMPIEZA AUTOMÁTICA -----
        st.markdown("---")
        st.header("🧹 2. Limpieza Automática de Datos")
        
        st.markdown("**¿Qué estamos haciendo?** Limpiando automáticamente todos los datos para que estén listos para el análisis.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Propiedades en total", len(df))
        with col2:
            st.metric("Información disponible", len(df.columns))
        with col3:
            problemas = df.isnull().sum().sum() + len(df[df.duplicated()])
            st.metric("Problemas detectados", problemas)
        
        # Ejecutar limpieza automática
        with st.spinner("Limpiando datos automáticamente..."):
            df_original_etl = df.copy()
            df_clean = df.copy()
            cleaning_steps = []
            
            # 1. Eliminar duplicados
            duplicados_antes = len(df_clean[df_clean.duplicated()])
            df_clean = df_clean.drop_duplicates()
            duplicados_eliminados = duplicados_antes
            if duplicados_eliminados > 0:
                cleaning_steps.append(f"✅ Se eliminaron {duplicados_eliminados} propiedades duplicadas")
            
            # 2. Limpiar valores nulos de forma inteligente
            nulos_antes = df_clean.isnull().sum().sum()
            
            # Para columnas numéricas: rellenar con la mediana
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Para columnas de texto: rellenar con 'No especificado'
            text_cols = df_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna('No especificado')
            
            nulos_despues = df_clean.isnull().sum().sum()
            nulos_corregidos = nulos_antes - nulos_despues
            if nulos_corregidos > 0:
                cleaning_steps.append(f"✅ Se corrigieron {nulos_corregidos} datos faltantes")
            
            # 3. Validar y corregir tipos de datos
            tipos_corregidos = 0
            if 'price' in df_clean.columns:
                df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
                tipos_corregidos += 1
            
            if 'bedrooms' in df_clean.columns:
                df_clean['bedrooms'] = pd.to_numeric(df_clean['bedrooms'], errors='coerce')
                tipos_corregidos += 1
                
            if 'bathrooms' in df_clean.columns:
                df_clean['bathrooms'] = pd.to_numeric(df_clean['bathrooms'], errors='coerce')
                tipos_corregidos += 1
            
            if 'sqft_living' in df_clean.columns:
                df_clean['sqft_living'] = pd.to_numeric(df_clean['sqft_living'], errors='coerce')
                tipos_corregidos += 1
            
            if tipos_corregidos > 0:
                cleaning_steps.append(f"✅ Se corrigieron {tipos_corregidos} tipos de datos")
            
            # 4. Eliminar valores extremos poco realistas
            outliers_removidos = 0
            if 'price' in df_clean.columns:
                antes = len(df_clean)
                # Eliminar propiedades con precios irreales (menos de $1000 o más de $50M)
                df_clean = df_clean[(df_clean['price'] >= 1000) & (df_clean['price'] <= 50000000)]
                outliers_removidos += antes - len(df_clean)
            
            if outliers_removidos > 0:
                cleaning_steps.append(f"✅ Se eliminaron {outliers_removidos} propiedades con precios irreales")
            
            # Guardar dataset limpio
            clean_file_path = "data/dataset_limpio.csv"
            try:
                df_clean.to_csv(clean_file_path, index=False)
            except:
                pass  # No issue if can't save locally
            
            # Mostrar resultados
            st.success("✅ ¡Limpieza automática completada!")
            
            # Comparación de salud del dataset
            st.subheader("📊 Comparación: Antes vs Después")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📑 Dataset Original**")
                st.metric("Total de propiedades", len(df_original_etl))
                st.metric("Datos faltantes", df_original_etl.isnull().sum().sum())
                st.metric("Propiedades duplicadas", len(df_original_etl[df_original_etl.duplicated()]))
                original_quality = ((len(df_original_etl) - df_original_etl.isnull().sum().sum()) / len(df_original_etl) * 100)
                st.metric("Calidad de datos", f"{original_quality:.1f}%")
            
            with col2:
                st.markdown("**✨ Dataset Limpio**")
                st.metric("Total de propiedades", len(df_clean), delta=len(df_clean) - len(df_original_etl))
                st.metric("Datos faltantes", df_clean.isnull().sum().sum(), 
                         delta=df_clean.isnull().sum().sum() - df_original_etl.isnull().sum().sum())
                st.metric("Propiedades duplicadas", len(df_clean[df_clean.duplicated()]),
                         delta=len(df_clean[df_clean.duplicated()]) - len(df_original_etl[df_original_etl.duplicated()]))
                clean_quality = ((len(df_clean) - df_clean.isnull().sum().sum()) / len(df_clean) * 100)
                st.metric("Calidad de datos", f"{clean_quality:.1f}%")
            
            # Resumen de lo que se hizo
            with st.expander("🔍 Ver detalles de lo que se limpió"):
                st.markdown("**Pasos de limpieza realizados:**")
                for step in cleaning_steps:
                    st.write(step)
                if not cleaning_steps:
                    st.write("✨ Los datos ya estaban en perfecto estado!")
            
            # Descargar dataset limpio
            csv_clean = df_clean.to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar datos limpios",
                data=csv_clean,
                file_name="propiedades_limpias.csv",
                mime="text/csv"
            )
            
            # Guardar en session state para uso posterior
            st.session_state['df_clean'] = df_clean
            st.session_state['cleaning_completed'] = True
        
        # ----- SECCIÓN DE RESPUESTAS DE NEGOCIO -----
        if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
            df_clean = st.session_state['df_clean']
            
            st.markdown("---")
            st.header("💼 3. Respuestas Rápidas de Negocio")
            st.markdown("**Respuestas directas y claras para tomar decisiones de inversión inmobiliaria**")
            
            # Calcular métricas clave una sola vez
            if 'price' in df_clean.columns:
                precio_promedio = df_clean['price'].mean()
                precio_mediano = df_clean['price'].median()
                
                # Análisis por rangos de precio para oportunidades
                df_clean['rango_precio'] = pd.cut(df_clean['price'], 
                                                 bins=[0, 300000, 600000, 1000000, float('inf')], 
                                                 labels=['Económico', 'Medio', 'Alto', 'Lujo'])
                
                # Análisis de retorno (basado en precio por sqft)
                if 'sqft_living' in df_clean.columns:
                    df_clean['precio_por_sqft'] = df_clean['price'] / df_clean['sqft_living']
                    precio_sqft_promedio = df_clean['precio_por_sqft'].mean()
                
                # Distribución por ubicación (si existe zipcode)
                mejor_ubicacion = None
                if 'zipcode' in df_clean.columns:
                    precio_por_zona = df_clean.groupby('zipcode')['price'].mean().sort_values()
                    mejor_ubicacion = precio_por_zona.index[0] if len(precio_por_zona) > 0 else None
            
            # Crear pestañas para las 4 respuestas
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏠 ¿Dónde invertir?", 
                "💰 ¿Precio ideal?", 
                "📍 ¿Zonas subvaloradas?", 
                "📊 ¿Más rentables?"
            ])
            
            with tab1:
                st.subheader("🏠 ¿Dónde invertir para obtener mejor retorno?")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**💡 Para inversión inteligente:**")
                    
                    # Propiedades por debajo del precio promedio
                    if 'price' in df_clean.columns:
                        oportunidades = df_clean[df_clean['price'] < precio_promedio]
                        st.metric("Propiedades económicas", f"{len(oportunidades):,}")
                        st.metric("Ahorro promedio", f"${precio_promedio - oportunidades['price'].mean():,.0f}")
                    
                    if mejor_ubicacion:
                        st.info(f"🎯 **Zona recomendada:** Código postal {mejor_ubicacion}")
                    
                with col2:
                    st.markdown("**📋 Estrategia recomendada:**")
                    st.markdown("""
                    • Busca propiedades por **debajo de $500,000**
                    • Prioriza casas con **3+ habitaciones** 
                    • Considera propiedades que necesiten **renovación menor**
                    • Evita los códigos postales más caros
                    """)
            
            with tab2:
                st.subheader("💰 ¿Cuál es el precio ideal de mi propiedad?")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Precios de referencia del mercado:**")
                    if 'price' in df_clean.columns:
                        st.metric("Precio típico", f"${precio_mediano:,.0f}")
                        st.metric("Precio promedio", f"${precio_promedio:,.0f}")
                        
                        if 'sqft_living' in df_clean.columns:
                            st.metric("Por pie cuadrado", f"${precio_sqft_promedio:.0f}/sqft")
                
                with col2:
                    st.markdown("**📊 Para calcular tu precio ideal:**")
                    st.markdown(f"""
                    • **Casa pequeña** (menos de 1,500 sqft): ~$300,000 - $450,000
                    • **Casa mediana** (1,500 - 2,500 sqft): ~$450,000 - $650,000  
                    • **Casa grande** (más de 2,500 sqft): ~$650,000+
                    • **Con vista al agua**: Agregar 20-30% al precio base
                    """)
            
            with tab3:
                st.subheader("📍 ¿Qué zonas están subvaloradas?")
                
                if 'zipcode' in df_clean.columns and 'price' in df_clean.columns:
                    # Top 5 zonas más económicas con buenas propiedades
                    zonas_economicas = df_clean.groupby('zipcode').agg({
                        'price': ['mean', 'count']
                    }).round(0)
                    zonas_economicas.columns = ['precio_promedio', 'cantidad']
                    zonas_economicas = zonas_economicas[zonas_economicas['cantidad'] >= 10].sort_values('precio_promedio')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Top 5 zonas con mejor valor:**")
                        for i, (zona, data) in enumerate(zonas_economicas.head(5).iterrows(), 1):
                            st.write(f"{i}. **Zona {zona}**: ${data['precio_promedio']:,.0f}")
                    
                    with col2:
                        st.markdown("**💡 Oportunidades detectadas:**")
                        st.markdown("""
                        • Zonas con precios **30% por debajo** del promedio
                        • Áreas en **desarrollo** con potencial de crecimiento  
                        • Propiedades cerca de **transporte público**
                        • Vecindarios con **mejoras recientes** en infraestructura
                        """)
                else:
                    st.info("💡 Para análisis de zonas específicas, necesitamos datos de ubicación más detallados.")
            
            with tab4:
                st.subheader("📊 ¿Qué tipo de propiedades son más rentables?")
                
                if 'bedrooms' in df_clean.columns and 'price' in df_clean.columns:
                    # Análisis de rentabilidad por tipo de propiedad
                    rentabilidad = df_clean.groupby('bedrooms')['price'].agg(['mean', 'count']).round(0)
                    rentabilidad = rentabilidad[rentabilidad['count'] >= 20]  # Solo con suficientes muestras
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏆 Rentabilidad por tipo:**")
                        for habitaciones, data in rentabilidad.iterrows():
                            tipo = "Estudio" if habitaciones == 0 else f"{int(habitaciones)} habitaciones"
                            st.metric(f"{tipo}", f"${data['mean']:,.0f}")
                    
                    with col2:
                        st.markdown("**📈 Recomendaciones de inversión:**")
                        mejor_tipo = rentabilidad['mean'].idxmin()  # El más barato = mejor retorno potencial
                        
                        st.markdown(f"""
                        • **Mejor relación precio-demanda**: {int(mejor_tipo)} habitaciones
                        • **Para alquiler**: Propiedades de 2-3 habitaciones
                        • **Para reventa rápida**: Casas familiares (3-4 habitaciones)
                        • **Evitar**: Propiedades de más de 5 habitaciones (nicho muy específico)
                        """)
                        
                        # ROI estimado
                        st.success("💰 **ROI estimado**: 8-12% anual en alquiler + apreciación")
                else:
                    st.info("💡 Para análisis de rentabilidad, necesitamos datos de habitaciones.")
            
            # Resumen ejecutivo
            st.markdown("---")
            st.markdown("### 📋 Resumen Ejecutivo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎯 ACCIÓN INMEDIATA**")
                st.markdown("Buscar propiedades por debajo de $500K con 3 habitaciones en zonas en desarrollo")
            
            with col2:
                st.markdown("**💰 PRESUPUESTO SUGERIDO**")
                st.markdown(f"Entre ${precio_mediano * 0.8:,.0f} - ${precio_mediano * 1.2:,.0f} para mejor balance riesgo-retorno")
            
            with col3:
                st.markdown("**⏰ TIMEFRAME**")
                st.markdown("Invertir en los próximos 3-6 meses aprovechando oportunidades actuales")
        
        # ----- SECCIÓN DE FEATURE ENGINEERING -----
        st.markdown("---")
        st.header("⚙️ 4. Feature Engineering")
        
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
            st.header("📊 5. Dataset Procesado")
            
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
    st.title("🤖 Módulo IA: Insights de Negocio con Inteligencia Artificial")
    st.markdown("**¿Qué hace esta sección?** Usa inteligencia artificial para responder preguntas de negocio sobre las propiedades.")
    
    # Verificar que hay datos
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_original
    
    if df is None:
        st.warning("⚠️ Por favor, primero carga datos en el módulo ETL")
        st.stop()
    
    # Configuración de API simplificada
    st.markdown("### 🔐 Configuración de API")
    
    api_key = st.text_input(
        "🔑 Clave de API de Groq (opcional):",
        type="password",
        help="Obtén tu clave gratuita en https://console.groq.com/ para activar el análisis con IA"
    )
    
    if api_key:
        # Preguntas de negocio orientadas a proyección comercial
        st.markdown("---")
        st.markdown("### 🎯 Preguntas de Negocio")
        
        # Definir preguntas más orientadas al negocio
        business_questions = {
            "Oportunidades de Inversión": {
                "question": "¿En qué áreas puedo encontrar las mejores propiedades por el precio?",
                "description": "Identifica zonas con propiedades de buen valor para inversión"
            },
            "Proyección de Ventas": {
                "question": "¿Cuál sería el precio ideal para vender mi propiedad?",
                "description": "Estima precios de venta basado en características similares"
            },
            "Análisis de Mercado": {
                "question": "¿Qué tipo de propiedades son más populares y rentables?",
                "description": "Analiza tendencias de demanda por tipo de propiedad"
            },
            "Comparativa Regional": {
                "question": "¿Dónde están las propiedades más caras vs más baratas?",
                "description": "Compara precios promedio por ubicación"
            },
            "Retorno de Inversión": {
                "question": "¿Qué propiedades me darían mejor retorno si las compro para rentar?",
                "description": "Calcula potencial de retorno para inversión en alquiler"
            }
        }
        
        selected_question = st.selectbox(
            "Selecciona una pregunta de negocio:",
            options=list(business_questions.keys()),
            format_func=lambda x: f"📊 {x}: {business_questions[x]['question']}"
        )
        
        if selected_question:
            st.info(f"💡 {business_questions[selected_question]['description']}")
        
        # Botón para análisis específico
        if st.button("📊 Analizar Pregunta de Negocio", type="primary"):
            if not api_key:
                st.error("❌ Por favor, ingresa tu clave de API de Groq")
            else:
                selected_q_data = business_questions[selected_question]
                with st.spinner("🤔 La IA está analizando tu pregunta de negocio..."):
                    # Crear prompt específico para la pregunta
                    prompt = f"""
                    Analiza estos datos de propiedades inmobiliarias para responder esta pregunta de negocio:
                    
                    PREGUNTA: {selected_q_data['question']}
                    CONTEXTO: {selected_q_data['description']}
                    
                    Datos disponibles:
                    - {len(df)} propiedades
                    - Precio promedio: ${df['price'].mean():,.0f} (si tiene columna price)
                    
                    Por favor responde de forma práctica y orientada a decisiones de negocio.
                    """
                    
                    insights, error = generate_ai_insights(df, api_key, custom_prompt=prompt)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.success("✅ Análisis completado")
                        st.markdown("### 🎯 Respuesta de Negocio")
                        st.markdown(insights)
        
        # Análisis automático completo
        st.markdown("---")
        st.markdown("### 🚀 Análisis Completo Automático")
        
        if st.button("🔍 Generar Reporte Completo", type="secondary"):
            with st.spinner("🤖 Generando reporte completo con IA..."):
                insights, error = generate_ai_insights(df, api_key)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Reporte generado")
                    st.markdown("### 📊 Insights Completos")
                    st.markdown(insights)
                    
                    # Descargar reporte
                    st.download_button(
                        label="⬇️ Descargar reporte completo",
                        data=insights,
                        file_name=f"reporte_ia_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
        
        # Chat interactivo personalizado
        st.markdown("---")
        st.markdown("### 💬 Consultas Personalizadas")
        st.markdown("**Haz preguntas específicas sobre las propiedades:**")
        
        user_question = st.text_input(
            "Tu pregunta:",
            placeholder="¿Cuáles son las mejores zonas para invertir con poco presupuesto?"
        )
        
        if user_question and st.button("📤 Obtener Respuesta"):
            with st.spinner("💭 Analizando tu pregunta..."):
                prompt = f"""
                Usuario pregunta: {user_question}
                
                Analiza los datos de propiedades inmobiliarias y responde de forma clara y práctica.
                Enfócate en dar consejos útiles para decisiones de negocio.
                """
                answer, error = generate_ai_insights(df, api_key, custom_prompt=prompt)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.markdown("**🤖 Respuesta:**")
                    st.markdown(answer)
    
    else:
        st.info("💡 Para usar el análisis con IA, ingresa tu clave de API de Groq (es gratuita)")
        
        # Mostrar métricas básicas sin IA
        st.markdown("### 📊 Resumen Básico de Datos")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Propiedades", f"{len(df):,}")
        with col2:
            st.metric("Variables Disponibles", df.shape[1])
        with col3:
            if 'price' in df.columns:
                st.metric("Precio Promedio", f"${df['price'].mean():,.0f}")

# ============================================================================
# MÓDULO 4: INTEGRACIÓN CON IA (GROQ) - ANTERIOR
# ============================================================================
elif modulo == "🤖 IA - Insights Inteligentes (OLD)":
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
