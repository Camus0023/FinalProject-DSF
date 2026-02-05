import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dashboard de Análisis de Datos",
    layout="wide"
)

st.title("📊 Dashboard sencillo para análisis de datos")
st.write("Carga un archivo CSV y explora la información de forma visual y rápida.")

# Cargar archivo
uploaded_file = st.file_uploader("📂 Carga tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ---- Información general ----
    st.subheader("🔍 Vista general de los datos")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Filas", df.shape[0])

    with col2:
        st.metric("Columnas", df.shape[1])

    st.dataframe(df.head())

    # ---- Estadísticas ----
    st.subheader("📈 Estadísticas descriptivas")
    st.dataframe(df.describe())

    # ---- Selección de columnas numéricas ----
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) > 0:
        st.subheader("📊 Visualizaciones")

        selected_col = st.selectbox(
            "Selecciona una columna numérica",
            numeric_cols
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("Histograma")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_col], ax=ax)
            st.pyplot(fig)

        st.write("Gráfica de barras (conteo por rangos)")
        fig, ax = plt.subplots()
        df[selected_col].value_counts().head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("⚠️ El archivo no contiene columnas numéricas para graficar.")

else:
    st.info("👆 Sube un archivo CSV para comenzar.")
