import pandas as pd
import numpy as np

def analizar_calidad_datos(df):
    reporte = {}

    # -------------------------------
    # 1. Valores nulos
    # -------------------------------
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100

    reporte["valores_nulos"] = pd.DataFrame({
        "nulos": nulos,
        "porcentaje (%)": porcentaje_nulos
    }).query("nulos > 0")

    # -------------------------------
    # 2. Duplicados
    # -------------------------------
    duplicados = df.duplicated().sum()
    reporte["duplicados"] = duplicados

    # -------------------------------
    # 3. Tipos de datos inconsistentes
    # -------------------------------
    tipos = df.dtypes
    columnas_objeto = df.select_dtypes(include="object").columns

    inconsistencias = {}

    for col in columnas_objeto:
        valores_unicos = df[col].dropna().map(type).value_counts()
        if len(valores_unicos) > 1:
            inconsistencias[col] = valores_unicos

    reporte["formatos_inconsistentes"] = inconsistencias

    # -------------------------------
    # 4. Outliers (IQR)
    # -------------------------------
    outliers = {}

    columnas_numericas = df.select_dtypes(include=[np.number]).columns

    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        cantidad_outliers = df[
            (df[col] < limite_inferior) | (df[col] > limite_superior)
        ].shape[0]

        if cantidad_outliers > 0:
            outliers[col] = cantidad_outliers

    reporte["outliers"] = outliers

    return reporte


df = pd.read_csv("/home/camus/Desktop/Fundamentos de Ciencia de Datos/Tests/test1-stream/FinalProject-DSF/data/data_imperfecto_v2.csv")
reporte = analizar_calidad_datos(df)
print("Valores nulos: ", reporte["valores_nulos"])
print("Duplicados: ", reporte["duplicados"])
print("Formatos inconsistentes: ", reporte["formatos_inconsistentes"])
print("Outliers: ", reporte["outliers"])


