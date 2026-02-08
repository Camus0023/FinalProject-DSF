# Configuración específica para Streamlit Cloud

import streamlit as st
import os

# Configurar el entorno para Streamlit Cloud
if 'STREAMLIT_SERVER_HEADLESS' in os.environ:
    # Estamos en Streamlit Cloud
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('client.showErrorDetails', True)

# Función para verificar que todos los archivos necesarios existen
def check_dependencies():
    """Verifica que todos los archivos necesarios estén presentes"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        os.path.join(base_dir, "data", "data_imperfecto_v2.csv"),
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def get_data_path():
    """Obtiene la ruta correcta para el archivo de datos"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "data_imperfecto_v2.csv")