import streamlit as st
from utils import load_models, generate_image, classify_image
from PIL import Image

st.set_page_config(layout='wide')

load_models()

st.title('Generación y Clasificación de Imágenes con HuggingFace')

with st.container():
    col1, col2 = st.columns(2)

    # Columna 1: Generación de Imágenes
    with col1:
        st.header("Generador de Imágenes")
        prompt = st.text_input("Introduce una descripción para generar una imagen:")
        if st.button('Generar Imagen'):
            if prompt:
                image = generate_image(prompt)
                st.image(image, caption="Imagen Generada")
                st.session_state.generated_image = image
            else:
                st.warning("Por favor, introduce una descripción.")

    # Columna 2: Clasificación de Imágenes
    with col2:
        st.header("Clasificador de Imágenes")
        uploaded_file = st.file_uploader("Carga una imagen para clasificar:", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen a clasificar")
            if st.button('Clasificar Imagen'):
                prediction = classify_image(image)
                st.write(f"Clasificación: {prediction}")

# Función de clasificación de imagen generada
if 'generated_image' in st.session_state and st.button('Clasificar Imagen Generada en Col1'):
    prediction = classify_image(st.session_state.generated_image)
    st.write(f"Clasificación de la imagen generada: {prediction}")