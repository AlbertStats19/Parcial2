from diffusers import StableDiffusionPipeline
from transformers import pipeline
import torch
import streamlit as st

def load_models():
    # Modelos sin GPU porque mi computador NO tiene NVDIA
    if "model_gen" not in st.session_state:
        model_id = "CompVis/stable-diffusion-v1-4"
        model_gen = StableDiffusionPipeline.from_pretrained(model_id)
        model_gen.to("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.model_gen = model_gen
    
    if "model_class" not in st.session_state:
        model_class = pipeline('image-classification', model='microsoft/resnet-50')
        st.session_state.model_class = model_class

def generate_image(prompt):
    # Genera una imagen a partir de un prompt
    with torch.no_grad():
        image = st.session_state.model_gen(prompt).images[0]
    return image

def classify_image(image):
    # Clasifica una imagen por medio de computer vision
    response = st.session_state.model_class(image)
    prediction = response[0]['label']
    return prediction