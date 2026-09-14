import streamlit as st
from fastai.vision.all import *

# Define custom labeling function to avoid pickle errors
def parent_label(o):
    return Path(o).parent.name

st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit to classify it.")

@st.cache_resource
def load_model():
    return load_learner('model.pkl')

learn = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded Image', width=200)
    
    pred, pred_idx, probs = learn.predict(img)
    st.write(f"### Prediction: {pred}")
    st.write(f"Confidence: {probs[pred_idx]:.4f}")
