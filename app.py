import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit to classify it.")

@st.cache_resource
def load_model():
    # Load fastai learner
    from fastai.vision.all import load_learner, Path
    
    def parent_label(o):
        return Path(o).parent.name

    return load_learner('model.pkl')

try:
    learn = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', width=200)
        
        pred, pred_idx, probs = learn.predict(img)
        
        st.write(f"### Prediction: {pred}")
        st.write(f"**Confidence:** {probs[pred_idx]:.4f}")

except Exception as e:
    st.error("Model failed to load. Ensure Streamlit Cloud Python version is set to 3.11 in Advanced Settings.")
    st.exception(e)
