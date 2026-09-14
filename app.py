import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit to classify it.")

@st.cache_resource
def load_onnx_model():
    # Load ONNX session (Pure C++ runtime, zero pickle/version errors)
    return ort.InferenceSession('model.onnx')

try:
    session = load_onnx_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB').resize((28, 28))
        st.image(img, caption='Uploaded Image', width=200)
        
        # Convert image to Normalized NumPy Array matching PyTorch preprocessing
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # Rearrange dimensions from (H, W, C) to (1, C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        
        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_np})[0]
        
        # Softmax calculation
        probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[0][pred_idx])
        
        st.write(f"### Prediction: {pred_idx}")
        st.write(f"**Confidence:** {confidence:.4f}")

except Exception as e:
    st.error("Error executing ONNX inference.")
    st.exception(e)
