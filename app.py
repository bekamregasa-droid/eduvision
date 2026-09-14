import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.title("MNIST Digit Classifier (PyTorch + ONNX Engine)")
st.write("Upload a handwritten digit image to perform real-time ONNX inference.")

@st.cache_resource
def load_onnx_session():
    # Production-grade C++ execution session
    return ort.InferenceSession('model.onnx')

try:
    session = load_onnx_session()
    uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L').resize((28, 28))
        st.image(img, caption='Uploaded Image', width=180)
        
        # NumPy Pipeline matching PyTorch Normalization
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.1307) / 0.3081
        img_np = np.expand_dims(img_np, axis=(0, 1)).astype(np.float32)
        
        # Execute ONNX runtime graph
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_np})[0]
        
        # Calculate probabilities
        probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[0][pred_idx])
        
        st.write(f"### Predicted Digit: {pred_idx}")
        st.write(f"**Confidence Score:** {confidence * 100:.2f}%")

except Exception as e:
    st.error("Inference execution failed.")
    st.exception(e)
