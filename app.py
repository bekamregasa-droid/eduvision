import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit to classify it.")

@st.cache_resource
def load_model():
    model = torch.jit.load('model.pt', map_location=torch.device('cpu'))
    model.eval()
    return model

try:
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', width=200)
        
        # Preprocessing matching MNIST fastai pipeline
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        
        tensor_img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
        st.write(f"### Prediction: {pred_idx}")
        st.write(f"**Confidence:** {probs[pred_idx].item():.4f}")

except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
