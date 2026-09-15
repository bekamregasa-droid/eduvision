# EduVision: Offline Handwritten Digit Evaluator

EduVision is a lightweight computer vision tool designed to help educators evaluate handwritten student quizzes without relying on cloud infrastructure or active internet connectivity.

## Technical Architecture
- **Model Training:** Built and trained using native PyTorch.
- **Model Compilation:** Compiled to ONNX (`model.onnx`) format for lightweight, cross-platform execution without framework overhead.
- **Inference Pipeline:** Executes client-side via ONNX Runtime (`onnxruntime`), preprocessing normalized grayscale digit tensors.

## Repository Structure
- `app.py`: Streamlit / Gradio application script for processing uploaded images.
- `model.onnx`: Exported PyTorch model binary ready for runtime execution.
- `requirements.txt`: Minimal dependencies for low-overhead deployment.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
