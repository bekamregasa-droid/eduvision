# EduVision: Offline Handwritten Digit Evaluator

EduVision is a lightweight computer vision tool designed to help educators evaluate handwritten student quizzes without relying on cloud infrastructure or active internet connectivity[span_0](start_span)[span_0](end_span).

## Technical Architecture
- **Model Training:** Built and trained using native PyTorch (`torch.nn`)[span_1](start_span)[span_1](end_span).
- **Model Compilation:** Compiled to ONNX (`model.onnx`) format for lightweight, cross-platform execution without framework overhead[span_2](start_span)[span_2](end_span).
- **Inference Pipeline:** Executes client-side via ONNX Runtime (`onnxruntime`), preprocessing normalized grayscale digit tensors[span_3](start_span)[span_3](end_span).

## Repository Structure
- `app.py`: Streamlit / Gradio application script for processing uploaded images.
- `model.onnx`: Exported PyTorch model binary ready for runtime execution.
- `requirements.txt`: Minimal dependencies for low-overhead deployment.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
