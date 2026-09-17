# EduVision: is Offline Handwritten Digit Evaluator

EduVision is a lightweight computer vision tool designed to help educators evaluate handwritten student quizzes. It doesn't rely on cloud infrastructure or active internet connectivity.

## Technical Architecture

* **Model Compilation:** Runs an optimized ONNX (`model.onnx`) model for lightweight, cross-platform execution without framework overhead.
* **Inference Pipeline:** Executes client-side via ONNX Runtime (`onnxruntime`), preprocessing normalized grayscale digit tensors.
* **User Interface:** Built with Gradio for a responsive, mobile-optimized UI.

## Repository Structure

* `app.py`: Gradio application script for processing uploaded images and grading answers.
* `model.onnx`: Exported ONNX model binary ready for runtime execution.
* `requirements.txt`: Minimal dependencies required for deployment (`gradio`, `onnxruntime`, `opencv-python-headless`, `Pillow`, `numpy`).

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
