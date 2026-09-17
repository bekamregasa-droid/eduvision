import gradio as gr
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

model = ort.InferenceSession("model.onnx")
answer_key = [4, 3, 2, 7, 8]

def extract_digit(band):
    _, thresh = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    if w < 8 or h < 8:
        return 0
    crop = band[y:y+h, x:x+w]
    crop = cv2.resize(crop, (28, 28)).astype(np.float32) / 255.0
    crop = crop.reshape(1, 1, 28, 28)
    input_name = model.get_inputs()[0].name
    result = model.run(None, {input_name: crop})
    return int(np.argmax(result[0]))

def grade(image):
    if image is None:
        return ""
    gray = np.array(image.convert("L"))
    h = gray.shape[0]
    answer_zone = gray[int(h * 0.28):]
    band_h = answer_zone.shape[0] // 5
    predictions = []
    for i in range(5):
        band = answer_zone[i * band_h:(i + 1) * band_h, :]
        predictions.append(extract_digit(band))
    score = sum(p == a for p, a in zip(predictions, answer_key))
    return f"EduVision detects:\nScore: {score}/5"

css = """
.gradio-container {
    max-width: 440px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: #e8e8e8 !important;
    font-family: Arial, sans-serif !important;
    border-radius: 0 !important;
}
.main {
    padding: 0 !important;
}
#ev-header {
    background: #1a7a82;
    padding: 20px 22px 15px 22px;
}
#ev-header h1 {
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin: 0 0 3px 0;
    line-height: 1.1;
}
#ev-header p {
    color: rgba(255,255,255,0.92);
    font-size: 15px;
    font-style: italic;
    margin: 0;
}
#photo-card {
    background: white;
    border-radius: 16px;
    margin: 14px 14px 10px 14px;
    padding: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.13);
}
#bottom-row {
    display: flex;
    gap: 10px;
    margin: 0 14px 14px 14px;
    align-items: stretch;
}
#result-box textarea {
    background: #1a7a82 !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 17px !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 16px 14px !important;
    resize: none !important;
    line-height: 1.5 !important;
    box-shadow: none !important;
}
#ok-btn button {
    background: #1d8e97 !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 22px !important;
    font-weight: bold !important;
    height: 100% !important;
    min-width: 80px !important;
    min-height: 72px !important;
    box-shadow: none !important;
}
#ok-btn button:hover {
    background: #1a7a82 !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(css=css, title="EduVision") as app:
    gr.HTML("""
        <div id="ev-header">
            <h1>EduVision</h1>
            <p>Offline AI quiz grader for rural classroom</p>
        </div>
    """)

    with gr.Column(elem_id="photo-card"):
        photo = gr.Image(
            type="pil",
            label=None,
            show_label=False,
            height=420
        )

    with gr.Row(elem_id="bottom-row"):
        result = gr.Textbox(
            elem_id="result-box",
            show_label=False,
            interactive=False,
            lines=2,
            scale=3
        )
        ok = gr.Button("OK", elem_id="ok-btn", scale=1)

    photo.change(fn=grade, inputs=photo, outputs=result)
    ok.click(fn=lambda: "", inputs=None, outputs=result)

app.launch(share=True)
