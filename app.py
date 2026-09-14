import gradio as gr
from fastai.vision.all import load_learner, PILImage

learn = load_learner('model.pkl')

def predict(img):
    if img is None:
        return {}
    fastai_img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(fastai_img)
    labels = learn.dls.vocab
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="MNIST Digit Classifier",
    description="Upload a handwritten digit image for real-time deep learning inference."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
