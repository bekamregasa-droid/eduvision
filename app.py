import os
import gradio as gr
from fastai.vision.all import *

# Explicitly re-declare parent_label so FastAI can unpickle model.pkl
def parent_label(o):
    return Path(o).parent.name

learn = load_learner('model.pkl')
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

image = gr.Image(type="pil")
label = gr.Label(num_top_classes=2)

demo = gr.Interface(
    fn=predict, 
    inputs=image, 
    outputs=label,
    title="MNIST Digit Classifier"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
