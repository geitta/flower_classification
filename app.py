import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
### model and transforms preparation
#create vit model
vit, vit_transforms = create_vit_model(num_classes=5):
#load saved weights
vit.load_state_dict(torch.load(f="pretrained_vit_festure_extractor_flower_classification.pth", 
                               map_location=torch.device('cpu')))
### predict function
# create predict function
def predict(img) -> Tuple[Dict, float]:
  """transforms and perfroms a prediction on img and returns prediction and time taken"""
  #start the time
  start_time = time.time()
  #transform the target image and add a batch dim
  img = vit_transforms(img).unsqueeze(0)
  #put the model in eval mode and turn on inference
  vit.eval()
  with torch.inference_mode():
    # pass the transformed imag thru the model and turn the prediction logits into prediction probabilities
    pred_probs = torch.softmax(vit(img), dim =1)
  # create a prediction label and prediction probability
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
  # calculate prediction time
  end_time = time.time()
  pred_time = round(end_time - start_time, 5)
  # return the prediction dictionary and prediction time
  return pred_labels_and_probs, pred_time
### gradio app
# create title, description and article strings
title = "flower classification"
description = "a vit_16 feature extractor computer vision model to classify images of flowers as: daisy, dandelion, rose, sunflower, and tulip"
article = "created at [flower classification](githhublink to repository)"

# create examples list from examples dir
example_list = [["examples/" + example] for example in os.listdir("examples")]

# create gradio demo
demo = gr.Interface(fn=predict, #mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs
                    outputs=[gr.Label(num_top_classes=5, label="predictions"), #what are the outputs
                             gr.Number(label="prediction time (s)")], #our fn has two outputs, therefore we have 2 outputs
                    examples = example_list, 
                    title=title, 
                    description=description, 
                    article=article
                    )
#launch the demo
demo.launch()
