## Flower Classification
Multi-class flower image classification (daisy, dandelion, sunflower, tulip, rose) using transfer learning. 

## Description
This project uses deep learning (transfer learning) to classify images of five common flowers: daisy, dandelion, sunflower, tulip, and rose. It includes a notebook for experimentation, a model definition, a pretrained model file, and an app.py file used for model deployment to huggingface spaces.
Threee feature extractor models were used:
* VIT_B_16
* ResNet50
* EffNet_B2
After evalutation, the VIT_B_16 feature extranctor model was chosen for deployment.

## Try it here
[https://huggingface.co/spaces/geitta/flower_classification](https://huggingface.co/spaces/geitta/flower_classification)

## Model info
* Based on a pretrained **Vision Transformer (ViT)**  
* Fine-tuned for **5 flower classes**  
* Pretrained weights: `pretrained_vit_festure_extractor_flower_classification.pth`

## Files
flower_classification/
├── app.py # inference script for hugging face space
├── flowers_classification.ipynb # Training, evaluation and deployment notebook
├── model.py # Model definition for hugging face space
├── pretrained_vit_festure_extractor_flower_classification.pth # state dictionary for vit model
├── requirements.txt
└── examples/ # Sample images for the hugging face space

## sample output (from huggingface space)
<img width="1616" height="810" alt="image" src="https://github.com/user-attachments/assets/0b361965-07f2-4f0a-8c03-8568c6045d9c" />
