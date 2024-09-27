from typing import Callable, List
from models.model import Eva02ConvNextClassifier
from inference.inference import load_models
from utils.transform import AlbumentationsTransform
import streamlit as st

import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title='Sketch Image Classification', layout='wide')

st.title(':crystal_ball: Sketch Image Classifier in 500 classes:crystal_ball:')
st.markdown('---')

st.info('분류하고 싶은 이미지를 업로드 해주세요!')
uploaded_file = st.file_uploader('이미지 업로드', type=['jpg', 'png', 'jpeg'], label_visibility="hidden")


@st.cache_resource
def load_model(root_path: str):
    return load_models(root_path=root_path)

def inference(
    models: List,
    device: torch.device,
    transform: Callable,
    uploaded_file
):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)

    for model in models:
        model.to(device)
        model.eval()
    
    with torch.no_grad():
        image = image.to(device)

        all_logits = []
        for model in models:
            logits = model(image)
            all_logits.append(logits)
        
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        avg_probs = F.softmax(avg_logits, dim=1)
        preds = avg_probs.argmax(dim=1)
        predicted_probabilities = avg_probs[torch.arange(avg_probs.size(0)), preds]

    return preds, predicted_probabilities

def show_result(uploaded_file, probability, info_df: pd.DataFrame, class_name: int):
    original_image = Image.open(uploaded_file)

    class_info = info_df[info_df['target'] == class_name].iloc[0]
    class_image_path = os.path.join('../data/train', class_info['image_path'])
    class_image = Image.open(class_image_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image.resize((512, 512)), caption='업로드 한 이미지', use_column_width=True)
    
    with col2:
        st.image(class_image.resize((512, 512)), caption=f'클래스 {class_name}의 예시 이미지', use_column_width=True)
    
    if probability.item() > 0.5:
        st.info(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')
    else:
        st.error(f'예측된 클래스는 {class_name}이고, 신뢰도는 {probability.item():.2%}입니다.')

    
transform = AlbumentationsTransform(is_train=False)
models = load_model('./eva02convnext2_large_mixupcutmix_kfold') # model checkpoints folder location
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_info = pd.read_csv('../data/train.csv')

if st.button("예측하기"):
    # 모델 예측
    result, probability = inference(models, device, transform, uploaded_file)
    # 예측 결과를 보여주는 부분 (가상의 예측 결과)
    show_result(uploaded_file, probability, train_info, result.item())





