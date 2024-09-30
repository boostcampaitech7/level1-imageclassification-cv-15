# 🤗 [CV 기초 프로젝트] 스케치 이미지 분류 경진대회 🤗

이 레포지토리는 스케치 이미지 분류 프로젝트를 진행하며 관련 내용을 기록하고자 만들었습니다.

## 프로젝트

- 사용 언어: python
- 사용 라이브러리: pytorch, pytorch-lightning

- 주요 폴더 설명


- **experiments**: 데이터 증강 기법 및 모델링을 진행하며 작성한 노트북      
    - **basic**: 단일 모델 모델링      
    - **ensemble**: 앙상블 모델링   
        - **bagging**: 2개 이상의 모델의 예측 결과를 이용해 hard voting 또는 soft voting 하여 추론 실험   
        - **snapshot**: 단일 모델의 다양한 시점에서의 가중치를 활용하여 voting 하는 방식으로 추론 실험   
        - **stacking**: 2개의 모델을 활용하여 마지막 layer의 feature map을 concat하여 classifier의 input으로 활용하여 모델 학습 및 추론 실험   
        - **kfold**: dataset의 fold를 k개로 분할하여 각 fold가 모두 validation_set으로 활용되게끔 k개의 모델을 학습시키고 해당 모델들을 활용한 앙상블 추론   
    - **etc**: 실험하며 사용했던 temp 노트북, accuracy를 가늠하기 위한 노트북, 이미지   
    - **lightning_logs**: 학습 손실 및 검증 손실 로그 기록

- **sktech**: 최종 모델 eva02+convnext2_large_224_in22k (stacking)을 kfold 기법을 활용한 학습 및 추론 코드

    - **sketch_predictor**: streamlit을 활용한 웹 프로토타입 코드   

## 데이터 증강 실험

1. Affine 변환, ElasticTransform, Gaussian Noise, Motion Blur 등을 기본으로 사용했습니다.   
2. Mixup 및 Cutmix 확률로 적용하였습니다. (각 25% 확률)

- 두 증강 모두 약간의 성능 향상을 보였습니다.

## 앙상블 실험

### Stacking
- 같은 ViT 계열인 BeiT와 Swin Transformer를 stacking 하는 것보다 CNN 계열 모델과 ViT 계열 모델을 stacking 하는 방식이 좀 더 성능 개선에 도움을 주었습니다.
- 단일 모델로 가장 괜찮은 성능을 보인 ViT 계열의 모델 eva02와 CNN 계열의 모델 ConvNexTv2를 stacking 하여 모델링을 진행했습니다.

### Bagging(Voting)
- 마찬가지로 단일 모델만 사용하는 것 대비 성능 향상을 보였습니다.
- 다만, input image size 각각 다른 모델을 사용하고 너무 무거운 모델을 다량으로 사용하면, 추론 시간이 과도하게 소요되어 비효율적이라 판단해 보류하였습니다.

### Snapshot
- 모델의 validation_loss가 파동을 그리는 경우(로컬 미니마에 빠지는 경우)에는 해당 위치 가중치를 가진 모델들을 앙상블 하면 성능 개선을 볼 수 있을 것 같았습니다.
- 하지만, 모델 학습 과정에서 완만하게 validation_loss를 그리는 차트만 경험해서 실제 적용 시에 큰 성능 향상을 보이진 못했습니다. (ex - epoch: 10, 11, 12의 모델)

### K-Fold
- train dataset을 기존 8:2로 랜덤 분할하여 train, validation data를 split하는 과정을 균일하게 5개의 fold로 나누어 각각의 fold가 모두 validation data로 활용되게끔 5개의 모델을 학습하여 앙상블을 진행하였습니다.
- 해당 앙상블은 큰 성능 향상을 보였습니다.

### 최종 모델
* ViT 기반의 가장 좋은 성능을 보인 eva02와 CNN 기반의 가장 좋은 성능을 보인 ConvNexTv2를 stacking하여 모델 구조를 설계하였고, 해당 모델을 K-Fold 기법을 활용해 5개의 모델로 학습하여 앙상블하여 추론하는 방식으로 진행하였습니다. 해당 모델의 노트북 코드는 experiments/ensemble 폴더에, 모듈화된 코드와 streamlit 코드는 sketch 폴더에 구현되어있습니다.

