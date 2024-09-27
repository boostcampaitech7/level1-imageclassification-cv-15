import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import DATASET_MEAN, DATASET_STD

import numpy as np
import torch

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),  # 최대 15도 회전
                    A.Affine(scale=(1, 1.5), shear=(-10, 10), p=0.5),
                    A.ElasticTransform(alpha=10, sigma=50, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),  # 밝기 및 대비 무작위 조정
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    # A.CoarseDropout(p=0.5, max_holes=8, min_height=(30), max_height=(50), min_width=30, max_width=50, fill_value=255),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed['image']  # 변환된 이미지의 텐서를 반환