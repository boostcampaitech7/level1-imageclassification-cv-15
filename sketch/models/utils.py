
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)  # Apply softmax to get log-probabilities
        # Compute loss
        return torch.mean(torch.sum(-target * log_probs, dim=-1))

def rand_bbox(size, lam):
    """
    Bounding box의 위치를 무작위로 선택.
    :param size: 이미지 크기 (batch_size, channels, height, width)
    :param lam: lambda 값 (patch 비율)
    :return: x1, y1, x2, y2 (bounding box 좌표)
    """
    W = size[2]  # 이미지의 폭
    H = size[3]  # 이미지의 높이
    cut_rat = np.sqrt(1. - lam)  # lambda에 기반한 패치 비율
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Bounding box의 중앙 좌표를 무작위로 결정
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box의 좌측 상단과 우측 하단 좌표 계산
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def mixup_cutmix_collate_fn(batch, alpha=1.0, num_classes=500):
    images, labels = zip(*batch)
    
    # 이미지를 배치 텐서로 변환
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # 원핫 인코딩 (num_classes는 전체 클래스 개수)
    labels = F.one_hot(labels, num_classes=num_classes).float()

    # 0.5 확률로는 아무 것도 적용하지 않음
    prob = np.random.rand()
    
    if prob < 0.5:
        # 아무 증강도 적용하지 않음
        mixed_images, mixed_labels = images, labels
    else:
        # 나머지 0.5 확률 중 절반은 Mixup, 절반은 CutMix
        if prob < 0.75:
            # Mixup 적용
            lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
            index = torch.randperm(images.size(0))
            mixed_images = images.clone()
            mixed_labels = labels.clone()

            for i in range(images.size(0)):  # 각 이미지에 대해 반복
                mixed_images[i] = lam * images[i] + (1 - lam) * images[index[i]]
                mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[index[i]]
        else:
            # CutMix 적용
            lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
            index = torch.randperm(images.size(0))
            mixed_images = images.clone()
            mixed_labels = labels.clone()
            
            for i in range(images.size(0)):  # 각 이미지에 대해 반복
                x1, y1, x2, y2 = rand_bbox(images.size(), lam)
                # i번째 이미지에 대해 index[i]번째 이미지의 패치를 적용
                mixed_images[i, :, x1:x2, y1:y2] = images[index[i], :, x1:x2, y1:y2]
                # 라벨도 비율에 따라 섞어줌
                lam_i = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
                mixed_labels[i] = lam_i * labels[i] + (1 - lam_i) * labels[index[i]]

    return mixed_images, mixed_labels

