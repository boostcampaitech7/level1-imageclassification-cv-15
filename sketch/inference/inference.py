from models.model import Eva02ConvNextClassifier

from typing import List
import pandas as pd
import tqdm
from datasets.dataset import CustomDataset
from utils.transform import AlbumentationsTransform
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

# 모델 로드 함수
def load_models(root_path: str):
    models = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            models.append(Eva02ConvNextClassifier.load_from_checkpoint(file_path))
    return models

# 추론 함수
def inference(models: List, device: torch.device, test_loader: DataLoader):
    for model in models:
        model.to(device)
        model.eval()

    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)

            all_logits = []
            for model in models:
                logits = model(images)
                all_logits.append(logits)

            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            avg_probs = F.softmax(avg_logits, dim=1)
            preds = avg_probs.argmax(dim=1)

            predictions.extend(preds.cpu().detach().numpy())

    return predictions

# main 함수
def main():
    root_path = './checkpoints/eva02convnext2_large_mixupcutmix_kfold'
    testdata_dir = "../data/test"
    testdata_info_file = "../data/test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_info = pd.read_csv(testdata_info_file)

    # 테스트 데이터셋 준비
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=AlbumentationsTransform(is_train=False),
        is_inference=True
    )

    # DataLoader 정의
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    # 모델 로드 및 추론 수행
    models = load_models(root_path=root_path)
    predictions = inference(models=models, device=device, test_loader=test_loader)

    # 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})

    os.makedirs("./result", exist_ok=True)
    test_info.to_csv("./result/eva02convnext2_kfold.csv", index=False)

if __name__ == "__main__":
    main()
