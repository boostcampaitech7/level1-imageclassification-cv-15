from torch.utils.data import Subset
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from models.model import Eva02ConvNextClassifier
from datasets.dataset import CustomDataset
from utils.transform import AlbumentationsTransform


def load_data(train_data_dir, train_info_file):
    train_info = pd.read_csv(train_info_file)
    num_classes = len(train_info['target'].unique())
    
    dataset = CustomDataset(
        root_dir=train_data_dir,
        info_df=train_info,
        transform=AlbumentationsTransform(is_train=True)
    )
    return dataset, num_classes


def create_model(train_dataset, val_dataset, num_classes):
    return Eva02ConvNextClassifier(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=num_classes,
        lr=3e-5,
        weight_decay=1e-2
    )


def train_model(model, fold):
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/eva02convnext2_large_mixupcutmix_kfold/fold_{fold + 1}",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_weights_only=True
    )

    trainer = Trainer(max_steps=10000, gradient_clip_val=2, callbacks=[checkpoint_callback, early_stop_callback], accelerator='gpu')
    trainer.fit(model)


def main():
    traindata_dir = "../data/train"
    traindata_info_file = "../data/train.csv"

    dataset, num_classes = load_data(traindata_dir, traindata_info_file)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.targets)):  # Assuming `dataset.targets` exists
        print(f'Fold {fold + 1}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        model = create_model(train_subset, val_subset, num_classes)
        train_model(model, fold)


if __name__ == "__main__":
    main()
