import os 

import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from module import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindata_dir = "../../mpark/data/train"
traindata_info_file = "../../mpark/data/train.csv"
save_result_path = "../../mpark/train_result"

train_info = pd.read_csv(traindata_info_file)


num_classes = len(train_info['target'].unique())


train_df, val_df = train_test_split(
    train_info, 
    test_size=0.2,
    stratify=train_info['target']
)

transform_selector = TransformSelector(
    transform_type = "albumentations"
)
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)


train_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=train_transform
)
val_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=val_df,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False
)

model_selector = ModelSelector(
    model_type='timm', 
    num_classes=num_classes,
    model_name='resnet18', 
    pretrained=True
)
model = model_selector.get_model()


model.to(device)

optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001
)


scheduler_step_size = 30  
scheduler_gamma = 0.1  


steps_per_epoch = len(train_loader)


epochs_per_lr_decay = 2
scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=scheduler_step_size, 
    gamma=scheduler_gamma
)

loss_fn = Loss()

trainer = Trainer(
    model=model, 
    device=device, 
    train_loader=train_loader,
    val_loader=val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn, 
    epochs=5,
    result_path=save_result_path
)

trainer.train()