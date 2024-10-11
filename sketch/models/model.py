import pytorch_lightning as pl

import timm
from transformers import ConvNextV2ForImageClassification

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .utils import mixup_cutmix_collate_fn, SoftTargetCrossEntropy

class Eva02ConvNextClassifier(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, num_classes=500, lr=3e-5, weight_decay=1e-2):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.convnext = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-22k-224")
        self.eva02 = timm.create_model('eva02_large_patch14_clip_224.merged2b', pretrained=True, num_classes=0)

        self.convnext.train()

        self.convnext.classifier = nn.Identity()

        self.convnext_output_dim = 1536
        self.eva02_output_dim = 1024
        
        combined_dim = self.eva02_output_dim + self.convnext_output_dim # self.swin_output_dim + 

        self.classifier = nn.Linear(combined_dim, num_classes)
        
        self.lr = lr

        self.weight_decay = weight_decay
        self.loss_fn_crossentropy = nn.CrossEntropyLoss()
        self.loss_fn = SoftTargetCrossEntropy()


    def forward(self, pixel_values):
        convnext_features = self.convnext(pixel_values).logits

        eva02_output = self.eva02.forward_features(pixel_values)
        eva02_features = self.eva02.forward_head(eva02_output, pre_logits=True)

        combined_features = torch.cat((convnext_features, eva02_features), dim=1)
        
        # 결합된 특징을 classifier에 통과시켜 최종 출력
        logits = self.classifier(combined_features)
        return logits

    def training_step(self, batch):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn_crossentropy(logits, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.classifier.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}, 
                {'params': self.convnext.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
                {'params': self.eva02.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
            ]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Define train_loader
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            collate_fn=mixup_cutmix_collate_fn
        )
        return train_loader

    def val_dataloader(self):
        # Define val_loader
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        return val_loader