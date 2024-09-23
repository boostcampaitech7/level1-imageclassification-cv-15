from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        
        common_transforms = [
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
        
        if is_train:
            
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),  
                    transforms.RandomRotation(15),  
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
                ] + common_transforms
            )
        else:
            
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  
        
        transformed = self.transform(image)  
        
        return transformed  
