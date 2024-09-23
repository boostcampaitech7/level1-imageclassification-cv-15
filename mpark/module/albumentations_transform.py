import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        
        common_transforms = [
            A.Resize(224, 224),  
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
            ToTensorV2()  
        ]
        
        if is_train:
            
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  
                    A.Rotate(limit=15),  
                    A.RandomBrightnessContrast(p=0.2),  
                ] + common_transforms
            )
        else:
            
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        
        transformed = self.transform(image=image)  
        
        return transformed['image']  