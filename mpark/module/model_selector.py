from .simple_cnn import SimpleCNN
from .torchvision_model import TorchvisionModel
from .timm_model import TimmModel

import torch.nn as nn

class ModelSelector:
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        return self.model