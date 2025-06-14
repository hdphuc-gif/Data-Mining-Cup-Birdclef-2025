import torch
import torch.nn as nn
import timm

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=206, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)



    def forward(self, x):
        x = self.backbone(x)     # [batch, 1280]
        print("Shape after backbone:", x.shape)
        x = self.classifier(x)   # [batch, num_classes]
        return x
