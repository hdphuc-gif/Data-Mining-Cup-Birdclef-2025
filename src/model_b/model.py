import torch
import torch.nn as nn
import timm


class ConvNeXtImageWrapper(nn.Module):
    def __init__(self, num_classes=206, model_name="convnext_tiny", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.backbone(x)
        print("Shape after backbone:", x.shape)
        x = self.classifier(x)
        return x
