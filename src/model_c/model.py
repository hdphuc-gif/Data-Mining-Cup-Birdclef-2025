import torch
import torch.nn as nn
import timm


class ConvNeXtBirdWrapper(nn.Module):
    def __init__(self, num_classes=206, model_name="convnext_base.fb_in22k_ft_in1k", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        print("Shape after backbone:", x.shape)
        if x.ndim == 4:
            x = torch.flatten(x, 1)  # về [batch, C]
        x = self.classifier(x)
        return x

