import torch
from src.model_a.model import EfficientNetWrapper
from src.common.utils import load_single_input  # bạn cần định nghĩa
import yaml
import argparse

def load_model(cfg, device):
    model = EfficientNetWrapper()
    model.load_state_dict(torch.load(cfg['weights'], map_location=device))
    model.to(device)
    model.eval()
    return model

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    input_tensor = load_single_input(cfg['inference']['input_path'])  # shape: (1, 1, 224, 224)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
    print("Predicted probabilities:", output.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
