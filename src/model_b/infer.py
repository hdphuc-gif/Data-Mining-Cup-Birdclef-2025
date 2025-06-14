import torch
from src.model_b.model import ConvNeXtImageWrapper
from src.common.utils import load_single_input
import yaml
import argparse

def load_model(cfg, device):
    model = ConvNeXtImageWrapper()
    model.load_state_dict(torch.load(cfg['weights'], map_location=device))
    model.to(device)
    model.eval()
    return model

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    input_tensor = load_single_input(cfg['inference']['input_path'])
    if input_tensor.shape[1] == 1:
        input_tensor = input_tensor.repeat(1, 3, 1, 1)
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
