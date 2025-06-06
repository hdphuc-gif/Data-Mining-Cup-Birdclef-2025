import torch
import torch.nn as nn
import yaml
import argparse
import os
import csv
import json

from src.model_a.model import EfficientNetWrapper
from src.model_b.model import ConvNeXtImageWrapper
from src.model_c.model import ConvNeXtBirdWrapper
from src.ensemble.ensemble import EnsembleModel
from src.common.utils import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_true.append(targets.numpy())
            y_pred.append((outputs.cpu().numpy() > 0.5).astype(int))
            y_prob.append(outputs.cpu().numpy())

    y_true = torch.tensor(y_true).view(-1, y_prob[0].shape[1]).numpy()
    y_pred = torch.tensor(y_pred).view(-1, y_prob[0].shape[1]).numpy()
    y_prob = torch.tensor(y_prob).view(-1, y_prob[0].shape[1]).numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_macro": float(roc_auc_score(y_true, y_prob, average="macro"))
    }
    return metrics

def load_model(cls, weight_path, device):
    model = cls()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a = load_model(EfficientNetWrapper, cfg['weights'][0], device)
    model_b = load_model(ConvNeXtImageWrapper, cfg['weights'][1], device)
    model_c = load_model(ConvNeXtBirdWrapper, cfg['weights'][2], device)

    ensemble = EnsembleModel(model_a, model_b, model_c, weights=cfg['weights_ratio'])
    ensemble.to(device)

    val_loader = load_dataset(cfg['data']['val'], shuffle=False)
    metrics = evaluate(ensemble, val_loader, device)

    print("Ensemble metrics:", metrics)

    os.makedirs(cfg['save_path'], exist_ok=True)
    with open(os.path.join(cfg['save_path'], "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Ghi log CSV
    log_path = "logs/ensemble_log.csv"
    os.makedirs("logs", exist_ok=True)
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ensemble_name", "model_weights", "val_accuracy", "val_f1", "val_roc_auc"])
        writer.writerow([
            cfg.get("ensemble_name", "default"),
            cfg["weights_ratio"],
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["roc_auc_macro"]
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
