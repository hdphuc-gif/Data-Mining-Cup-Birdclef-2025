import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model_a.model import EfficientNetWrapper
from src.common.utils import load_dataset  # bạn cần tự định nghĩa
import yaml
import os
import argparse

def train(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetWrapper()
    model.to(device)

    # 1. Tạo class2idx từ tập train
    import pandas as pd
    train_df = pd.read_csv(cfg['data']['train'])
    all_labels = sorted(train_df['label'].unique())
    class2idx = {label: idx for idx, label in enumerate(all_labels)}

    # 2. Truyền class2idx vào DataLoader (nếu hàm load_dataset nhận class2idx)
    train_loader = load_dataset(cfg['data']['train'], audio_dir='', class2idx=class2idx,transform=None)
    val_loader = load_dataset(cfg['data']['val'], audio_dir='', class2idx=class2idx, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    os.makedirs(cfg['save_path'], exist_ok=True)
    best_val_loss = float('inf')
    save_best = cfg.get('save_best_only', False)

    for epoch in range(cfg['training']['epochs']):
        train_loss = train(model, train_loader, val_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
        # 💾 Ghi log vào logs/
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"{cfg['model']}_log.csv")
        write_header = not os.path.exists(log_path)

        with open(log_path, 'a') as log_file:
            if write_header:
                log_file.write("epoch,train_loss,val_loss\n")
            log_file.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f}\n")
        
        if save_best:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(cfg['save_path'], "efficientnet_best.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(cfg['save_path'], f"efficientnet_epoch{epoch+1}.pth"))

    if not save_best:
        torch.save(model.state_dict(), os.path.join(cfg['save_path'], "efficientnet_final.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
