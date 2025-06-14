#!/bin/bash

echo "🚀 Bắt đầu huấn luyện mô hình..."

# Train từng mô hình
#python -m src.model_a.train --config configs/model_a.yaml
python src/model_a/train.py --config configs/model_a.yaml
python src/model_b/train.py --config configs/model_b.yaml
python src/model_c/train.py --config configs/model_c.yaml

echo "✅ Huấn luyện xong. Bắt đầu đánh giá..."

# Evaluate từng mô hình
python src/evaluate.py --config configs/model_a.yaml
python src/evaluate.py --config configs/model_b.yaml
python src/evaluate.py --config configs/model_c.yaml

# Evaluate tổ hợp mô hình
python evaluate_ensemble.py --config configs/ensemble.yaml

echo "🎉 Pipeline hoàn tất! Xem logs/ và results/ để biết chi tiết."