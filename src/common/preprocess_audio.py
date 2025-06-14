import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import noisereduce as nr
import re
import silero_vad
import torch

# ====== Cấu hình ======
CSV_PATH = "/kaggle/input/birdclef-2025/train.csv"
AUDIO_DIR = "data/raw/tra/kaggle/input/birdclef-2025/train_audio"
OUTPUT_DIR = "/kaggle/working/"
SEGMENT_DURATION = 5   # giây
SAMPLE_RATE = 32000    # sample rate gốc

# ====== Load Silero VAD model và hàm get_speech_timestamps ======
model = silero_vad.load_silero_vad()

# ====== Đọc dữ liệu ======
df = pd.read_csv(CSV_PATH)
df_filtered = df[df["rating"] >= 0].copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    file_path = os.path.join(AUDIO_DIR, row["filename"])
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"[LỖI] Không thể đọc {file_path}: {e}")
        continue

    duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(np.ceil(duration / SEGMENT_DURATION))

    for i in range(num_segments):
        start = int(i * SEGMENT_DURATION * sr)
        end = int(min((i + 1) * SEGMENT_DURATION * sr, len(y)))
        segment = y[start:end]

        # Bỏ qua segment quá ngắn (< 1s) để tránh lỗi
        if len(segment) < 1 * sr:
            continue

        # Giảm nhiễu
        segment_denoised = nr.reduce_noise(y=segment, sr=sr)
        segment_denoised = segment_denoised.astype(np.float32)

        # Kiểm tra dữ liệu hợp lệ
        if np.isnan(segment_denoised).any():
            continue
        if np.abs(segment_denoised).max() < 1e-4:
            continue

        # Normalize nếu cần
        max_abs = np.abs(segment_denoised).max()
        if max_abs > 1:
            segment_denoised = segment_denoised / max_abs

        # ====== RESAMPLE về 16kHz, mono để dùng cho Silero VAD ======
        if sr != 16000:
            segment_resampled = librosa.resample(segment_denoised, orig_sr=sr, target_sr=16000)
            sr_vad = 16000
        else:
            segment_resampled = segment_denoised
            sr_vad = sr

        if len(segment_resampled.shape) > 1:
            segment_resampled = segment_resampled[:, 0]

        segment_resampled = segment_resampled.astype('float32')
        audio_tensor = torch.from_numpy(segment_resampled)

        # ====== PHÁT HIỆN TIẾNG NGƯỜI bằng Silero VAD ======
        try:
            speech_timestamps = silero_vad.get_speech_timestamps(audio_tensor, model, sampling_rate=16000)
            if speech_timestamps:
                print(f"[INFO] Silero VAD phát hiện tiếng người (speech) trong segment {i} của {row['filename']}")
                continue  # Bỏ segment này
        except Exception as e:
            print(f"[LỖI] Silero VAD failed for segment {i} ({file_path}): {e}")
            continue

        # ====== Lưu segment hợp lệ ======
        species = row["primary_label"]
        safe_species = re.sub(r'[^\w\-_\. ]', '_', species)
        output_folder = os.path.join(OUTPUT_DIR, safe_species)
        os.makedirs(output_folder, exist_ok=True)

        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', row['filename'].replace('.ogg', ''))
        output_filename = f"{safe_filename}_seg{i}.wav"
        output_path = os.path.join(output_folder, output_filename)

        if os.path.exists(output_path):
            continue

        try:
            sf.write(output_path, segment_denoised, sr)
        except Exception as e:
            print(f"[LỖI] Không thể ghi {output_path}: {e}")
