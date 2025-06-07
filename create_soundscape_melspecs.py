import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from main import CFG, audio2melspec
import cv2

cfg = CFG()

soundscape_dir = Path(cfg.train_soundscapes)
audio_files = list(soundscape_dir.glob("*.ogg"))

print(f"Found {len(audio_files)} soundscape files")

all_melspecs = {}

for audio_file in tqdm(audio_files):
    audio_data, _ = librosa.load(audio_file, sr=cfg.FS, duration=cfg.TARGET_DURATION)
    
    target_samples = int(cfg.TARGET_DURATION * cfg.FS)
    if len(audio_data) < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
    
    mel_spec = audio2melspec(audio_data, cfg)
    
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    
    all_melspecs[audio_file.stem] = mel_spec.astype(np.float32)

output_file = "data/train_soundscapes_melspec_first_5sec.npy"
np.save(output_file, all_melspecs, allow_pickle=True)

print(f"Saved {len(all_melspecs)} melspectrograms to {output_file}")
print(f"Shape: {next(iter(all_melspecs.values())).shape}")