import os
import logging
import random
import gc
import pickle
import time
import cv2
import math
import warnings
from pathlib import Path
import lightning as L

from lightning.pytorch import seed_everything
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification.auroc import AUROC
from tqdm import tqdm

import timm

from dotenv import load_dotenv

load_dotenv()  # take environment variables

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# KAGGLE
# birdclef_2025_path = kagglehub.competition_download('birdclef-2025')
# viniciusschmidt_birdclef_first_5_sec_512_window_fmax_14000_path = kagglehub.dataset_download('viniciusschmidt/birdclef-first-5-sec-512-window-fmax-14000')
# kdmitrie_bc25_separation_voice_from_data_path = kagglehub.notebook_output_download('kdmitrie/bc25-separation-voice-from-data')
# dmunster_bird25_weightedblend_0_88_path = kagglehub.notebook_output_download('dmunster/bird25-weightedblend-0-88')

birdclef_2025_path = "./data/kagglehub/competitions/birdclef-2025"
viniciusschmidt_birdclef_first_5_sec_512_window_fmax_14000_path = "./data/kagglehub/datasets/viniciusschmidt/birdclef-first-5-sec-512-window-fmax-14000"
kdmitrie_bc25_separation_voice_from_data_path = "./data/kagglehub/notebooks/kdmitrie/bc25-separation-voice-from-data/output/versions/13"
dmunster_bird25_weightedblend_0_88_path = "./data/kagglehub/notebooks/dmunster/bird25-weightedblend-0-88/output/versions/2"

"""## Configuration"""

class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 1

    OUTPUT_DIR = '/kaggle/working/'

    train_datadir = birdclef_2025_path + '/train_audio'
    train_csv = birdclef_2025_path + '/train.csv'
    test_soundscapes = birdclef_2025_path + '/test_soundscapes'
    train_soundscapes = birdclef_2025_path + '/train_soundscapes'
    submission_csv = birdclef_2025_path + '/sample_submission.csv'
    taxonomy_csv = birdclef_2025_path + '/taxonomy.csv'
    nfnet_pred_path = dmunster_bird25_weightedblend_0_88_path + "/nfnet_pred.pkl"
    seresnext_pred_path = dmunster_bird25_weightedblend_0_88_path + "/seresnext_pred.pkl"

    with open(kdmitrie_bc25_separation_voice_from_data_path + '/train_voice_data.pkl', 'rb') as f:
        human_voices_train = pickle.load(f)

    # spectrogram_npy = '/kaggle/input/birdclef-first-5-sec-humanless/birdclef2025_melspec_first_5sec_humanless.npy'
    spectrogram_npy = viniciusschmidt_birdclef_first_5_sec_512_window_fmax_14000_path + '/versions/1/birdclef2025_melspec_first_5sec_humanless_512_window_fmax_14000.npy'

    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1

    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 512
    HOP_LENGTH = 128
    N_MELS = 64
    N_MFCC = 128
    FMIN = 50
    FMAX = 14000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    batch_size = 32
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0]

    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5
    mixup_alpha = 0.5

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()

print(f"Using {cfg.device}")

"""## Utilities"""

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

import pickle
import pandas as pd

def get_pseudo_train_df():
    train_df = pd.read_csv(cfg.train_csv)
    per_class_thresholds = get_per_class_thresholds(train_df)

    nfnet_df = pd.DataFrame(pd.read_pickle(cfg.nfnet_pred_path))
    seresnext_df = pd.DataFrame(pd.read_pickle(cfg.seresnext_pred_path))

    df = nfnet_df.merge(seresnext_df, on='row_id', how='inner')

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = list(taxonomy_df['primary_label'].unique())
    idx_to_label = {idx: label for idx, label in enumerate(species_ids)}
    num_classes = len(species_ids)

    weights = [0.75, 0.25]
    for col in species_ids:
        df[col] = np.average(
            [df[f'{col}_x'], df[f'{col}_y']],
            weights=weights,
            axis=0
        )

    df[['filename', 'end_time']] = df['row_id'].str.rsplit('_', n=1, expand=True)
    df['end_time'] = df['end_time'].astype(int)
    df = df.groupby('filename').sample(1).reset_index(drop=True)

    df['filepath'] = df['filename'].apply(lambda sid: f'{cfg.train_soundscapes}/{sid}.ogg')
    df['primary_label'] = df[species_ids].idxmax(axis=1)


    df['secondary_labels'] = (
        df[species_ids]
        .apply(lambda col: col >= per_class_thresholds[col.name], axis=0)
        .apply(lambda x: x.index[x].values, axis=1)
        .astype("string")
        .str.replace("[]", "['']")
    )

    plabel_df = df.drop(species_ids, axis=1)

    return pd.concat([train_df, plabel_df])

def get_per_class_thresholds(df):
    label_counts = df['primary_label'].value_counts().to_dict()

    # Step 2: Set dynamic thresholds (less common → lower threshold)
    min_thresh = 0.55
    max_thresh = 0.99

    max_count = max(label_counts.values())
    min_count = min(label_counts.values())

    per_class_thresholds = {}
    for label, count in label_counts.items():
        commonness = (count - min_count) / (max_count - min_count + 1e-8)  # 0 = rare, 1 = common
        per_class_thresholds[label] = min_thresh + (commonness ** 0.5) * (max_thresh - min_thresh)  # Apply square root

    # Sanity check
    print("Dynamic threshold example:")
    for label in sorted(label_counts, key=label_counts.get)[:5]:  # 5 least frequent
        print(f"{label} (rare): {per_class_thresholds[label]:.3f}")
    for label in sorted(label_counts, key=label_counts.get, reverse=True)[:5]:  # 5 most frequent
        print(f"{label} (common): {per_class_thresholds[label]:.3f}")
    return per_class_thresholds


"""## Pre-processing
These functions handle the transformation of audio files to mel spectrograms for model input, with flexibility controlled by the `LOAD_DATA` parameter. The process involves either loading pre-computed spectrograms from this [dataset](https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms) (when `LOAD_DATA=True`) or dynamically generating them (when `LOAD_DATA=False`), transforming audio data into spectrogram representations, and preparing it for the neural network.
"""

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm

def remove_human_voices(audio_data, windows, cfg):
    for i, window in enumerate(windows):
        if i == 0:
            output_array = audio_data[0:int(window['start']*cfg.FS)]
        if i == (len(windows) - 1):
            output_array = np.append(output_array, audio_data[int(window['end']*cfg.FS):])
            continue
        output_array = np.append(
            output_array,
            audio_data[int(window['end']*cfg.FS):int(windows[i+1]['start']*cfg.FS)]
        )
    return output_array

def process_audio_file(audio_path, cfg, end_time=None):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        if windows := cfg.human_voices_train.get(audio_path, None):
            audio_data = remove_human_voices(audio_data, windows, cfg)

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Extract random 5 seconds or before end_time
        if not pd.isna(end_time):
            random_audio = audio_data[int(end_time * cfg.FS)-target_samples:int(end_time * cfg.FS)]
        elif len(audio_data) < target_samples:
            random_audio = np.pad(audio_data,
                            (0, target_samples - len(audio_data)),
                            mode='constant')
        elif len(audio_data) == target_samples:
            random_audio = audio_data
        else:
            start_idx = np.random.randint(low=0, high=(len(audio_data) - target_samples))
            end_idx = start_idx + target_samples
            random_audio = audio_data[start_idx:end_idx]

        mel_spec = audio2melspec(random_audio, cfg)

        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 1000:
            break
        try:
            samplename = row['samplename']
            filepath = row['filepath']

            mel_spec = process_audio_file(filepath, cfg)

            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec

        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")

    return all_bird_data

"""## Dataset Preparation and Data Augmentations
We'll convert audio to mel spectrograms and apply random augmentations with 50% probability each - including time stretching, pitch shifting, and volume adjustments. This randomized approach creates diverse training samples from the same audio files
"""

def apply_spec_augmentations(spec):
    """Apply augmentations to spectrogram"""

    # Time masking (horizontal stripes)
    if random.random() < 0.5:
        num_masks = random.randint(1, 3)
        for _ in range(num_masks):
            width = random.randint(5, 20)
            start = random.randint(0, spec.shape[2] - width)
            spec[0, :, start:start+width] = 0

    # Frequency masking (vertical stripes)
    if random.random() < 0.5:
        num_masks = random.randint(1, 3)
        for _ in range(num_masks):
            height = random.randint(5, 20)
            start = random.randint(0, spec.shape[1] - height)
            spec[0, start:start+height, :] = 0

    # Random brightness/contrast
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        bias = random.uniform(-0.1, 0.1)
        spec = spec * gain + bias
        spec = torch.clamp(spec, 0, 1)

    return spec

class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # if 'filepath' not in self.df.columns:
        self.df['filepath'] = self.df.apply(
            lambda row: self.cfg.train_datadir + '/' + row['filename']
            if pd.isna(row.get('end_time')) and pd.isna(row.get('filepath'))
            else row['filepath'],
            axis=1
        )
            # self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename

        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        end_time = row.get('end_time')
        spec = None

        if not pd.isna(end_time):
            spec = process_audio_file(row['filepath'], self.cfg, row['end_time'])
        elif self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']

            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""

        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

class LightningBirdCLEFDataset(L.LightningDataModule):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode

        # REQUIRE spectrograms to be provided - no disk I/O fallbacks
        if spectrograms is None:
            raise ValueError("spectrograms must be provided - this dataset is memory-only")
        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # Create samplename mapping
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        # Filter out samples that don't have spectrograms in memory
        initial_count = len(self.df)
        self.df = self.df[self.df['samplename'].isin(self.spectrograms.keys())].reset_index(drop=True)
        final_count = len(self.df)

        print(f"Memory-only dataset: {final_count}/{initial_count} samples have spectrograms in memory for {mode}")

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        end_time = row.get('end_time')
        spec = None

        if not pd.isna(end_time):
            spec = process_audio_file(row['filepath'], self.cfg, row['end_time'])
        elif self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']

            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""

        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target

def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    result = {key: [] for key in batch[0].keys()}

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])

    return result

"""## Model Definition"""

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.feat_dim = backbone_out

        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits,
                                       logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LightningBirdClEFModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = True
        self.cfg = cfg
        self.criterion = get_criterion(self.cfg)
        self.model = BirdCLEFModel(cfg)

        self.train_auc = AUROC(task='multilabel', num_labels=cfg.num_classes, average="macro")
        self.val_auc = AUROC(task='multilabel', num_labels=cfg.num_classes, average="macro")

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def training_step(self, batch, batch_idx):
        inputs = batch['melspec']
        targets = batch['target']

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        probs = torch.sigmoid(outputs)
        self.train_auc.update(probs, targets.int())

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['melspec']
        targets = batch['target']

        outputs = self.model(inputs, targets)
        loss = self.criterion(outputs, targets)

        probs = torch.sigmoid(outputs)
        self.val_auc.update(probs, targets.int())

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def on_training_epoch_end(self):
        auc = self.train_auc.compute()
        self.log('train_auc', auc, prog_bar=True, logger=True)
        self.train_auc.reset()

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log('val_auc', auc, prog_bar=True, logger=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return get_optimizer(self.model, self.cfg)



"""## Training Utilities
We are configuring our optimization strategy with the AdamW optimizer, cosine scheduling, and the BCEWithLogitsLoss criterion.
"""

def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
    return optimizer

def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None
    else:
        scheduler = None
    return scheduler

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
    return criterion

def calculate_auc(targets, outputs):

    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):

        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)

    return np.mean(aucs) if aucs else 0.0

"""## Training!"""
if __name__ == "__main__":
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    cfg = CFG()
    set_seed(cfg.seed)
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')

    print("Starting Lightning K-Fold Cross-Validation!")
    print(f"Workers: {cfg.num_workers}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Folds: {len(cfg.selected_folds)} folds")


    # Load data
    train_df = get_pseudo_train_df()
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    cfg.num_classes = len(taxonomy_df['primary_label'].tolist())

    # Option to generate spectrograms or use pre-computed
    generate_spectrograms_flag = False
    spectrograms = None

    if generate_spectrograms_flag:
        print("Will generate spectrograms from audio files")
        spectrograms = generate_spectrograms(train_df, cfg)
    else:
        print("Loading pre-computed spectrograms...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} spectrograms into RAM")
        except Exception as e:
            print(f"Failed to load: {e}. Generating spectrograms...")
            spectrograms = generate_spectrograms(train_df, cfg)

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    best_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')

        # Prepare datasets
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = LightningBirdCLEFDataset(fold_train_df, cfg, spectrograms, mode='train')
        val_dataset = LightningBirdCLEFDataset(fold_val_df, cfg, spectrograms, mode='valid')

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Create Lightning model
        model = LightningBirdClEFModel(cfg)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/fold_{fold}",
            filename=f"best_model_fold_{fold}",
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            verbose=True
        )

        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            deterministic=True
        )

        # Train!
        print(f"Training fold {fold}...")
        trainer.fit(model, train_loader, val_loader)

        # Get best score
        if checkpoint_callback.best_model_score is not None:
            best_score = checkpoint_callback.best_model_score.detach().cpu()
            best_scores.append(best_score)
            print(f"Fold {fold} complete! Best val_auc: {best_score:.4f}")

    print("\n" + "="*60)
    print("K-Fold Cross-Validation Results:")
    for i, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[i]}: {score:.4f}")
    print(f"Mean val_auc: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")
