import torch
from main import CFG
import numpy as np
import pandas as pd

# cfg = CFG()
# model = BirdCLEFModel(cfg)
# checkpoint = torch.load("checkpoints/fold_0/best_model_fold_0-v19_50ep.ckpt")
# model.load_state_dict(checkpoint['state_dict'])
# # model.eval()


# x = np.load('data/train_soundscapes_melspec_first_5sec.npy', allow_pickle=True, mmap_mode=None).item()
# # y = np.load('data/kagglehub/datasets/viniciusschmidt/birdclef-first-5-sec-humanless-fmax-16000/versions/1/birdclef2025_melspec_first_5sec_humanless_fmax_16000.npy', allow_pickle=True).item()
# print(len(x))


# nfnet_df = pd.DataFrame(pd.read_pickle(cfg.nfnet_pred_path))
# print(nfnet_df['row_id'])

x = {'f': 2}
print('f' in x)