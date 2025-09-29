# part3_dataset.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from part2_helpers import read_video_frames, list_videos

class SimpleVideoList:
    """Simple helper that expects videos/normal and videos/anomaly subfolders."""
    def __init__(self, root_folder):
        self.normal = list_videos(os.path.join(root_folder, "normal"))
        self.anom = list_videos(os.path.join(root_folder, "anomaly"))

    def train_list(self):
        # return small balanced list for quick experiments
        # label 0=normal, 1=anomaly
        files = []
        for v in self.normal:
            files.append((v, 0))
        for v in self.anom:
            files.append((v, 1))
        random.shuffle(files)
        return files

class SnippetDataset(Dataset):
    """
    Loads a video and returns snippet-level features (placeholder extraction).
    Each video -> 32 snippets, each snippet is represented as 2048-d feature (placeholder).
    For demo we compute cheap features on CPU; replace extractor for real experiments.
    """
    def __init__(self, file_label_list, snippet_frames=16, resize=(320,240), max_frames=512):
        self.data = file_label_list
        self.snippet_frames = snippet_frames
        self.resize = resize
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath, label = self.data[idx]
        frames = read_video_frames(filepath, resize=self.resize, max_frames=self.max_frames)
        # split to 32 segments
        num_snips = 32
        per = max(1, len(frames)//num_snips)
        feats = []
        for i in range(num_snips):
            s = i*per
            e = min(len(frames), s + self.snippet_frames)
            if s >= len(frames):
                snippet = [frames[-1]] * self.snippet_frames
            else:
                snippet = frames[s:e]
                while len(snippet) < self.snippet_frames:
                    snippet.append(snippet[-1])
            # simple placeholder feature: average RGB -> flattened small vector -> pad to 2048
            avg = np.mean(snippet, axis=0).astype('float32') / 255.0
            tmp = avg.mean(axis=2)  # grayscale HxW
            tmp = cv2_resize_and_flat(tmp, (32,32))  # helper below
            vec = tmp.reshape(-1)
            feat = np.zeros(2048, dtype='float32')
            feat[:vec.shape[0]] = vec
            feats.append(feat)
        feats = np.stack(feats, axis=0)  # (32,2048)
        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.float32)

# small helper uses cv2 for resizing (import only here to keep dependencies local)
import cv2
def cv2_resize_and_flat(img, size):
    return cv2.resize(img, size).astype('float32')
