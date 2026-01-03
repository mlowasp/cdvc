import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import torch
from torch.utils.data import Dataset

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


@dataclass
class VideoSample:
    path: str
    label: int


def list_videos(root: str, class_names: List[str]) -> List[VideoSample]:
    samples: List[VideoSample] = []
    for label, cname in enumerate(class_names):
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith(VIDEO_EXTS):
                samples.append(VideoSample(os.path.join(cdir, fn), label))
    return samples


def compute_pos_weight(samples: List[VideoSample]) -> float:
    pos = sum(1 for s in samples if s.label == 1)
    neg = sum(1 for s in samples if s.label == 0)
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


class SafeVideoDataset(Dataset):
    """
    RAM-safe: decodes ONLY requested frames by seeking.
    Returns: frames [T,3,H,W], label, path
    """
    def __init__(
        self,
        root_split_dir: str,
        class_names: List[str],
        num_frames: int = 64,
        sampling: str = "random",   # random | uniform
        resize_short: int = 256,
        crop_size: int = 224,
        fixed_seed: Optional[int] = None,
    ):
        self.samples = list_videos(root_split_dir, class_names)
        if not self.samples:
            raise RuntimeError(f"No videos found under: {root_split_dir}")

        self.num_frames = int(num_frames)
        self.sampling = sampling
        self.resize_short = int(resize_short)
        self.crop_size = int(crop_size)
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.samples)

    def _get_total_frames(self, cap) -> int:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(1, n)

    def _sample_indices(self, total_frames: int, key: str = "") -> List[int]:
        T = self.num_frames
        if self.fixed_seed is not None:
            rnd = random.Random((self.fixed_seed, key, total_frames, T))
        else:
            rnd = random

        if total_frames <= 1:
            return [0] * T

        if self.sampling == "uniform":
            return [int(x) for x in torch.linspace(0, total_frames - 1, steps=T).tolist()]

        # random contiguous window if possible
        if total_frames >= T:
            start = rnd.randint(0, total_frames - T)
            return list(range(start, start + T))

        # otherwise pad
        idx = list(range(total_frames))
        while len(idx) < T:
            idx.append(total_frames - 1)
        return idx

    def _resize_and_crop(self, frame_rgb):
        h, w, _ = frame_rgb.shape
        if h < w:
            new_h = self.resize_short
            new_w = int(round(w * (self.resize_short / h)))
        else:
            new_w = self.resize_short
            new_h = int(round(h * (self.resize_short / w)))

        frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ch = self.crop_size
        cw = self.crop_size
        y0 = max(0, (new_h - ch) // 2)
        x0 = max(0, (new_w - cw) // 2)
        frame = frame[y0:y0 + ch, x0:x0 + cw, :]

        if frame.shape[0] != ch or frame.shape[1] != cw:
            frame = cv2.copyMakeBorder(
                frame,
                0, max(0, ch - frame.shape[0]),
                0, max(0, cw - frame.shape[1]),
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return frame

    def _read_frames_by_indices(self, path: str, indices: List[int]) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        frames: List[torch.Tensor] = []
        last_good = None

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                # fallback to last good frame if available
                if last_good is not None:
                    frames.append(last_good.clone())
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._resize_and_crop(frame)
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            last_good = t
            frames.append(t)

        cap.release()

        if not frames:
            # absolute fallback: black frame
            frames = [torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)]

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        return frames[: self.num_frames]

    def get_clip(self, idx: int, clip_key: str = "") -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample.path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {sample.path}")
        total_frames = self._get_total_frames(cap)
        cap.release()

        indices = self._sample_indices(total_frames, key=clip_key)
        frames = self._read_frames_by_indices(sample.path, indices)
        frames = torch.stack(frames, dim=0)  # [T,3,H,W]
        label = torch.tensor(sample.label, dtype=torch.long)
        return frames, label, sample.path

    def __getitem__(self, idx: int):
        # training: each access yields one random clip
        return self.get_clip(idx, clip_key=str(random.random()))
