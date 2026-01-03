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


def _uniform_indices(num_total: int, num_frames: int) -> List[int]:
    if num_total <= 0:
        return [0] * num_frames
    if num_total >= num_frames:
        # linspace indices
        return [int(round(i)) for i in torch.linspace(0, num_total - 1, steps=num_frames).tolist()]
    # pad by repeating last
    idx = list(range(num_total))
    while len(idx) < num_frames:
        idx.append(num_total - 1)
    return idx


def _random_indices(num_total: int, num_frames: int) -> List[int]:
    if num_total <= 0:
        return [0] * num_frames
    if num_total >= num_frames:
        start = random.randint(0, max(0, num_total - num_frames))
        return list(range(start, start + num_frames))
    idx = list(range(num_total))
    while len(idx) < num_frames:
        idx.append(num_total - 1)
    return idx


class VideoDataset(Dataset):
    """
    Returns:
      frames: FloatTensor [T, 3, H, W] in [0,1]
      label:  LongTensor []
      path:   str
    """
    def __init__(
        self,
        root_split_dir: str,
        class_names: List[str],
        num_frames: int = 32,
        sampling: str = "uniform",
        resize_short: int = 256,
        crop_size: int = 224,
        backend: str = "opencv",
        fps: Optional[int] = None,
    ):
        self.samples = list_videos(root_split_dir, class_names)
        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found under: {root_split_dir}")
        self.class_names = class_names
        self.num_frames = num_frames
        self.sampling = sampling
        self.resize_short = resize_short
        self.crop_size = crop_size
        self.backend = backend
        self.fps = fps  # currently used only by torchvision backend if you extend it

    def __len__(self):
        return len(self.samples)

    def _read_video_opencv(self, path: str) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        frames = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if ok and frame is not None:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded for: {path}")
        return frames

    def _resize_and_crop(self, frame_rgb):
        # frame_rgb: HxWx3 uint8
        h, w, _ = frame_rgb.shape
        # resize so that short side = resize_short
        if h < w:
            new_h = self.resize_short
            new_w = int(round(w * (self.resize_short / h)))
        else:
            new_w = self.resize_short
            new_h = int(round(h * (self.resize_short / w)))
        frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # center crop
        ch = self.crop_size
        cw = self.crop_size
        y0 = max(0, (new_h - ch) // 2)
        x0 = max(0, (new_w - cw) // 2)
        frame = frame[y0:y0 + ch, x0:x0 + cw, :]
        if frame.shape[0] != ch or frame.shape[1] != cw:
            # pad if needed
            frame = cv2.copyMakeBorder(
                frame,
                0, max(0, ch - frame.shape[0]),
                0, max(0, cw - frame.shape[1]),
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return frame

    def _sample_indices(self, n_total: int) -> List[int]:
        if self.sampling == "random":
            return _random_indices(n_total, self.num_frames)
        return _uniform_indices(n_total, self.num_frames)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames_np = self._read_video_opencv(sample.path)

        n_total = len(frames_np)
        indices = self._sample_indices(n_total)

        out = []
        for i in indices:
            i = min(max(i, 0), n_total - 1)
            fr = self._resize_and_crop(frames_np[i])
            # to tensor [3,H,W], float [0,1]
            t = torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0
            out.append(t)

        frames = torch.stack(out, dim=0)  # [T,3,H,W]
        label = torch.tensor(sample.label, dtype=torch.long)
        return frames, label, sample.path


def compute_pos_weight(samples: List[VideoSample]) -> float:
    # For BCEWithLogitsLoss pos_weight, where positive is label==1
    pos = sum(1 for s in samples if s.label == 1)
    neg = sum(1 for s in samples if s.label == 0)
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)
