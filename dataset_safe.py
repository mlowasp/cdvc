import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
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


class SafeVideoDatasetTwoStream(Dataset):
    """
    RAM-safe dataset that returns:
      rgb  : FloatTensor [3, T, H, W] in [-1, 1]
      flow : FloatTensor [2, T, H, W] in [-1, 1]
      label: LongTensor []
      path : str
    """
    def __init__(
        self,
        root_split_dir: str,
        class_names: List[str],
        num_frames: int = 64,
        sampling: str = "random",           # random | uniform
        resize_short: int = 256,
        crop_size: int = 224,
        fixed_seed: Optional[int] = None,
        flow_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.samples = list_videos(root_split_dir, class_names)
        if not self.samples:
            raise RuntimeError(f"No videos found under: {root_split_dir}")

        self.num_frames = int(num_frames)
        self.sampling = sampling
        self.resize_short = int(resize_short)
        self.crop_size = int(crop_size)
        self.fixed_seed = fixed_seed

        self.flow_cfg = flow_cfg or {}
        self.flow_clip = float(self.flow_cfg.get("clip_value", 20.0))

        # Farneback params
        self.fb_pyr_scale = float(self.flow_cfg.get("pyr_scale", 0.5))
        self.fb_levels = int(self.flow_cfg.get("levels", 3))
        self.fb_winsize = int(self.flow_cfg.get("winsize", 15))
        self.fb_iterations = int(self.flow_cfg.get("iterations", 3))
        self.fb_poly_n = int(self.flow_cfg.get("poly_n", 5))
        self.fb_poly_sigma = float(self.flow_cfg.get("poly_sigma", 1.2))
        self.fb_flags = int(self.flow_cfg.get("flags", 0))

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

    def _resize_and_crop(self, frame_rgb: np.ndarray) -> np.ndarray:
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

    def _read_rgb_frames(self, path: str, indices: List[int]) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        frames: List[np.ndarray] = []
        last_good = None

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_good is not None:
                    frames.append(last_good.copy())
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._resize_and_crop(frame)
            last_good = frame
            frames.append(frame)

        cap.release()

        if not frames:
            frames = [np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)]

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        return frames[: self.num_frames]

    def _compute_flow_farneback(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        """
        frames_rgb: list length T of HxWx3 uint8
        returns flow: [T, H, W, 2] float32
        """
        T = len(frames_rgb)
        H, W, _ = frames_rgb[0].shape

        flows = np.zeros((T, H, W, 2), dtype=np.float32)

        prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
        for t in range(1, T):
            nxt = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None,
                self.fb_pyr_scale,
                self.fb_levels,
                self.fb_winsize,
                self.fb_iterations,
                self.fb_poly_n,
                self.fb_poly_sigma,
                self.fb_flags
            )
            flows[t] = flow
            prev = nxt

        return flows

    def _normalize_flow(self, flow_thw2: np.ndarray) -> np.ndarray:
        """
        flow_thw2: [T,H,W,2] float32 in pixel displacement units
        normalize to [-1,1] by clipping per component.
        """
        c = self.flow_clip
        flow_thw2 = np.clip(flow_thw2, -c, c)
        flow_thw2 = flow_thw2 / c  # now in [-1,1]
        return flow_thw2.astype(np.float32)

    def get_clip(self, idx: int, clip_key: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]

        cap = cv2.VideoCapture(sample.path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {sample.path}")
        total_frames = self._get_total_frames(cap)
        cap.release()

        indices = self._sample_indices(total_frames, key=clip_key)
        frames_rgb = self._read_rgb_frames(sample.path, indices)

        # RGB tensor: [3,T,H,W] in [-1,1]
        rgb_t = torch.from_numpy(np.stack(frames_rgb, axis=0)).permute(3, 0, 1, 2).float() / 255.0  # [3,T,H,W] in [0,1]
        rgb_t = rgb_t * 2.0 - 1.0

        # Flow: [T,H,W,2] -> [2,T,H,W] in [-1,1]
        flow_thw2 = self._compute_flow_farneback(frames_rgb)
        flow_thw2 = self._normalize_flow(flow_thw2)
        flow_t = torch.from_numpy(flow_thw2).permute(3, 0, 1, 2).contiguous()  # [2,T,H,W]

        label = torch.tensor(sample.label, dtype=torch.long)
        return rgb_t, flow_t, label, sample.path

    def __getitem__(self, idx: int):
        # training: each access yields one random clip
        return self.get_clip(idx, clip_key=str(random.random()))


class SingleVideoInferTwoStream:
    """
    Inference helper for a single video without directory scanning.
    """
    def __init__(self, video_path: str, num_frames: int, resize_short: int, crop_size: int,
                 fixed_seed: int = 12345, flow_cfg: Optional[Dict[str, Any]] = None):
        self.video_path = video_path
        self.num_frames = num_frames
        self.resize_short = resize_short
        self.crop_size = crop_size
        self.fixed_seed = fixed_seed
        self.flow_cfg = flow_cfg or {}

        # Create an uninitialized dataset instance and set fields directly
        self._ds = SafeVideoDatasetTwoStream.__new__(SafeVideoDatasetTwoStream)
        self._ds.samples = [VideoSample(video_path, 0)]
        self._ds.num_frames = int(num_frames)
        self._ds.sampling = "random"
        self._ds.resize_short = int(resize_short)
        self._ds.crop_size = int(crop_size)
        self._ds.fixed_seed = int(fixed_seed)
        self._ds.flow_cfg = self.flow_cfg

        # copy farneback params
        self._ds.flow_clip = float(self.flow_cfg.get("clip_value", 20.0))
        self._ds.fb_pyr_scale = float(self.flow_cfg.get("pyr_scale", 0.5))
        self._ds.fb_levels = int(self.flow_cfg.get("levels", 3))
        self._ds.fb_winsize = int(self.flow_cfg.get("winsize", 15))
        self._ds.fb_iterations = int(self.flow_cfg.get("iterations", 3))
        self._ds.fb_poly_n = int(self.flow_cfg.get("poly_n", 5))
        self._ds.fb_poly_sigma = float(self.flow_cfg.get("poly_sigma", 1.2))
        self._ds.fb_flags = int(self.flow_cfg.get("flags", 0))

    def get_clip(self, clip_key: str):
        return SafeVideoDatasetTwoStream.get_clip(self._ds, 0, clip_key=clip_key)
