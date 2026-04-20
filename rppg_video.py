# lib/rppg_video.py
"""
Video loading, face cropping, and tensor conversion utilities for rPPG pipeline.
"""

import cv2
import numpy as np
import torch

TARGET_LEN = 128  # Number of frames to standardize to
FACE_SIZE  = 128  # Output face crop size (square)


def load_video(path: str):
    """
    Load a video file and return a list of RGB frames along with the video FPS.

    Args:
        path (str): Path to video file.

    Returns:
        frames (list of np.ndarray): List of RGB frames.
        fps (float): Frames per second of the video.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames, fps


def fix_frames(frames: list, target: int = TARGET_LEN) -> list:
    """
    Adjust the number of frames to a target length.
    - If too long, evenly sample frames.
    - If too short, repeat the last frame.

    Args:
        frames (list): List of frames.
        target (int): Desired number of frames.

    Returns:
        list: Resampled or padded frames.
    """
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target).astype(int)
        return [frames[i] for i in idx]

    while len(frames) < target:
        frames.append(frames[-1])

    return frames


def extract_face(frames: list, size: int = FACE_SIZE) -> list:
    """
    Crop a central square region from each frame as the "face".

    Args:
        frames (list): List of frames.
        size (int): Output size of cropped face.

    Returns:
        list: List of cropped and resized face images.
    """
    out = []

    for f in frames:
        h, w, _ = f.shape
        cy, cx = h // 2, w // 2
        half = min(h, w) // 4
        crop = f[max(0, cy - half):cy + half, max(0, cx - half):cx + half]

        if crop.size == 0:
            crop = f

        out.append(cv2.resize(crop, (size, size)))

    return out


def frames_to_tensor(faces: list, device: torch.device) -> torch.Tensor:
    """
    Convert a list of face images into a PyTorch tensor suitable for rPPG models.

    Args:
        faces (list): List of cropped face frames.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (1, C, T, H, W), normalized to [0, 1].
    """
    arr = np.stack(faces, axis=0).transpose(3, 0, 1, 2)[np.newaxis]  # C x T x H x W
    return torch.from_numpy(arr.copy()).float().to(device) / 255.0