"""
lib/rppg_pipeline.py
Fetches videos from S3 skin-analysis-data bucket,
runs rPPG models sequentially per architecture,
saves results to DB.
"""

import os
import json
import uuid
import tempfile
import logging
import numpy as np
import torch
import boto3
import gc

from lib.rppg_video import load_video, fix_frames, extract_face, frames_to_tensor
from lib.rppg_inference import (
    load_model, run_inference,
    compute_hr, compute_hrv,
    compute_hr_peak, compute_hrv_peak,
    compute_rr, compute_vitals
)
from lib.util import insert_rppg_result

logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

S3_INPUT_BUCKET  = "skin-analysis-data"
S3_MODEL_BUCKET  = "mltrainingodf"
S3_MODEL_PREFIX  = "open_rppg_models/open-rppg"

TMP_MODEL_DIR = "/tmp/models"
os.makedirs(TMP_MODEL_DIR, exist_ok=True)

# Run architectures in this exact order
ARCHITECTURE_ORDER = ["UBFC", "PURE", "iBVP", "BP4D", "MA-UBFC", "SCAMPS"]


# ─────────────────────────────────────────────
# PHYSFORMER INPUT PREP
# ─────────────────────────────────────────────
def prepare_physformer_input(x):
    B, C, T, H, W = x.shape
    if T < 32:
        repeat = 32 - T
        x = torch.cat([x, x[:, :, -1:].repeat(1, 1, repeat, 1, 1)], dim=2)
    x = x[:, :, :32]
    x = x.reshape(B, 3, 4, 8, H, W)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, 24, 4, H, W)
    x = x.repeat(1, 4, 1, 1, 1)
    return x


# ─────────────────────────────────────────────
# S3 HELPERS
# ─────────────────────────────────────────────
def _fetch_user_folders(user_id=None, s3_client=None):
    paginator = s3_client.get_paginator("list_objects_v2")
    folders   = {}

    prefix = f"{user_id}/" if user_id else ""

    for page in paginator.paginate(
        Bucket=S3_INPUT_BUCKET, Prefix=prefix, Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"].rstrip("/")
            folders[folder] = {
                "user_id":   folder,
                "video_key": None,
                "json_key":  None
            }

    if user_id and user_id not in folders:
        folders[user_id] = {
            "user_id":   user_id,
            "video_key": None,
            "json_key":  None
        }

    for folder in list(folders.keys()):
        for page in paginator.paginate(
            Bucket=S3_INPUT_BUCKET, Prefix=f"{folder}/"
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".webm") or key.endswith(".mp4"):
                    folders[folder]["video_key"] = key
                elif key.endswith("data.json"):
                    folders[folder]["json_key"] = key

    return sorted(
        [v for v in folders.values() if v["video_key"]],
        key=lambda x: x["user_id"]
    )


def _download_video(bucket, key, s3_client):
    suffix = os.path.splitext(key)[1]
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    s3_client.download_file(bucket, key, tmp.name)
    logger.info(f"⬇ Video: s3://{bucket}/{key}")
    return tmp.name


def _read_json(bucket, key, s3_client):
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Could not read {key}: {e}")
        return {}


def _list_model_keys_for_arch(arch, s3_client):
    """Return all .pth keys under the given architecture folder."""
    prefix    = f"{S3_MODEL_PREFIX}/{arch}"
    paginator = s3_client.get_paginator("list_objects_v2")
    keys      = []
    for page in paginator.paginate(Bucket=S3_MODEL_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pth"):
                keys.append(obj["Key"])
    return keys


# ─────────────────────────────────────────────
# RUN A SINGLE ARCHITECTURE (one or more .pth files)
# Returns dict  { model_name: { production: ..., reference: ... } }
# Cleans up all memory before returning.
# ─────────────────────────────────────────────
def _run_architecture(arch, tensor, fps, s3_client):
    """
    Download every model under `arch`, run inference, collect vitals,
    then wipe every intermediate tensor / model from memory.
    Returns arch_results dict (plain Python — no tensors).
    """
    arch_results = {}
    model_keys   = _list_model_keys_for_arch(arch, s3_client)

    if not model_keys:
        logger.warning(f"No .pth files found for architecture: {arch}")
        return arch_results

    for key in model_keys:
        local_model = os.path.join(TMP_MODEL_DIR, os.path.basename(key))
        model       = None
        signal      = None
        tensor_input = None

        try:
            logger.info(f"⬇ Model [{arch}]: {key}")
            s3_client.download_file(S3_MODEL_BUCKET, key, local_model)

            model, model_name = load_model(local_model)
            model.to("cpu").float()

            # PhysFormer needs a reshaped input
            tensor_input = (
                prepare_physformer_input(tensor)
                if "physformer" in model_name.lower()
                else tensor
            )

            signal = run_inference(model, tensor_input)

            # ── Production (FFT / zero-crossing) ──
            hr_z     = compute_hr(signal, fps)
            hrv_z    = compute_hrv(signal, fps)
            rr_z     = compute_rr(hr_z)
            vitals_z = compute_vitals(model_name, hr_z, hrv_z, rr_z, method="fft")

            # ── Reference (peak detection) ─────────
            hr_p     = compute_hr_peak(signal, fps)
            hrv_p    = compute_hrv_peak(signal, fps)
            rr_p     = compute_rr(hr_p)
            vitals_p = compute_vitals(model_name, hr_p, hrv_p, rr_p, method="peak")

            arch_results[model_name] = {
                "production": vitals_z,   # plain dict, no tensors
                "reference":  vitals_p,
            }

            logger.info(f"✅ {model_name} done")

        except Exception as e:
            import traceback
            logger.error(f"Model error [{arch}] ({key}): {e}")
            traceback.print_exc()

        finally:
            # ── Wipe everything for this model ─────
            if tensor_input is not None and tensor_input is not tensor:
                del tensor_input
            if signal is not None:
                del signal
            if model is not None:
                del model
            if os.path.exists(local_model):
                os.remove(local_model)
            gc.collect()

    return arch_results   # plain Python dict, safe to accumulate


# ─────────────────────────────────────────────
# PROCESS SINGLE USER
# ─────────────────────────────────────────────
def _process_user(folder, s3_client):
    uid       = folder["user_id"]
    video_key = folder["video_key"]
    json_key  = folder["json_key"]

    logger.info(f"Processing user: {uid}")

    input_data  = _read_json(S3_INPUT_BUCKET, json_key, s3_client) if json_key else {}
    local_video = _download_video(S3_INPUT_BUCKET, video_key, s3_client)

    try:
        frames, fps = load_video(local_video)

        if not frames:
            raise RuntimeError(f"No frames extracted from {video_key}")

        frames = frames[:128]
        faces  = fix_frames(extract_face(frames))

        if len(faces) < 32:
            logger.warning(f"Only {len(faces)} frames, padding to 32")
            faces = faces + faces[-1:] * (32 - len(faces))

        # Base tensor — stays alive across architectures (read-only, small)
        tensor = frames_to_tensor(faces, DEVICE).float()
        logger.info(f"Tensor shape: {tensor.shape}")

        # ── Run each architecture sequentially ───
        structured_summary = {}   # accumulates plain-Python dicts only

        for arch in ARCHITECTURE_ORDER:
            logger.info(f"━━━ Architecture: {arch} ━━━")
            arch_results = _run_architecture(arch, tensor, fps, s3_client)
            structured_summary.update(arch_results)
            # arch_results goes out of scope; gc cleans it
            gc.collect()
            logger.info(f"   {arch} complete — {len(arch_results)} model(s) processed")

        # ── Free the base tensor now ─────────────
        del tensor
        gc.collect()

        # ── Build the single merged JSON for DB ──
        production_result = {
            m: v["production"] for m, v in structured_summary.items()
        }
        reference_result = {
            m: v["reference"] for m, v in structured_summary.items()
        }

        db_row = insert_rppg_result(uid, input_data, production_result, reference_result)

        return {
            "user_id":    uid,
            "reading_id": db_row["reading_id"] if db_row else None,
            "models_run": len(structured_summary),
            "status":     "completed"
        }

    except Exception as e:
        import traceback
        logger.error(f"User {uid} failed: {e}")
        traceback.print_exc()
        return {
            "user_id": uid,
            "status":  "failed",
            "error":   str(e)
        }

    finally:
        try:
            os.remove(local_video)
        except Exception:
            pass


# ─────────────────────────────────────────────
# MAIN PIPELINE ENTRY
# ─────────────────────────────────────────────
def run_rppg_pipeline(user_id=None) -> list:
    s3_client    = boto3.client("s3")
    user_folders = _fetch_user_folders(user_id=user_id, s3_client=s3_client)

    if not user_folders:
        logger.warning(f"No valid folders found for user_id={user_id}")
        return []

    results = []
    for folder in user_folders:
        result = _process_user(folder, s3_client)
        results.append(result)
        logger.info(f"Result: {result}")

    return results