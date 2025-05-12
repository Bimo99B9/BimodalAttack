# utils/experiments_utils.py

#!/usr/bin/env python
import logging
import os
import csv
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    CLIPVisionModel,
    CLIPImageProcessor,
)


def load_advbench_dataset(filepath):
    pairs = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append((row["goal"], row["target"]))
    return pairs


def get_experiment_folder():
    base = "experiments"
    exps = [
        d
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("exp")
    ]
    maxn = 0
    for d in exps:
        try:
            n = int(d[3:])
            maxn = max(maxn, n)
        except ValueError:
            pass
    new = f"exp{maxn+1}"
    path = os.path.join(base, new)
    os.makedirs(path, exist_ok=True)
    return path


def get_images_folder(exp_folder, idx):
    p = os.path.join(exp_folder, f"images_{idx}")
    os.makedirs(p, exist_ok=True)
    return p


def write_parameters_csv(exp_folder, config_kwargs, seed, name, num_prompts):
    path = os.path.join(exp_folder, "parameters.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value"])
        w.writerow(["name", name])
        for k, v in config_kwargs.items():
            if k == "alpha":
                vstr = config_kwargs.get("alpha_str", v)
                w.writerow(["alpha", vstr])
            elif k == "eps":
                vstr = config_kwargs.get("eps_str", v)
                w.writerow(["eps", vstr])
            elif k.endswith("_str"):
                continue
            else:
                w.writerow([k, v])
        w.writerow(["seed", seed])
        w.writerow(["num_prompts", num_prompts])
    logging.info(f"Saved parameters CSV to {path}")


def load_model_and_processor(model_id):
    """
    Supports:
      - google/gemma-3-4b-it
      - llava-hf/llava-1.5-7b-hf
      - llava-rc   ← LLaVA w/ RCLIP ViT‐L backbone
    """
    # Gemma
    if model_id == "google/gemma-3-4b-it":
        m = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        m.eval()
        proc = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return m, proc

    # Base LLaVA
    if model_id == "llava-hf/llava-1.5-7b-hf":
        m = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        m.eval()
        proc = AutoProcessor.from_pretrained(model_id, use_fast=True)
        return m, proc

    # LLaVA + robust CLIP
    if model_id == "llava-rc":
        BASE = "llava-hf/llava-1.5-7b-hf"
        CLIP_ID = "RCLIP/CLIP-ViT-L-FARE2"

        # 1) load the robust CLIP vision tower
        clip_vision = CLIPVisionModel.from_pretrained(
            CLIP_ID, torch_dtype=torch.float16
        ).to("cuda")
        clip_vision.requires_grad_(False)

        # 2) load the base LLaVA
        llava = LlavaForConditionalGeneration.from_pretrained(
            BASE,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        llava.eval()

        # 3) swap in the RCLIP tower
        llava.vision_tower = clip_vision

        # 4) processor: start from LLaVA’s
        proc = AutoProcessor.from_pretrained(BASE, use_fast=True)

        # 5) attach a CLIPImageProcessor resized to the RCLIP size
        img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        img_proc.size = {
            "height": clip_vision.config.image_size,
            "width": clip_vision.config.image_size,
        }
        proc.image_processor = img_proc

        return llava, proc

    raise ValueError(f"Unrecognized model_id {model_id}")
