#!/usr/bin/env python
import logging
import os
import csv
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    Gemma3nForConditionalGeneration,
    CLIPVisionModel,
    CLIPImageProcessor,
)


def load_advbench_dataset(filepath):
    """Loads goal-target pairs from the adversarial benchmark CSV."""
    pairs = []
    try:
        with open(filepath, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pairs.append((row["goal"], row["target"]))
    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {filepath}")
    return pairs


def get_experiment_folder():
    """Creates and returns a new numbered experiment folder."""
    base = "experiments"
    os.makedirs(base, exist_ok=True)
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
    """Creates and returns a folder for storing generated images for a specific run."""
    p = os.path.join(exp_folder, f"images_{idx}")
    os.makedirs(p, exist_ok=True)
    return p


def write_parameters_csv(
    exp_folder, config_kwargs, seed, name, num_prompts, attack_type
):
    """Saves experiment parameters to a CSV file."""
    path = os.path.join(exp_folder, "parameters.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value"])
        w.writerow(["name", name])
        for k, v in config_kwargs.items():
            # Handle special string representations for fractions
            if k == "alpha":
                vstr = config_kwargs.get("alpha_str", v)
                w.writerow(["alpha", vstr])
            elif k == "eps":
                vstr = config_kwargs.get("eps_str", v)
                w.writerow(["eps", vstr])
            # Skip internal string representations
            elif k.endswith("_str"):
                continue
            else:
                w.writerow([k, v])
        w.writerow(["attack_type", attack_type])  # Save attack type
        w.writerow(["seed", seed])
        w.writerow(["num_prompts", num_prompts])
    logging.info(f"Saved parameters CSV to {path}")


def load_model_and_processor(model_id):
    """
    Loads a specified multimodal model and its associated processor.
    Supports:
      - google/gemma-3n-e2b-it
      - google/gemma-3-4b-it
      - llava-hf/llava-1.5-7b-hf
      - llava-rc      (LLaVA w/ RCLIP ViT-L backbone)
    """
    logging.info(f"Loading model and processor for: {model_id}")

    # Gemma3n
    if model_id == "google/gemma-3n-e2b-it":
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        logging.info(f"Loaded Gemma3n model with processor: {processor.__class__.__name__}")

    # Gemma3
    elif model_id == "google/gemma-3-4b-it":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    # Base LLaVA
    elif model_id == "llava-hf/llava-1.5-7b-hf":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    # LLaVA + robust CLIP
    elif model_id == "llava-rc":
        BASE = "llava-hf/llava-1.5-7b-hf"
        CLIP_ID = "RCLIP/CLIP-ViT-L-FARE2"
        logging.info(f"Constructing LLaVA-RC: Base={BASE}, CLIP={CLIP_ID}")

        clip_vision = CLIPVisionModel.from_pretrained(
            CLIP_ID, torch_dtype=torch.float16
        ).to("cuda")
        clip_vision.requires_grad_(False)

        llava = LlavaForConditionalGeneration.from_pretrained(
            BASE,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")

        llava.vision_tower = clip_vision
        processor = AutoProcessor.from_pretrained(BASE, use_fast=True)

        img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        img_proc.size = {
            "height": clip_vision.config.image_size,
            "width": clip_vision.config.image_size,
        }
        processor.image_processor = img_proc
        model = llava

    else:
        raise ValueError(f"Unrecognized model_id {model_id}")

    model.eval()
    logging.info("Model and processor loaded successfully.")
    return model, processor
