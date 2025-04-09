#!/usr/bin/env python
import logging
import os
import csv

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)


# --- Load advbench dataset ---
def load_advbench_dataset(filepath):
    pairs = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append((row["goal"], row["target"]))
    return pairs


# --- Folder helper functions ---


def get_experiment_folder():
    base_dir = "experiments"
    existing_experiments = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")
    ]
    max_num = 0
    for d in existing_experiments:
        try:
            num = int(d[3:])
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    new_exp_num = max_num + 1
    experiment_folder_name = f"exp{new_exp_num}"
    experiment_folder = os.path.join(base_dir, experiment_folder_name)
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder


def get_images_folder(experiment_folder, pair_number):
    images_folder = os.path.join(experiment_folder, f"images_{pair_number}")
    os.makedirs(images_folder, exist_ok=True)
    return images_folder


def write_parameters_csv(experiment_folder, config_kwargs, seed, name, num_prompts):
    parameters_csv_path = os.path.join(experiment_folder, "parameters.csv")
    with open(parameters_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["name", name])
        for key, value in config_kwargs.items():
            if key == "alpha":
                if "alpha_str" in config_kwargs:
                    writer.writerow(["alpha", config_kwargs["alpha_str"]])
                else:
                    writer.writerow([key, value])
            elif key == "eps":
                if "eps_str" in config_kwargs:
                    writer.writerow(["eps", config_kwargs["eps_str"]])
                else:
                    writer.writerow([key, value])
            elif key.endswith("_str"):
                continue
            else:
                writer.writerow([key, value])
        writer.writerow(["seed", seed])
        writer.writerow(["num_prompts", num_prompts])
    logging.info(f"Saved parameters CSV to {parameters_csv_path}")


# --- Load model and processor ---


def load_model_and_processor(model_id):
    if model_id == "llava-hf/llava-1.5-7b-hf":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_id)
    elif model_id == "google/gemma-3-4b-it":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    else:
        raise ValueError(f"Model {model_id} not supported.")
    return model, processor
