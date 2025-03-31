#!/usr/bin/env python
from itertools import zip_longest
import argparse
import requests
import nanogcg
import torch
from nanogcg import GCGConfig
import time
import gc
import logging
import random
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from utils.experiments_utils import (
    load_advbench_dataset,
    get_experiment_folder,
    get_images_folder,
    write_parameters_csv,
)

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

os.makedirs("experiments", exist_ok=True)

EXPERIMENT_SEED = 1
USE_ALL_PROMPTS = False
NUM_PROMPTS = 5
ADV_BENCH_FILE = "data/advbench/harmful_behaviors.csv"

# Load and optionally slice the dataset
advbench_pairs = load_advbench_dataset(ADV_BENCH_FILE)
if not USE_ALL_PROMPTS:
    advbench_pairs = advbench_pairs[:NUM_PROMPTS]


def set_global_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# Choose model ID. (To use Gemma 3, set MODEL_ID to "google/gemma-3-4b-it")
# MODEL_ID = "llava-hf/llava-1.5-7b-hf"
# For testing Gemma 3 uncomment the following line:
MODEL_ID = "google/gemma-3-4b-it"
model, processor = load_model_and_processor(MODEL_ID)

if MODEL_ID == "llava-hf/llava-1.5-7b-hf":
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((336, 336)),
            T.ToTensor(),
        ]
    )
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    tokenizer = processor.tokenizer
elif MODEL_ID == "google/gemma-3-4b-it":
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((896, 896), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((896, 896)),
            T.ToTensor(),
        ]
    )
    normalize = T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    tokenizer = processor.tokenizer
else:
    raise ValueError(f"Model {MODEL_ID} not supported.")

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")
image = transform(raw_image).unsqueeze(0).to(model.device)


# --- Run experiments across multiple (goal, target) pairs ---
def run_experiment(name, config_kwargs, advbench_pairs):
    experiment_folder = get_experiment_folder()
    logging.info(f"Experiment folder created: {experiment_folder}")
    torch.cuda.empty_cache()
    gc.collect()
    set_global_seed(EXPERIMENT_SEED)

    # Aggregated results lists
    all_losses = []  # list of lists, one per run
    all_best_losses = []  # best loss per run
    all_best_iters = []  # best iteration per run
    all_best_strings = []  # best string per run
    all_gradient_times = []  # list per run
    all_sampling_times = []
    all_pgd_times = []
    all_loss_times = []
    all_total_times = []
    # Details per run: (adversarial_suffixes, model_outputs)
    all_details = []

    # Loop over each (goal, target) pair
    for idx, (goal, target_text) in enumerate(advbench_pairs):
        pair_number = idx + 1  # 1-indexed pair number
        images_folder = get_images_folder(experiment_folder, pair_number)
        # Build a config for the current run using a unique images folder.
        config = GCGConfig(
            **{k: v for k, v in config_kwargs.items() if not k.endswith("_str")},
            seed=EXPERIMENT_SEED,
            verbosity="DEBUG",
            experiment_folder=experiment_folder,
            images_folder=images_folder,
        )
        logging.info(
            f"--- Running prompt-target pair {pair_number}/{len(advbench_pairs)} ---"
        )
        messages = [{"role": "user", "content": goal}]
        target = target_text
        try:
            start_time = time.time()
            result = nanogcg.run(
                model,
                tokenizer,
                processor,
                messages,
                goal,
                target,
                image,
                config,
                normalize=normalize,
            )
            run_time = time.time() - start_time
            run_loss = result.best_loss
            run_losses = result.losses
        except Exception as e:
            logging.exception(
                f"Error during experiment run {pair_number} with seed {EXPERIMENT_SEED}:"
            )
            from nanogcg import GCGResult

            result = GCGResult(
                best_loss=float("nan"),
                best_string="",
                losses=[],
                strings=[],
                adversarial_suffixes=[],
                model_outputs=[],
                gradient_times=[],
                sampling_times=[],
                pgd_times=[],
                loss_times=[],
                total_times=[],
            )
            run_loss = float("nan")
            run_time = 0
            run_losses = []
        logging.info(
            f"Run {pair_number} (Seed={EXPERIMENT_SEED}) -> Loss={run_loss:.4f}, Time={run_time:.2f}s"
        )

        all_losses.append(run_losses)
        all_best_losses.append(run_loss)
        if run_losses:
            best_iter = run_losses.index(min(run_losses))
        else:
            best_iter = -1
        all_best_iters.append(best_iter)
        all_best_strings.append(result.best_string)
        all_gradient_times.append(result.gradient_times)
        all_sampling_times.append(result.sampling_times)
        all_pgd_times.append(result.pgd_times)
        all_loss_times.append(result.loss_times)
        all_total_times.append(result.total_times)
        all_details.append((result.adversarial_suffixes, result.model_outputs))

    # --- Write aggregated CSV files ---
    losses_csv_path = os.path.join(experiment_folder, "losses.csv")
    with open(losses_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Iteration"] + [f"Run {i+1}" for i in range(len(all_losses))]
        writer.writerow(header)
        num_iterations = max((len(loss_list) for loss_list in all_losses), default=0)
        for i in range(num_iterations):
            row = [i]
            for loss_list in all_losses:
                row.append(loss_list[i] if i < len(loss_list) else "")
            writer.writerow(row)
    logging.info(f"Saved aggregated losses CSV to {losses_csv_path}")

    details_csv_path = os.path.join(experiment_folder, "details.csv")
    with open(details_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Iteration"]
        for i in range(len(all_details)):
            header += [f"Run {i+1} Suffix", f"Run {i+1} Output"]
        writer.writerow(header)
        num_iterations = max((len(details[0]) for details in all_details), default=0)
        for i in range(num_iterations):
            row = [i]
            for adv_suffixes, model_outputs in all_details:
                suffix = adv_suffixes[i] if i < len(adv_suffixes) else ""
                output = model_outputs[i] if i < len(model_outputs) else ""
                row += [suffix, output]
            writer.writerow(row)
    logging.info(f"Saved aggregated details CSV to {details_csv_path}")

    times_csv_path = os.path.join(experiment_folder, "times.csv")
    with open(times_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Iteration"]
        for i in range(len(all_total_times)):
            header += [
                f"Run {i+1} Gradient Time",
                f"Run {i+1} Sampling Time",
                f"Run {i+1} PGD Time",
                f"Run {i+1} Loss Time",
                f"Run {i+1} Total Time",
            ]
        writer.writerow(header)
        num_iterations = max((len(t) for t in all_total_times), default=0)
        for i in range(num_iterations):
            row = [i]
            for grad_times, sample_times, pgd_times, loss_times, tot_times in zip(
                all_gradient_times,
                all_sampling_times,
                all_pgd_times,
                all_loss_times,
                all_total_times,
            ):
                row.append(grad_times[i] if i < len(grad_times) else "")
                row.append(sample_times[i] if i < len(sample_times) else "")
                row.append(pgd_times[i] if i < len(pgd_times) else "")
                row.append(loss_times[i] if i < len(loss_times) else "")
                row.append(tot_times[i] if i < len(tot_times) else "")
            writer.writerow(row)
    logging.info(f"Saved aggregated times CSV to {times_csv_path}")

    write_parameters_csv(experiment_folder, config_kwargs, EXPERIMENT_SEED, name, NUM_PROMPTS)

    best_strings_path = os.path.join(experiment_folder, "best_strings.txt")
    with open(best_strings_path, "w") as f:
        for i, s in enumerate(all_best_strings):
            f.write(f"Run {i+1}: {s}\n")
    logging.info(f"Saved best strings to {best_strings_path}")

    summary_csv_path = os.path.join(experiment_folder, "summary.csv")
    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        avg_best_loss = np.mean(all_best_losses) if all_best_losses else float("nan")
        std_best_loss = np.std(all_best_losses) if all_best_losses else float("nan")
        writer.writerow(["Average Best Loss", avg_best_loss])
        writer.writerow(["Std Best Loss", std_best_loss])

        def compute_avg_and_std(time_lists):
            means = [np.mean(t) if len(t) > 0 else float("nan") for t in time_lists]
            return np.mean(means), np.std(means)

        avg_grad, std_grad = compute_avg_and_std(all_gradient_times)
        avg_sample, std_sample = compute_avg_and_std(all_sampling_times)
        avg_pgd, std_pgd = compute_avg_and_std(all_pgd_times)
        avg_loss, std_loss = compute_avg_and_std(all_loss_times)
        avg_total, std_total = compute_avg_and_std(all_total_times)

        writer.writerow(["Average Gradient Time", avg_grad])
        writer.writerow(["Std Gradient Time", std_grad])
        writer.writerow(["Average Sampling Time", avg_sample])
        writer.writerow(["Std Sampling Time", std_sample])
        writer.writerow(["Average PGD Time", avg_pgd])
        writer.writerow(["Std PGD Time", std_pgd])
        writer.writerow(["Average Loss Time", avg_loss])
        writer.writerow(["Std Loss Time", std_loss])
        writer.writerow(["Average Total Time", avg_total])
        writer.writerow(["Std Total Time", std_total])
    logging.info(f"Saved aggregated summary CSV to {summary_csv_path}")

    plt.figure(figsize=(10, 6), dpi=200)
    for i, loss_list in enumerate(all_losses):
        plt.plot(loss_list, linestyle="-", linewidth=1, label=f"Run {i+1}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{name} (Aggregated)")
    ax = plt.gca()

    config_text = "\n".join(
        f"{k}: {v}" for k, v in config_kwargs.items() if not k.endswith("_str")
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)

    ax.text(
        0.98,
        0.98,
        config_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plot_filename = os.path.join(experiment_folder, "losses_aggregated.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved aggregated loss plot to {plot_filename}")


def fraction_type(s):
    try:
        if "/" in s:
            num, denom = s.split("/")
            return float(num) / float(denom)
        return float(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid fraction value: {s}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an experiment with multiple prompts and targets from advbench."
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps")
    parser.add_argument("--search_width", type=int, required=True, help="Search width")
    parser.add_argument(
        "--dynamic_search",
        type=str2bool,
        required=True,
        help="Whether dynamic search is enabled",
    )
    parser.add_argument(
        "--min_search_width", type=int, required=True, help="Minimum search width"
    )
    parser.add_argument(
        "--pgd_attack",
        type=str2bool,
        required=True,
        help="Whether PGD attack is enabled",
    )
    parser.add_argument(
        "--gcg_attack",
        type=str2bool,
        required=True,
        help="Whether GCG attack is enabled",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        required=True,
        help="Alpha value as a fraction (e.g. '2/255')",
    )
    parser.add_argument(
        "--eps",
        type=str,
        required=True,
        help="Epsilon value as a fraction (e.g. '64/255')",
    )
    parser.add_argument(
        "--debug_output", type=str2bool, required=True, help="Debug output flag"
    )
    parser.add_argument(
        "--joint_eval", type=str2bool, required=True, help="Joint evaluation flag"
    )
    parser.add_argument(
        "--pgd_after_gcg", type=str2bool, required=True, help="PGD after GCG flag"
    )
    args = parser.parse_args()

    alpha_float = fraction_type(args.alpha)
    eps_float = fraction_type(args.eps)

    config_kwargs = {
        "num_steps": args.num_steps,
        "search_width": args.search_width,
        "dynamic_search": args.dynamic_search,
        "min_search_width": args.min_search_width,
        "pgd_attack": args.pgd_attack,
        "gcg_attack": args.gcg_attack,
        "alpha": alpha_float,
        "eps": eps_float,
        "debug_output": args.debug_output,
        "alpha_str": args.alpha,
        "eps_str": args.eps,
        "joint_eval": args.joint_eval,
        "pgd_after_gcg": args.pgd_after_gcg,
    }
    run_experiment(args.name, config_kwargs, advbench_pairs)
