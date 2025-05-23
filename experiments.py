# experiments.py
import argparse
import csv
import gc
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from bimodalattack import BimodalAttackConfig
import bimodalattack
from transformers import CLIPImageProcessor
from utils.experiments_utils import (
    load_advbench_dataset,
    get_experiment_folder,
    get_images_folder,
    write_parameters_csv,
    load_model_and_processor,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXPERIMENT_SEED = 1
USE_ALL_PROMPTS = False
NUM_PROMPTS = 1
ADV_BENCH_FILE = "data/advbench/harmful_behaviors.csv"

os.makedirs("experiments", exist_ok=True)

# --------------------------------------------------------------------------- #
# default AdvBench pairs (may be replaced by a custom one later)
advbench_pairs = load_advbench_dataset(ADV_BENCH_FILE)
if not USE_ALL_PROMPTS:
    advbench_pairs = advbench_pairs[:NUM_PROMPTS]
# --------------------------------------------------------------------------- #


def set_global_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(name, config_kwargs, advbench_pairs):
    experiment_folder = get_experiment_folder()
    logging.info(f"Experiment folder created: {experiment_folder}")
    torch.cuda.empty_cache()
    gc.collect()
    set_global_seed(EXPERIMENT_SEED)

    # NEW -- save the exact (goal,target) pairs used for this run ─────────────
    with open(
        os.path.join(experiment_folder, "prompts.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["Run", "goal", "target"])
        for i, (g, t) in enumerate(advbench_pairs, start=1):
            w.writerow([i, g, t])
    # -----------------------------------------------------------------------

    all_losses, all_best_losses, all_best_iters, all_best_strings = [], [], [], []
    (
        all_gradient_times,
        all_sampling_times,
        all_pgd_times,
        all_loss_times,
        all_total_times,
    ) = ([], [], [], [], [])
    all_details = []

    for idx, (goal, target_text) in enumerate(advbench_pairs, start=1):
        images_folder = get_images_folder(experiment_folder, idx)
        config = BimodalAttackConfig(
            **{
                k: v
                for k, v in config_kwargs.items()
                if not k.endswith("_str") and k != "model"
            },
            seed=EXPERIMENT_SEED,
            verbosity="DEBUG",
            experiment_folder=experiment_folder,
            images_folder=images_folder,
        )
        logging.info(f"--- Running prompt-target pair {idx}/{len(advbench_pairs)} ---")
        messages = [{"role": "user", "content": goal}]

        try:
            start_time = time.time()
            result = bimodalattack.run(
                model,
                tokenizer,
                processor,
                messages,
                goal,
                target_text,
                image,
                config,
                normalize=normalize,
            )
            run_time = time.time() - start_time
            run_loss = result.best_loss
            run_losses = result.losses
        except Exception as e:
            from bimodalattack import BimodalAttackResult

            result = BimodalAttackResult(
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
            run_time, run_loss, run_losses = 0, float("nan"), []

            logging.error(
                f"Error during attack for prompt {idx}/{len(advbench_pairs)}: {goal} -> {target_text}"
            )
            logging.error(f"Exception: {e}")

        logging.info(
            f"Run {idx} (Seed={EXPERIMENT_SEED}) -> Loss={run_loss:.4f}, Time={run_time:.2f}s"
        )

        all_losses.append(run_losses)
        all_best_losses.append(run_loss)
        all_best_iters.append(run_losses.index(min(run_losses)) if run_losses else -1)
        all_best_strings.append(result.best_string)
        all_gradient_times.append(result.gradient_times)
        all_sampling_times.append(result.sampling_times)
        all_pgd_times.append(result.pgd_times)
        all_loss_times.append(result.loss_times)
        all_total_times.append(result.total_times)
        all_details.append((result.adversarial_suffixes, result.model_outputs))

    def write_csv(path, header, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    # losses.csv
    max_iters = max((len(l) for l in all_losses), default=0)
    loss_rows = [
        [i] + [(l[i] if i < len(l) else "") for l in all_losses]
        for i in range(max_iters)
    ]
    write_csv(
        os.path.join(experiment_folder, "losses.csv"),
        ["Iteration"] + [f"Run {i+1}" for i in range(len(all_losses))],
        loss_rows,
    )
    logging.info("Saved aggregated losses CSV")

    # details.csv
    max_iters = max((len(d[0]) for d in all_details), default=0)
    detail_rows = []
    for i in range(max_iters):
        row = [i]
        for adv, out in all_details:
            row += [adv[i] if i < len(adv) else "", out[i] if i < len(out) else ""]
        detail_rows.append(row)
    header = ["Iteration"] + sum(
        [[f"Run {i+1} Suffix", f"Run {i+1} Output"] for i in range(len(all_details))],
        [],
    )
    write_csv(os.path.join(experiment_folder, "details.csv"), header, detail_rows)
    logging.info("Saved aggregated details CSV")

    # times.csv
    max_iters = max((len(t) for t in all_total_times), default=0)
    time_rows = []
    for i in range(max_iters):
        row = [i]
        for gt, st, pt, lt, tt in zip(
            all_gradient_times,
            all_sampling_times,
            all_pgd_times,
            all_loss_times,
            all_total_times,
        ):
            row += [
                (gt[i] if i < len(gt) else ""),
                (st[i] if i < len(st) else ""),
                (pt[i] if i < len(pt) else ""),
                (lt[i] if i < len(lt) else ""),
                (tt[i] if i < len(tt) else ""),
            ]
        time_rows.append(row)
    header = ["Iteration"] + sum(
        [
            [
                f"Run {i+1} {t}"
                for t in [
                    "Gradient Time",
                    "Sampling Time",
                    "PGD Time",
                    "Loss Time",
                    "Total Time",
                ]
            ]
            for i in range(len(all_total_times))
        ],
        [],
    )
    write_csv(os.path.join(experiment_folder, "times.csv"), header, time_rows)
    logging.info("Saved aggregated times CSV")

    # ---------- changed: pass real #prompts instead of constant ------------
    write_parameters_csv(
        experiment_folder, config_kwargs, EXPERIMENT_SEED, name, len(advbench_pairs)
    )
    # -----------------------------------------------------------------------

    with open(os.path.join(experiment_folder, "best_strings.txt"), "w") as f:
        for i, s in enumerate(all_best_strings, start=1):
            f.write(f"Run {i}: {s}\n")
    logging.info("Saved best strings")

    avg_best = np.mean(all_best_losses) if all_best_losses else float("nan")
    std_best = np.std(all_best_losses) if all_best_losses else float("nan")
    summary = [["Average Best Loss", avg_best], ["Std Best Loss", std_best]]

    def comp(tlists):
        means = [np.mean(t) if t else float("nan") for t in tlists]
        return np.mean(means), np.std(means)

    for label, times in zip(
        ["Gradient", "Sampling", "PGD", "Loss", "Total"],
        [
            all_gradient_times,
            all_sampling_times,
            all_pgd_times,
            all_loss_times,
            all_total_times,
        ],
    ):
        avg, std = comp(times)
        summary += [[f"Average {label} Time", avg], [f"Std {label} Time", std]]
    write_csv(
        os.path.join(experiment_folder, "summary.csv"), ["Metric", "Value"], summary
    )
    logging.info("Saved aggregated summary CSV")

    plt.figure(figsize=(10, 6), dpi=200)
    for i, losses in enumerate(all_losses, start=1):
        plt.plot(losses, linestyle="-", linewidth=1, label=f"Run {i}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(name)
    config_text = "\n".join(
        f"{k}: {v}" for k, v in config_kwargs.items() if not k.endswith("_str")
    )
    plt.gca().text(
        0.98,
        0.98,
        config_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    plot_path = os.path.join(experiment_folder, "losses_aggregated.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved aggregated loss plot")


def fraction_type(s):
    if "/" in s:
        num, denom = s.split("/")
        return float(num) / float(denom)
    return float(s)


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--num_steps", type=int, required=True)
    p.add_argument("--search_width", type=int, required=True)
    p.add_argument(
        "--dynamic_search",
        type=lambda x: x.lower() in ("y", "yes", "true"),
        required=True,
    )
    p.add_argument("--min_search_width", type=int, required=True)
    p.add_argument(
        "--pgd_attack", type=lambda x: x.lower() in ("y", "yes", "true"), required=True
    )
    p.add_argument(
        "--gcg_attack", type=lambda x: x.lower() in ("y", "yes", "true"), required=True
    )
    p.add_argument("--alpha", type=str, required=True)
    p.add_argument("--eps", type=str, required=True)
    p.add_argument(
        "--debug_output",
        type=lambda x: x.lower() in ("y", "yes", "true"),
        required=True,
    )
    p.add_argument(
        "--joint_eval", type=lambda x: x.lower() in ("y", "yes", "true"), required=True
    )
    p.add_argument(
        "--model",
        choices=["gemma", "llava", "llava-rc"],
        required=True,
        help="Choose 'gemma', 'llava', or 'llava-rc' (LLaVA with robust CLIP encoder)",
    )

    p.add_argument("--goal", type=str, help="Custom goal prompt")
    p.add_argument("--target", type=str, help="Custom target text (required if --goal)")
    args = p.parse_args()

    if args.goal:
        if not args.target:
            raise ValueError("--target is required when --goal is provided")
        adv_pairs = [(args.goal, args.target)]
    else:
        adv_pairs = advbench_pairs

    # parse numeric fractions
    def fraction_type(s):
        if "/" in s:
            n, d = s.split("/")
            return float(n) / float(d)
        return float(s)

    alpha = fraction_type(args.alpha)
    eps = fraction_type(args.eps)

    # pick HF model id
    if args.model == "llava":
        MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    elif args.model == "llava-rc":
        MODEL_ID = "llava-rc"
    else:
        MODEL_ID = "google/gemma-3-4b-it"

    # load
    model, processor = load_model_and_processor(MODEL_ID)
    tokenizer = processor.tokenizer

    # set up transforms & normalize
    if args.model == "gemma":
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize((896, 896), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop((896, 896)),
                T.ToTensor(),
            ]
        )
        normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    elif args.model == "llava":
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize(336, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop((336, 336)),
                T.ToTensor(),
            ]
        )
        normalize = T.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        )

    else:  # llava-rc
        clip_proc: CLIPImageProcessor = processor.image_processor
        h = clip_proc.size["height"]
        w = clip_proc.size["width"]
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop((h, w)),
                T.ToTensor(),
            ]
        )
        normalize = T.Normalize(
            clip_proc.image_mean,
            clip_proc.image_std,
        )

    # load one test image for the attack pipeline
    raw = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    ).convert("RGB")
    image = transform(raw).unsqueeze(0).to(model.device)

    config_kwargs = {
        "num_steps": args.num_steps,
        "search_width": args.search_width,
        "dynamic_search": args.dynamic_search,
        "min_search_width": args.min_search_width,
        "pgd_attack": args.pgd_attack,
        "gcg_attack": args.gcg_attack,
        "alpha": alpha,
        "eps": eps,
        "debug_output": args.debug_output,
        "alpha_str": args.alpha,
        "eps_str": args.eps,
        "joint_eval": args.joint_eval,
        "model": args.model,
    }

    run_experiment(args.name, config_kwargs, adv_pairs)
