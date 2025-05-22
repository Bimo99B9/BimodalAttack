#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
import requests
from PIL import Image

from utils.experiments_utils import load_model_and_processor


def read_params(exp):
    p = os.path.join(exp, "parameters.csv")
    return pd.read_csv(p).set_index("Parameter")["Value"].to_dict()


def best_suffix(exp, run):
    with open(os.path.join(exp, "best_strings.txt"), encoding="utf-8") as f:
        for ln in f:
            if ln.startswith(f"Run {run}:"):
                return ln.split(":", 1)[1].strip()
    raise ValueError("run not found")


def best_iter(exp, run):
    df = pd.read_csv(os.path.join(exp, "losses.csv"))
    col = f"Run {run}"
    return int(df.loc[df[col].idxmin(), "Iteration"])


def goal_text(exp, run):
    custom = os.path.join(exp, "prompts.csv")
    if os.path.exists(custom):
        df = pd.read_csv(custom)
        return df.loc[df["Run"] == run, "goal"].values[0]
    beh = pd.read_csv("data/advbench/harmful_behaviors.csv")
    return beh.iloc[run - 1]["goal"]


def model_id(name):
    return {
        "llava": "llava-hf/llava-1.5-7b-hf",
        "llava-rc": "llava-rc",
        "gemma": "google/gemma-3-4b-it",
    }[name.lower()]


def default_image():
    """Return the default COCO image as a PIL.Image."""
    return Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    ).convert("RGB")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate a response for the given experiment/run. "
            "By default the *optimized* suffix and the run-specific adversarial image "
            "are used. "
            "Pass --simple to ignore the optimized suffix and use the default image "
            "(or no image if PGD=False)."
        )
    )
    ap.add_argument("exp", help="Experiment name (folder inside experiments/)")
    ap.add_argument("--run", type=int, default=1, help="Run number (default: 1)")
    ap.add_argument(
        "--simple",
        action="store_true",
        help=(
            "Ignore the optimized suffix and, if PGD=True, use the default COCO "
            "image instead of the adversarial image."
        ),
    )
    args = ap.parse_args()

    exp_dir = os.path.join("experiments", args.exp)
    prm = read_params(exp_dir)
    pgd = prm.get("pgd_attack", "True").lower() == "true"
    model, proc = load_model_and_processor(model_id(prm.get("model", "llava")))

    if args.simple:
        prompt = goal_text(exp_dir, args.run)
    else:
        suf = best_suffix(exp_dir, args.run)
        prompt = f"{goal_text(exp_dir, args.run)} {suf}"

    if pgd:
        if args.simple:
            img = default_image()
        else:
            img = Image.open(
                os.path.join(
                    exp_dir, f"images_{args.run}", f"{best_iter(exp_dir, args.run)}.png"
                )
            ).convert("RGB")

        chat = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            }
        ]
        tpl = proc.apply_chat_template(chat, add_generation_prompt=True)
        inputs = proc(images=img, text=tpl, return_tensors="pt").to(
            "cuda", torch.float16
        )
    else:
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        tpl = proc.apply_chat_template(chat, add_generation_prompt=True)
        inputs = proc(text=tpl, return_tensors="pt").to("cuda", torch.float16)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=250, do_sample=True)[0]
        print(proc.decode(out, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
