import copy
import gc
import logging
import queue
import threading
import time
from contextlib import contextmanager

from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks

import requests
from PIL import Image
import torchvision.transforms.functional as F
import os

# ---------------------------
# Initialize LlamaGuard moderation model
# ---------------------------
llama_guard_model_id = "meta-llama/Llama-Guard-3-8B"
llama_guard_device = "cuda"
llama_guard_dtype = torch.bfloat16

llama_guard_tokenizer = transformers.AutoTokenizer.from_pretrained(llama_guard_model_id)
llama_guard_model = transformers.AutoModelForCausalLM.from_pretrained(
    llama_guard_model_id, torch_dtype=llama_guard_dtype, device_map=llama_guard_device
)


def llama_guard_moderate(chat):
    """
    Expects a conversation as a list of dicts (e.g. [{'role': 'user', 'content': prompt},
    {'role': 'assistant', 'content': output}]). Returns the LlamaGuard evaluation string.
    """
    input_ids = llama_guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(
        llama_guard_device
    )
    output = llama_guard_model.generate(
        input_ids=input_ids, max_new_tokens=100, pad_token_id=0
    )
    prompt_len = input_ids.shape[-1]
    return llama_guard_tokenizer.decode(
        output[0][prompt_len:], skip_special_tokens=True
    )
    
def parse_conversation(prompt, output):
    """
    We want this format for LlamaGuard:
    
    moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
    ])
    
    And we provide "prompt", which is the content of the user message, and "output", which is the content of the assistant message.
    """
    result = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output.strip()},
    ]
    
    logger.info(f"Parsed conversation for LlamaGuard: {result}")
    return result 


# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


# ---------------------------
# Dataclass definitions
# ---------------------------
@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    dynamic_search: bool = False
    min_search_width: int = 32
    alpha: float = 0.01
    eps: float = 0.1
    pgd_attack: bool = False
    gcg_attack: bool = True
    debug_output: bool = False
    joint_eval: bool = False
    experiment_folder: str = "experiments/missing_folder"
    images_folder: str = "experiments/missing_folder/images"
    pgd_after_gcg: bool = False


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    adversarial_suffixes: List[str]
    model_outputs: List[str]
    gradient_times: List[float]
    sampling_times: List[float]
    loss_times: List[float]
    pgd_times: List[float]
    total_times: List[float] = None
    llama_guard_unsafe: bool = False


# ---------------------------
# AttackBuffer definition
# ---------------------------
class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss} | string: {optim_str}"
        logger.info(message)


# ---------------------------
# Candidate sampling helper function
# ---------------------------
def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """
    Returns search_width combinations of token ids based on the token gradient.
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    # Randomly choose positions to replace (n_replace per candidate)
    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[
        ..., :n_replace
    ]  # shape: (search_width, n_replace)

    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """
    Filters out sequences of token ids that change after retokenization.
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting filter_ids=False or trying a different optim_str_init"
        )

    return torch.stack(filtered_ids)


# ---------------------------
# Main GCG class definition
# ---------------------------
class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        processor,
        config: GCGConfig,
        normalize=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.normalize = normalize

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type for faster optimization."
            )
        if model.device == torch.device("cpu"):
            logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        # Set chat template based on model type.
        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            if config.pgd_attack:
                logger.warning(
                    "Tokenizer does not have a chat template. Using custom chat template for GCG+PGD attack."
                )
                custom_template = "USER: <image>\n{{ messages[0]['content'][0]['text'] }} \nASSISTANT: "
                tokenizer.chat_template = custom_template
                self.processor.chat_template = custom_template
            else:
                logger.warning(
                    "Tokenizer does not have a chat template. Using custom chat template for GCG only attack."
                )
                custom_template = (
                    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                )
                tokenizer.chat_template = custom_template
                self.processor.chat_template = custom_template

    def run(
        self, messages: Union[str, List[dict]], goal: str, target: str, image: torch.Tensor = None
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        self.initial_prompt = goal

        images_folder = config.images_folder
        os.makedirs(images_folder, exist_ok=True)

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
        logger.info(f"Messages 0: {messages}")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
            logger.info(f"Messages 1: {messages}")

        if (
            isinstance(messages[-1]["content"], str)
            and "{optim_str}" not in messages[-1]["content"]
        ):
            messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"
            logger.info(f"Messages 2: {messages}")

        if config.pgd_attack:
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [
                    {"type": "text", "text": messages[-1]["content"]},
                    {"type": "image"},
                ]
                logger.info(f"Messages 3: {messages}")
            elif isinstance(messages[-1]["content"], list):
                if not any(
                    item.get("type") == "image" for item in messages[-1]["content"]
                ):
                    messages[-1]["content"].append({"type": "image"})
        logger.info(f"Messages 4: {messages}")

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        logger.info(f"Prompt after applying chat template: {prompt}")

        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt.replace(tokenizer.bos_token, "")
        logger.info(f"Prompt after removing BOS token: {prompt}")

        if config.pgd_attack:
            if self.processor.__class__.__name__ == "Gemma3Processor":
                # Split the prompt on the {optim_str} placeholder.
                before_str, after_temp = prompt.split("{optim_str}", 1)
                # Now split after_temp on "<start_of_image>"
                if "<start_of_image>" in after_temp:
                    # Discard any text before the image token.
                    _, after_temp = after_temp.split("<start_of_image>", 1)
                    # Now split on "<end_of_turn>" to remove trailing generation prompt.
                    if "<end_of_turn>" in after_temp:
                        _, after_str = after_temp.split("<end_of_turn>", 1)
                        after_str = after_str.strip()
                    else:
                        after_str = ""
                else:
                    raise ValueError(
                        "Expected <start_of_image> token in Gemma PGD prompt."
                    )
                before_img_str = before_str.strip()
                before_suffix_str = (
                    ""  # No text between {optim_str} and the image token.
                )
            else:
                if "<start_of_image>" in prompt:
                    before_img_str, after_img_str = prompt.split("<start_of_image>", 1)
                elif "<image>" in prompt:
                    before_img_str, after_img_str = prompt.split("<image>", 1)
                else:
                    raise ValueError("No image token found in prompt for PGD attack")
                before_suffix_str, after_str = after_img_str.split("{optim_str}", 1)

            logger.info(f"Before image str: {before_img_str}")
            logger.info(f"Before suffix str: {before_suffix_str}")
            logger.info(f"After str: {after_str}")
            logger.info(f"Target: {target}")

            before_img_ids = tokenizer(
                before_img_str, padding=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            before_suffix_ids = tokenizer(
                before_suffix_str, padding=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            target_ids = tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
        else:
            before_str, after_str = prompt.split("{optim_str}")
            logger.info(f"Before str: {before_str}")
            logger.info(f"After str: {after_str}")
            logger.info(f"Target: {target}")
            before_ids = tokenizer(before_str, padding=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device, torch.int64)
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            target_ids = tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)

        if config.pgd_attack:
            before_img_embeds, before_suffix_embeds, after_embeds, target_embeds = [
                self.embedding_layer(ids)
                for ids in (before_img_ids, before_suffix_ids, after_ids, target_ids)
            ]
            self.before_img_ids = before_img_ids
            self.before_suffix_ids = before_suffix_ids
            self.after_ids = after_ids
            self.target_ids = target_ids
            self.before_img_embeds = before_img_embeds
            self.before_suffix_embeds = before_suffix_embeds
            self.after_embeds = after_embeds
            self.target_embeds = target_embeds
        else:
            before_embeds, after_embeds, target_embeds = [
                self.embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
            ]
            self.target_ids = target_ids
            self.before_embeds = before_embeds
            self.after_embeds = after_embeds
            self.target_embeds = target_embeds

        ### Attack buffer initialization
        buffer = self.init_buffer(image)
        optim_ids = buffer.get_best_ids()

        # Containers for metrics and outputs.
        losses = []
        optim_strings = []
        adv_suffixes = []
        model_outputs = []
        gradient_times = []
        sampling_times = []
        loss_times = []
        pgd_times = []
        total_times = []

        total_gradient_time = 0.0
        total_sampling_time = 0.0
        total_loss_time = 0.0
        total_pgd_time = 0.0

        llama_guard_found_unsafe = False

        # Initialize variables to track the best candidate (lowest loss).
        best_loss = float("inf")
        best_optim_ids = None
        best_image = None

        if config.pgd_attack:
            logger.warning(f"Using alpha: {config.alpha}, eps: {config.eps}")
            image.requires_grad = True
            image_original = image.clone()

        # Log the overall configuration.
        if config.pgd_attack and config.gcg_attack:
            if config.pgd_after_gcg:
                logger.info(
                    "Running GCG and PGD with PGD after GCG (GRADS -> GCG -> GRADS -> PGD)"
                )
            else:
                logger.info("Running PGD and GCG (GRADS -> PGD -> GRADS -> GCG)")
        elif config.pgd_attack and not config.gcg_attack:
            logger.info("Running only PGD (GRADS -> PGD)")
        elif config.gcg_attack and not config.pgd_attack:
            logger.info("Running only GCG (GRADS -> GCG)")

        for i in tqdm(range(config.num_steps)):
            logger.info(
                f"[Iteration {i}] Starting iteration with config: pgd_attack={config.pgd_attack}, gcg_attack={config.gcg_attack}, pgd_after_gcg={config.pgd_after_gcg}"
            )

            ### Phase A - Initial gradient computation (GRADS)
            start_grad = time.perf_counter()
            if config.pgd_attack:
                optim_ids_onehot_grad, image_grad = self.compute_gradient(
                    optim_ids, image
                )
            else:
                optim_ids_onehot_grad, _ = self.compute_gradient(optim_ids)
            grad_time = time.perf_counter() - start_grad
            gradient_times.append(grad_time)
            total_gradient_time += grad_time
            logger.info(
                f"[Iteration {i}] Phase A (GRADS) completed in {grad_time:.4f}s"
            )

            # Depending on ordering, either perform PGD update now (phases B & C) or skip them.
            if config.pgd_attack and not config.pgd_after_gcg:
                logger.info(f"[Iteration {i}] Running PGD before GCG (Phase B)")
                start_pgd = time.perf_counter()
                image = (
                    (image - config.alpha * config.eps * torch.sign(image_grad))
                    .detach()
                    .requires_grad_()
                )
                image = torch.clamp(
                    image, image_original - config.eps, image_original + config.eps
                )
                image = torch.clamp(image, 0, 1)
                pgd_time = time.perf_counter() - start_pgd
                pgd_times.append(pgd_time)
                total_pgd_time += pgd_time
                logger.info(
                    f"[Iteration {i}] Phase B (PGD update) completed in {pgd_time:.4f}s"
                )

                ### Phase C - Recompute gradient after PGD update.
                if config.gcg_attack:
                    start_grad = time.perf_counter()
                    optim_ids_onehot_grad, image_grad = self.compute_gradient(
                        optim_ids, image
                    )
                    grad_time = time.perf_counter() - start_grad
                    gradient_times.append(grad_time)
                    total_gradient_time += grad_time
                    logger.info(
                        f"[Iteration {i}] Phase C (Recompute GRADS) completed in {grad_time:.4f}s"
                    )
            elif config.pgd_after_gcg:
                logger.info(
                    f"[Iteration {i}] Skipping PGD update (B) and second GRADS (C) due to pgd_after_gcg flag; will perform PGD after GCG"
                )
            else:
                pgd_time = 0.0
                logger.info(
                    f"[Iteration {i}] Skipping PGD update (B) and second GRADS (C) due to no PGD attack"
                )

            ### Phase D - GCG candidate sampling and (partial) loss computation.
            logger.info(f"[Iteration {i}] Running GCG candidate sampling (Phase D)")
            sampled_ids, new_search_width, sampling_time, total_sampling_time = (
                self.candidate_sampling(
                    config,
                    i,
                    optim_ids,
                    optim_ids_onehot_grad,
                    tokenizer,
                    sampling_times,
                    total_sampling_time,
                )
            )
            logger.info(
                f"[Iteration {i}] Sampled {new_search_width} candidates in {sampling_time:.4f}s"
            )

            with torch.no_grad():
                start_loss = time.perf_counter()
                batch_size = (
                    new_search_width if config.batch_size is None else config.batch_size
                )

                if config.pgd_attack:
                    pixel_values = self.normalize(image)
                    image_features = model.get_image_features(
                        pixel_values=pixel_values,
                        # vision_feature_layer=-2,
                        # vision_feature_select_strategy="default",
                    )
                    if config.pgd_after_gcg:
                        if config.joint_eval:
                            candidate_input_embeds = self._build_input_embeds_gcg_pgd(
                                sampled_ids,
                                image_features,
                                search_width=new_search_width,
                                single=True,
                            )

                            loss = find_executable_batch_size(
                                self._compute_candidates_loss_original, batch_size
                            )(candidate_input_embeds)
                            best_loss_before_image = loss.min().item()
                            best_idx = loss.argmin()
                        else:
                            if config.gcg_attack:
                                candidate_input_embeds = self._build_input_embeds_gcg(
                                    sampled_ids,
                                    search_width=new_search_width,
                                    single=True,
                                )

                                loss = find_executable_batch_size(
                                    self._compute_candidates_loss_original, batch_size
                                )(candidate_input_embeds)
                                best_loss_before_image = loss.min().item()
                                best_idx = loss.argmin()
                            else:
                                best_loss_before_image = 0.0
                                best_idx = 0
                        logger.info(
                            f"[Iteration {i}] [GCG] Selected candidate index {best_idx} (pre-PGD update), loss before image evaluation: {best_loss_before_image:.4f}"
                        )
                        chosen_candidate = sampled_ids[best_idx].unsqueeze(0)
                    else:
                        if config.joint_eval:
                            candidate_input_embeds = self._build_input_embeds_pgd(
                                sampled_ids,
                                image_features,
                                search_width=new_search_width,
                                single=True,
                            )

                            loss = find_executable_batch_size(
                                self._compute_candidates_loss_original, batch_size
                            )(candidate_input_embeds)
                            best_loss_before_image = loss.min().item()
                            best_idx = loss.argmin()
                        else:
                            if config.gcg_attack:
                                candidate_input_embeds = self._build_input_embeds_gcg(
                                    sampled_ids,
                                    search_width=new_search_width,
                                    single=True,
                                )

                                loss = find_executable_batch_size(
                                    self._compute_candidates_loss_original, batch_size
                                )(candidate_input_embeds)
                                best_loss_before_image = loss.min().item()
                                best_idx = loss.argmin()
                            else:
                                best_loss_before_image = 0.0
                                best_idx = 0
                        logger.info(
                            f"[Iteration {i}] Best loss before evaluation with image: {best_loss_before_image:.4f}"
                        )
                        full_input_embeds = self._build_input_embeds_gcg_pgd(
                            sampled_ids[best_idx].unsqueeze(0), image_features
                        )

                        full_loss = self._compute_candidates_loss_original(
                            1, full_input_embeds
                        )
                        current_loss = full_loss.item()
                        optim_ids = sampled_ids[best_idx].unsqueeze(0)

                        losses.append(current_loss)
                        optim_str = tokenizer.batch_decode(optim_ids)[0]
                        optim_strings.append(optim_str)
                        if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                            buffer.add(current_loss, optim_ids)
                        # Update best candidate if this loss is the lowest so far.
                        if current_loss < best_loss:
                            best_loss = current_loss
                            best_optim_ids = optim_ids.clone()
                            best_image = image.clone()

                        logger.info(
                            f"[Iteration {i}] Final loss with image and suffix: {current_loss:.4f}"
                        )
                else:  # GCG without PGD.
                    candidate_input_embeds = self._build_input_embeds_gcg(
                        sampled_ids, search_width=new_search_width, no_joint_eval=True
                    )
                    loss = find_executable_batch_size(
                        self._compute_candidates_loss_original, batch_size
                    )(candidate_input_embeds)
                    current_loss = loss.min().item()
                    best_idx = loss.argmin()
                    optim_ids = sampled_ids[best_idx].unsqueeze(0)

                    losses.append(current_loss)
                    optim_str = tokenizer.batch_decode(optim_ids)[0]
                    optim_strings.append(optim_str)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)
                    # Update best candidate (no image available in non-PGD mode).
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_optim_ids = optim_ids.clone()

                    logger.info(
                        f"[Iteration {i}] Final loss with only suffix: {current_loss:.4f}"
                    )

                loss_time = time.perf_counter() - start_loss
                loss_times.append(loss_time)
                total_loss_time += loss_time
                logger.info(
                    f"[Iteration {i}] Loss computation completed in {loss_time:.4f}s"
                )

            # End Phase D

            # If PGD is to run after GCG, do it here.
            if config.pgd_after_gcg and config.pgd_attack:
                logger.info(
                    f"[Iteration {i}] Running PGD after GCG: computing gradient (Phase E)"
                )
                start_grad = time.perf_counter()
                optim_ids_onehot_grad, image_grad = self.compute_gradient(
                    optim_ids, image
                )
                grad_time = time.perf_counter() - start_grad
                gradient_times.append(grad_time)
                total_gradient_time += grad_time
                logger.info(
                    f"[Iteration {i}] Phase E (GRADS after GCG) completed in {grad_time:.4f}s"
                )

                logger.info(
                    f"[Iteration {i}] Running PGD after GCG: PGD update (Phase F)"
                )
                start_pgd = time.perf_counter()
                image = (
                    (image - config.alpha * config.eps * torch.sign(image_grad))
                    .detach()
                    .requires_grad_()
                )
                image = torch.clamp(
                    image, image_original - config.eps, image_original + config.eps
                )
                image = torch.clamp(image, 0, 1)
                pgd_time = time.perf_counter() - start_pgd
                pgd_times.append(pgd_time)
                total_pgd_time += pgd_time
                logger.info(
                    f"[Iteration {i}] Phase F (PGD update after GCG) completed in {pgd_time:.4f}s"
                )

                with torch.no_grad():
                    start_loss = time.perf_counter()
                    pixel_values = self.normalize(image)
                    image_features = model.get_image_features(
                        pixel_values=pixel_values,
                    )
                    full_input_embeds = self._build_input_embeds_gcg_pgd(
                        chosen_candidate, image_features
                    )

                    full_loss = self._compute_candidates_loss_original(
                        1, full_input_embeds
                    )
                    current_loss = full_loss.item()
                    optim_ids = chosen_candidate
                    logger.info(
                        f"[Iteration {i}] Final loss after [PGD after GCG]: {current_loss:.4f}"
                    )
                    loss_time = time.perf_counter() - start_loss
                    loss_times.append(loss_time)
                    total_loss_time += loss_time

                    losses.append(current_loss)
                    optim_str = tokenizer.batch_decode(optim_ids)[0]
                    optim_strings.append(optim_str)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)
                    # Also update best candidate if needed.
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_optim_ids = optim_ids.clone()
                        best_image = image.clone()

            # Save image and (optionally) generate debug output.
            if config.pgd_attack:
                self._save_image(image, os.path.join(images_folder, f"{i}.png"))
            if (config.debug_output and i % 10 == 0) or (i == config.num_steps - 1):
                with torch.no_grad():
                    if config.pgd_attack:
                        pixel_values = self.normalize(image)
                        image_features = model.get_image_features(
                            pixel_values=pixel_values,
                        )
                        input_embeds = self._build_input_embeds_gcg_pgd(
                            sampled_ids,
                            image_features,
                            search_width=new_search_width,
                            no_target=True,
                        )
                    else:
                        input_embeds = self._build_input_embeds_gcg(
                            sampled_ids, search_width=new_search_width, no_target=True
                        )

                    generated_ids = model.generate(
                        inputs_embeds=input_embeds, max_new_tokens=120
                    )
                    gen_output = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    logger.info(f"Output generated at iteration {i}: {gen_output}")

                    # ---------------------------
                    # LlamaGuard check in debug output
                    # ---------------------------
                    guard_chat = parse_conversation(self.initial_prompt, gen_output)
                    llama_guard_eval = llama_guard_moderate(guard_chat)
                    logger.info(
                        f"LlamaGuard evaluation at iteration {i}: {llama_guard_eval}"
                    )
                    if "unsafe" in llama_guard_eval.lower():
                        logger.warning(
                            f"LlamaGuard detected unsafe output at iteration {i}."
                        )
                        llama_guard_found_unsafe = True
            else:
                gen_output = ""
            model_outputs.append(gen_output)
            adv_suffixes.append(tokenizer.batch_decode(optim_ids)[0])
            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

            iter_total = grad_time + sampling_time + pgd_time + loss_time
            total_times.append(iter_total)
            logger.info(
                f"[Iteration {i}] Total iteration time: {iter_total:.4f}s "
                f"(Gradient: {grad_time:.4f}s, Sampling: {sampling_time:.4f}s, PGD: {pgd_time:.4f}s, Loss: {loss_time:.4f}s)"
            )

        num_iters = i + 1
        logger.warning(
            f"Average token gradient time: {total_gradient_time / num_iters:.4f}s"
        )
        logger.warning(
            f"Average PGD image update time: {total_pgd_time / num_iters:.4f}s"
        )
        logger.warning(
            f"Average candidate sampling time: {total_sampling_time / num_iters:.4f}s"
        )
        logger.warning(
            f"Average candidate loss computation time: {total_loss_time / num_iters:.4f}s"
        )

        # ---------------------------
        # Final additional inference using best candidate (and image if available)
        # ---------------------------
        with torch.no_grad():
            if config.pgd_attack and best_image is not None:
                pixel_values = self.normalize(best_image)
                image_features = model.get_image_features(pixel_values=pixel_values)
                input_embeds = self._build_input_embeds_gcg_pgd(
                    best_optim_ids,
                    image_features,
                    search_width=1,
                    no_target=True,
                )
            else:
                input_embeds = self._build_input_embeds_gcg(
                    best_optim_ids,
                    search_width=1,
                    no_target=True,
                )
            final_generated_ids = model.generate(
                inputs_embeds=input_embeds, max_new_tokens=120
            )
            best_gen_output = tokenizer.decode(
                final_generated_ids[0], skip_special_tokens=True
            )
            logger.info(f"Final output generated using best candidate: {best_gen_output}")

            # ---------------------------
            # Final LlamaGuard evaluation using best candidate output
            # ---------------------------
            guard_chat = parse_conversation(self.initial_prompt, best_gen_output)
            llama_guard_final_eval = llama_guard_moderate(guard_chat)
            logger.info(f"Final LlamaGuard evaluation using best candidate: {llama_guard_final_eval}")
            final_llama_guard_flag = "unsafe" in llama_guard_final_eval.lower()
            overall_llama_guard_unsafe = llama_guard_found_unsafe or final_llama_guard_flag

        # Optionally, you might want to append the final generated output to model_outputs:
        model_outputs.append(best_gen_output)

        # Use the best candidate (based on lowest loss) for the final result.
        min_loss_index = losses.index(min(losses))
        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            adversarial_suffixes=adv_suffixes,
            model_outputs=model_outputs,
            gradient_times=gradient_times,
            sampling_times=sampling_times,
            loss_times=loss_times,
            pgd_times=pgd_times,
            total_times=total_times,
            llama_guard_unsafe=overall_llama_guard_unsafe,
        )
        return result



    def init_buffer(self, image) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        ### Attack buffer creation and initialization
        buffer = AttackBuffer(config.buffer_size)
        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids
        else:
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        if config.pgd_attack:
            pixel_values = self.normalize(image)
            image_features = model.get_image_features(
                pixel_values=pixel_values,
            )
            init_buffer_embeds = self._build_input_embeds_gcg_pgd(
                init_buffer_ids,
                image_features,
                search_width=true_buffer_size,
                single=True,
            )
        else:
            self._build_input_embeds_gcg(
                init_buffer_ids, search_width=true_buffer_size, no_joint_eval=True
            )

        ## Initial buffer loss computation
        init_buffer_losses = find_executable_batch_size(
            self._compute_candidates_loss_original, true_buffer_size
        )(init_buffer_embeds)

        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        buffer.log_buffer(tokenizer)
        logger.info("Initialized attack buffer.")
        return buffer

    def candidate_sampling(
        self,
        config: GCGConfig,
        i,
        optim_ids,
        optim_ids_onehot_grad,
        tokenizer,
        sampling_times,
        total_sampling_time,
    ):
        ### Candidate sampling
        if config.dynamic_search:
            current_search_width = max(
                config.min_search_width,
                int(config.search_width * (1 - i / config.num_steps)),
            )
            logger.info(
                f"[Iteration {i}] Using dynamic search width: {current_search_width}"
            )
        else:
            current_search_width = config.search_width

        if config.gcg_attack:
            start_sample = time.perf_counter()
            sampled_ids = sample_ids_from_grad(
                optim_ids.squeeze(0),
                optim_ids_onehot_grad.squeeze(0),
                current_search_width,
                config.topk,
                config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )
            if config.filter_ids:
                sampled_ids = filter_ids(sampled_ids, tokenizer)
            new_search_width = sampled_ids.shape[0]
            sampling_time = time.perf_counter() - start_sample
            sampling_times.append(sampling_time)
            total_sampling_time += sampling_time
        else:
            sampled_ids = optim_ids
            new_search_width = 1
            sampling_time = 0.0

        return sampled_ids, new_search_width, sampling_time, total_sampling_time

    def compute_gradient(
        self, optim_ids: torch.Tensor, image: torch.Tensor = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        model = self.model
        embedding_layer = self.embedding_layer

        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        if self.config.gcg_attack:
            optim_ids_onehot.requires_grad_()
        else:
            optim_ids_onehot.requires_grad = False

        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.config.pgd_attack:
            pixel_values = self.normalize(image)
            image_features = model.get_image_features(
                pixel_values=pixel_values,
            )
            input_embeds = torch.cat(
                [
                    self.before_img_embeds,
                    image_features,
                    self.before_suffix_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

        output = model(inputs_embeds=input_embeds)
        logits = output.logits

        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()
        shift_labels = self.target_ids

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        if self.config.pgd_attack:
            if self.config.gcg_attack:
                optim_ids_onehot_grad, image_grad = torch.autograd.grad(
                    loss, (optim_ids_onehot, image)
                )
            else:
                image_grad = torch.autograd.grad(loss, image)[0]
                optim_ids_onehot_grad = None
            return optim_ids_onehot_grad, image_grad
        else:
            if self.config.gcg_attack:
                optim_ids_onehot_grad = torch.autograd.grad(loss, optim_ids_onehot)[0]
            else:
                optim_ids_onehot_grad = None
            return optim_ids_onehot_grad, None

    def _build_input_embeds_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width: int,
        single: bool = False,
    ) -> Tensor:
        if single:
            return torch.cat(
                [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

    def _build_input_embeds_gcg(
        self,
        sampled_ids: Tensor,
        search_width: int,
        single: bool = False,
        no_joint_eval: bool = False,
        no_target: bool = False,
    ) -> Tensor:

        if single:
            return torch.cat(
                [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

        if no_joint_eval:
            return torch.cat(
                [
                    self.before_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

        if no_target:
            return torch.cat(
                [
                    self.before_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

    def _build_input_embeds_gcg_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width=None,
        single: bool = False,
        no_target: bool = False,
    ) -> Tensor:

        if single:
            return torch.cat(
                [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

        elif no_target:
            return torch.cat(
                [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                ],
                dim=1,
            )

        else:
            return torch.cat(
                [
                    self.before_img_embeds,
                    image,
                    self.before_suffix_embeds,
                    self.embedding_layer(sampled_ids),
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

    def _compute_candidates_loss_original(
        self, search_batch_size: int, input_embeds: Tensor
    ) -> Tensor:
        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                outputs = self.model(inputs_embeds=input_embeds_batch)
                logits = outputs.logits
                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)
                if self.config.early_stop:
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                        )
                    ).item():
                        self.stop_flag = True
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
        return torch.cat(all_loss, dim=0)

    def _save_image(self, image, path):
        image = image.squeeze(0).detach().cpu().numpy()
        image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        image_pil.save(path)


# ---------------------------
# Runner function
# ---------------------------
def run(
    model: transformers.PreTrainedModel,
    tokenizer,
    processor,
    messages: Union[str, List[dict]],
    goal: str,
    target: str,
    image: Tensor = None,
    config: Optional[GCGConfig] = None,
    normalize=None,
) -> GCGResult:
    if config is None:
        config = GCGConfig()
    logger.setLevel(getattr(logging, config.verbosity))
    gcg = GCG(model, tokenizer, processor, config, normalize)
    result = gcg.run(messages, goal, target, image)
    return result
