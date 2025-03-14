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
from scipy.stats import spearmanr

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

import requests
from PIL import Image
import torchvision.transforms.functional as F
import os

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


@contextmanager
def timed_section(section_name: str):
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"[Timing] {section_name} took {elapsed_time:.4f} seconds")


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
    experiment_folder: str = "experiments/missing_folder"


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    adversarial_suffixes: List[str]
    image_ids: List[str]
    model_outputs: List[str]


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


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        processor,  # processor is passed in to build the prompt
        config: GCGConfig,
        transform=None,
        normalize=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.transform = transform
        self.normalize = normalize

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )

        self.stop_flag = False

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if model.device == torch.device("cpu"):
            logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            if config.pgd_attack:
                logger.warning(
                    "Tokenizer does not have a chat template. Using custom chat template for GCG+PGD attack."
                )
                tokenizer.chat_template = (
                    "USER: <image>\n"
                    "{{ messages[0]['content'][0]['text'] }} \n"
                    "ASSISTANT: "
                )
                self.processor.chat_template = tokenizer.chat_template
            else:
                logger.warning(
                    "Tokenizer does not have a chat template. Using custom chat template for GCG only attack."
                )
                tokenizer.chat_template = (
                    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                )
                self.processor.chat_template = tokenizer.chat_template

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
        image: Tensor = None,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        # Create an images folder within the experiment folder.
        images_folder = os.path.join(config.experiment_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        logger.info(f"Messages 0: {messages}")

        with timed_section("Chat template and tokenization setup"):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            else:
                messages = copy.deepcopy(messages)
                logger.info(f"Messages 1: {messages}")

            # Append the {optim_str} placeholder if not already present.
            if (
                isinstance(messages[-1]["content"], str)
                and "{optim_str}" not in messages[-1]["content"]
            ):
                messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"
                logger.info(f"Messages 2: {messages}")

            # For PGD attack, convert the user message to a list with text and image elements.
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

            # Apply the chat template.
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            logger.info(f"Prompt after applying chat template: {prompt}")

            # Remove BOS token if present.
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            logger.info(f"Prompt after removing BOS token: {prompt}")

            if config.pgd_attack:
                before_img_str, after_img_str = prompt.split("<image>")
                before_suffix_str, after_str = after_img_str.split("{optim_str}")
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

        embedding_layer = self.embedding_layer
        with timed_section("Embedding for before, after, and target"):
            if config.pgd_attack:
                before_img_embeds, before_suffix_embeds, after_embeds, target_embeds = [
                    embedding_layer(ids)
                    for ids in (
                        before_img_ids,
                        before_suffix_ids,
                        after_ids,
                        target_ids,
                    )
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
                    embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
                ]
                self.target_ids = target_ids
                self.before_embeds = before_embeds
                self.after_embeds = after_embeds
                self.target_embeds = target_embeds

        with timed_section("Attack buffer initialization"):
            buffer = self.init_buffer(image)

        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        total_gradient_time = 0.0
        total_sampling_time = 0.0
        total_loss_time = 0.0
        total_pgd_time = 0.0

        if config.pgd_attack:
            logger.warning(f"Using alpha: {config.alpha}, eps: {config.eps}")
            image.requires_grad = True
            image_original = image.clone()

        adv_suffixes = []
        image_ids = []
        model_outputs = []

        for i in tqdm(range(config.num_steps)):
            iter_start = time.perf_counter()

            with timed_section(f"Iteration {i}: Compute token and image gradient"):
                start_grad = time.perf_counter()
                if config.pgd_attack:
                    optim_ids_onehot_grad, image_grad = self.compute_gradient(
                        optim_ids, image
                    )
                else:
                    optim_ids_onehot_grad, _ = self.compute_gradient(optim_ids)
                grad_time = time.perf_counter() - start_grad
                total_gradient_time += grad_time

            if config.pgd_attack:
                with timed_section(f"Iteration {i}: PGD Update"):
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
                    total_pgd_time += pgd_time

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
                with timed_section(f"Iteration {i}: Candidate sampling"):
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
                    total_sampling_time += sampling_time
            else:
                sampled_ids = optim_ids
                new_search_width = 1
                sampling_time = 0.0

            with torch.no_grad():
                with timed_section(f"Iteration {i}: Candidate loss computation"):
                    start_loss = time.perf_counter()
                    batch_size = (
                        new_search_width
                        if config.batch_size is None
                        else config.batch_size
                    )

                    if config.pgd_attack:
                        pixel_values = self.normalize(image)
                        image_features = model.get_image_features(
                            pixel_values=pixel_values,
                            vision_feature_layer=-2,
                            vision_feature_select_strategy="default",
                        )
                        input_embeds = torch.cat(
                            [
                                self.before_img_embeds.repeat(new_search_width, 1, 1),
                                image_features.repeat(new_search_width, 1, 1),
                                self.before_suffix_embeds.repeat(
                                    new_search_width, 1, 1
                                ),
                                embedding_layer(sampled_ids),
                                self.after_embeds.repeat(new_search_width, 1, 1),
                                self.target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )
                    else:
                        input_embeds = torch.cat(
                            [
                                self.before_embeds.repeat(new_search_width, 1, 1),
                                embedding_layer(sampled_ids),
                                self.after_embeds.repeat(new_search_width, 1, 1),
                                self.target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )

                    loss = find_executable_batch_size(
                        self._compute_candidates_loss_original, batch_size
                    )(input_embeds)
                    current_loss = loss.min().item()
                    logger.info(f"[Iteration {i}] Current loss: {current_loss:.4f}")
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                    loss_time = time.perf_counter() - start_loss
                    total_loss_time += loss_time

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            if i % 10 == 0 and config.debug_output:
                current_image_id = os.path.join(images_folder, f"image_{i}.png")
                if config.pgd_attack:
                    self._save_image(image, current_image_id)
                    pixel_values = self.normalize(image)
                    image_features = model.get_image_features(
                        pixel_values=pixel_values,
                        vision_feature_layer=-2,
                        vision_feature_select_strategy="default",
                    )
                    input_embeds = torch.cat(
                        [
                            self.before_img_embeds.repeat(new_search_width, 1, 1),
                            image_features.repeat(new_search_width, 1, 1),
                            self.before_suffix_embeds.repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            self.after_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            self.before_embeds.repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            self.after_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                generated_ids = model.generate(
                    inputs_embeds=input_embeds, max_new_tokens=150
                )
                gen_output = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                logger.info(f"Output generated at iteration {i}: {gen_output}")
            else:
                current_image_id = ""
                gen_output = ""

            adv_suffixes.append(optim_str)
            image_ids.append(current_image_id)
            model_outputs.append(gen_output)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

            iter_total = time.perf_counter() - iter_start
            logger.info(
                f"[Iteration {i}] Total iteration time: {iter_total:.4f}s (Gradient: {grad_time:.4f}s, Sampling: {sampling_time:.4f}s, Loss: {loss_time:.4f}s)"
            )

        num_iters = i + 1
        logger.warning(
            f"Average token gradient time: {total_gradient_time / num_iters:.4f}s"
        )
        logger.warning(f"Average PGD update time: {total_pgd_time / num_iters:.4f}s")
        logger.warning(
            f"Average candidate sampling time: {total_sampling_time / num_iters:.4f}s"
        )
        logger.warning(
            f"Average candidate loss computation time: {total_loss_time / num_iters:.4f}s"
        )

        # Save the final image to the experiment's images folder.
        final_image_path = os.path.join(images_folder, f"image_{i}.png")
        self._save_image(image, final_image_path)

        min_loss_index = losses.index(min(losses))
        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            adversarial_suffixes=adv_suffixes,
            image_ids=image_ids,
            model_outputs=model_outputs,
        )
        return result

    def init_buffer(self, image) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")
        with timed_section("Attack buffer creation and initialization"):
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
                    vision_feature_layer=-2,
                    vision_feature_select_strategy="default",
                )
                init_buffer_embeds = torch.cat(
                    [
                        self.before_img_embeds.repeat(true_buffer_size, 1, 1),
                        image_features.repeat(true_buffer_size, 1, 1),
                        self.before_suffix_embeds.repeat(true_buffer_size, 1, 1),
                        self.embedding_layer(init_buffer_ids),
                        self.after_embeds.repeat(true_buffer_size, 1, 1),
                        self.target_embeds.repeat(true_buffer_size, 1, 1),
                    ],
                    dim=1,
                )
            else:
                init_buffer_embeds = torch.cat(
                    [
                        self.before_embeds.repeat(true_buffer_size, 1, 1),
                        self.embedding_layer(init_buffer_ids),
                        self.after_embeds.repeat(true_buffer_size, 1, 1),
                        self.target_embeds.repeat(true_buffer_size, 1, 1),
                    ],
                    dim=1,
                )
            with timed_section("Initial buffer loss computation"):
                init_buffer_losses = find_executable_batch_size(
                    self._compute_candidates_loss_original, true_buffer_size
                )(init_buffer_embeds)
            for i in range(true_buffer_size):
                buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
            buffer.log_buffer(tokenizer)
            logger.info("Initialized attack buffer.")
        return buffer

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
                vision_feature_layer=-2,
                vision_feature_select_strategy="default",
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


def run(
    model: transformers.PreTrainedModel,
    tokenizer,
    processor,
    messages: Union[str, List[dict]],
    target: str,
    image: Tensor = None,
    config: Optional[GCGConfig] = None,
    transform=None,
    normalize=None,
) -> GCGResult:
    """
    Generates a single optimized string using GCG.
    """
    if config is None:
        config = GCGConfig()
    logger.setLevel(getattr(logging, config.verbosity))
    gcg = GCG(model, tokenizer, processor, config, transform, normalize)
    result = gcg.run(messages, target, image)
    return result
