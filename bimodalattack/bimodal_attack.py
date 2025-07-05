import copy
import logging
import time
import gc

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from bimodalattack.utils import (
    INIT_CHARS,
    find_executable_batch_size,
    get_nonascii_toks,
    sample_ids_from_grad,
    filter_ids,
    save_image,
)

import os


# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger("gcg")
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
class BimodalAttackConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: Optional[int] = None
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
    model: str = "llava"


@dataclass
class BimodalAttackResult:
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
# Main BimodalAttack class definition
# ---------------------------
class BimodalAttack:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        processor,
        config: BimodalAttackConfig,
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

        # Check the model type.
        if hasattr(self.model.config, "model_type"):
            logger.info(f"Model type: {self.model.config.model_type}")

        # Check the model identifier or name/path (if available).
        if hasattr(self.model.config, "_name_or_path"):
            logger.info(f"Model identifier: {self.model.config._name_or_path}")

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
        self,
        messages: Union[str, List[dict]],
        goal: str,
        target: str,
        image: torch.Tensor = None,
    ) -> BimodalAttackResult:
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

        # Find the last message from a user to append the optimization string
        last_user_idx = -1
        for i, msg in reversed(list(enumerate(messages))):
            if msg.get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx != -1:
            content = messages[last_user_idx].get("content", "")

            # Handle multimodal content (list of dicts)
            if isinstance(content, list):
                text_part = next((p for p in content if p.get("type") == "text"), None)
                if text_part:
                    if "{optim_str}" not in text_part.get("text", ""):
                        text_part["text"] += " {optim_str}"
                else:
                    # If no text part exists, add one
                    content.append({"type": "text", "text": "{optim_str}"})
                messages[last_user_idx]["content"] = content
                logger.info(f"Messages 2 (list content): {messages}")

            # Handle simple string content
            elif isinstance(content, str):
                if "{optim_str}" not in content:
                    messages[last_user_idx]["content"] += " {optim_str}"
                    logger.info(f"Messages 2 (str content): {messages}")
        else:
            logger.warning("No user message found to append optimization string.")

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
            if self.processor.__class__.__name__ in (
                "Gemma3Processor",
                "Gemma3nProcessor",
            ):
                # First, split the prompt on the optimization placeholder.
                before_str, after_temp = prompt.split("{optim_str}", 1)
                before_img_str = before_str.strip()

                # Define the correct single placeholder token based on the processor
                if self.processor.__class__.__name__ == "Gemma3Processor":
                    image_placeholder = "<image_token>"
                elif self.processor.__class__.__name__ == "Gemma3nProcessor":
                    image_placeholder = "<image_soft_token>"

                # Use partition to find the placeholder and retain it in the result.
                if image_placeholder in after_temp:
                    before_suffix, sep, after_str = after_temp.partition(
                        image_placeholder
                    )
                    before_suffix_str = (before_suffix + sep).strip()
                    after_str = after_str.strip()
                else:
                    logger.error(
                        f"Expected '{image_placeholder}' token in Gemma PGD prompt. Segment after '{{optim_str}}':\n{after_temp}"
                    )
                    logger.error(f"Full prompt: {prompt}")
                    logger.error(f"Messages: {messages}")
                    raise ValueError(
                        f"Expected '{image_placeholder}' token in Gemma PGD prompt."
                    )

            else:
                # Fallback for other models like Llava
                if "<image_token>" in prompt:
                    before_img_str, after_img_str = prompt.split("<image_token>", 1)
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
            before_str, after_str = prompt.split("{optim_str}", 1)
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

        # llama_guard_found_unsafe = False

        # Initialize variables to track the best candidate (lowest loss).
        best_loss = float("inf")
        best_optim_ids = None
        best_image = None
        current_loss = None

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

                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, image_grad, image_original
                )
                # image = self.perform_autopgd_step(
                # image, config.eps, image_grad, image_original, current_loss, i
                # )

                pgd_time = time.perf_counter() - start_pgd
                pgd_times.append(pgd_time)
                total_pgd_time += pgd_time
                logger.info(
                    f"[Iteration {i}] Phase B (PGD update) completed in {pgd_time:.4f}s"
                )

                ### Phase C - Recompute gradient after PGD update. Do not recompute gradients if using joint eval.
                if config.gcg_attack and not config.joint_eval:
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
                    if self.processor.__class__.__name__ in (
                        "Gemma3Processor",
                        "Gemma3nProcessor",
                    ):
                        image_features = model.get_image_features(
                            pixel_values=pixel_values
                        )
                    else:
                        image_features = model.get_image_features(
                            pixel_values=pixel_values,
                            vision_feature_layer=-2,
                            vision_feature_select_strategy="default",
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
                logger.info(
                    f"[Iteration {i}] Current loss: {current_loss:.4f} | "
                    f"Best loss: {buffer.get_lowest_loss():.4f} | "
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
                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, image_grad, image_original
                )
                # image = self.perform_autopgd_step(
                #     image, config.eps, image_grad, image_original, current_loss, i
                # )

                pgd_time = time.perf_counter() - start_pgd
                pgd_times.append(pgd_time)
                total_pgd_time += pgd_time
                logger.info(
                    f"[Iteration {i}] Phase F (PGD update after GCG) completed in {pgd_time:.4f}s"
                )

                with torch.no_grad():
                    start_loss = time.perf_counter()
                    pixel_values = self.normalize(image)
                    if self.processor.__class__.__name__ in (
                        "Gemma3Processor",
                        "Gemma3nProcessor",
                    ):
                        image_features = model.get_image_features(
                            pixel_values=pixel_values
                        )
                    else:
                        image_features = model.get_image_features(
                            pixel_values=pixel_values,
                            vision_feature_layer=-2,
                            vision_feature_select_strategy="default",
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
                save_image(image, os.path.join(images_folder, f"{i}.png"))
            if config.debug_output and i % 10 == 0:
                with torch.no_grad():
                    if config.pgd_attack:
                        pixel_values = self.normalize(image)
                        if self.processor.__class__.__name__ in (
                            "Gemma3Processor",
                            "Gemma3nProcessor",
                        ):
                            image_features = model.get_image_features(
                                pixel_values=pixel_values
                            )
                        else:
                            image_features = model.get_image_features(
                                pixel_values=pixel_values,
                                vision_feature_layer=-2,
                                vision_feature_select_strategy="default",
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

        min_loss_index = losses.index(min(losses))
        result = BimodalAttackResult(
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
            if self.processor.__class__.__name__ in (
                "Gemma3Processor",
                "Gemma3nProcessor",
            ):
                image_features = model.get_image_features(pixel_values=pixel_values)
            else:
                image_features = model.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=-2,
                    vision_feature_select_strategy="default",
                )

            init_buffer_embeds = self._build_input_embeds_gcg_pgd(
                init_buffer_ids,
                image_features,
                search_width=true_buffer_size,
                single=True,
            )
        else:
            init_buffer_embeds = self._build_input_embeds_gcg(
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
        config: BimodalAttackConfig,
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
            if self.processor.__class__.__name__ in (
                "Gemma3Processor",
                "Gemma3nProcessor",
            ):
                image_features = model.get_image_features(pixel_values=pixel_values)
            else:
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

    def perform_pgd_step(
        self, image: Tensor, eps: float, alpha: float, image_grad, image_original
    ) -> Tensor:
        image = (image - alpha * eps * torch.sign(image_grad)).detach().requires_grad_()
        image = torch.clamp(image, image_original - eps, image_original + eps)
        image = torch.clamp(image, 0, 1)

        return image

    def perform_autopgd_step(
        self,
        image: torch.Tensor,
        eps: float,
        image_grad: torch.Tensor,
        image_original: torch.Tensor,
        current_loss: Optional[float],
        iter_idx: int,
    ) -> torch.Tensor:
        alpha = 0.75
        checkpoint_interval = 10
        rho = 0.75

        # Initialize APGD state on the first call.
        if not hasattr(self, "pgd_state_initialized"):
            self.pgd_prev_image = image.clone()
            self.pgd_best_image = image.clone()
            self.pgd_best_loss = (
                current_loss if current_loss is not None else float("inf")
            )
            self.pgd_current_eta = 2 * eps  # initial step size as in the paper
            self.pgd_improvement_count = 0
            self.pgd_last_best_loss = self.pgd_best_loss
            self.pgd_state_initialized = True

        # --- APGD Update Step with Momentum ---
        # 1. Compute the gradient sign (for ℓ∞ attacks, using the descent direction).
        grad_sign = torch.sign(image_grad)

        # Take a descent step
        z = image - self.pgd_current_eta * grad_sign
        z = torch.max(torch.min(z, image_original + eps), image_original - eps).clamp(
            0, 1
        )

        # 2. Incorporate momentum:
        new_image = (
            image + alpha * (z - image) + (1 - alpha) * (image - self.pgd_prev_image)
        )
        new_image = torch.max(
            torch.min(new_image, image_original + eps), image_original - eps
        ).clamp(0, 1)

        self.pgd_prev_image = image.clone()

        # --- Step-size Adaptation and Best-Image Tracking ---
        if current_loss is not None:
            # Update best seen point.
            if current_loss < self.pgd_best_loss:
                self.pgd_best_loss = current_loss
                self.pgd_best_image = new_image.clone()

            # Count improvements relative to the best loss so far.
            if current_loss < self.pgd_last_best_loss:
                self.pgd_improvement_count += 1

            # At checkpoint intervals, decide whether to reduce the step size.
            if (iter_idx + 1) % checkpoint_interval == 0:
                improvement_fraction = self.pgd_improvement_count / checkpoint_interval
                if (
                    improvement_fraction < rho
                    or self.pgd_best_loss == self.pgd_last_best_loss
                ):
                    # Halve the step size and restart from the best solution so far.
                    self.pgd_current_eta /= 2
                    new_image = self.pgd_best_image.clone()
                    self.pgd_prev_image = self.pgd_best_image.clone()
                # Reset the improvement count and update the checkpoint best loss.
                self.pgd_improvement_count = 0
                self.pgd_last_best_loss = self.pgd_best_loss

        return new_image

    def _build_input_embeds(
        self,
        sampled_ids: Tensor,
        image: Optional[Tensor] = None,
        search_width: Optional[int] = None,
        mode: str = "gcg",
        single: bool = False,
        no_joint_eval: bool = False,
        no_target: bool = False,
    ) -> Tensor:
        """
        Unified embed builder:
        - automatically repeats any 1-element batch dims to `search_width`
        - handles 'gcg', 'pgd', and 'gcg_pgd' modes with identical semantics
        """
        mt = self.model.config.model_type
        emb = self.embedding_layer
        sw = search_width

        # segment→tensor map
        def get(seg: str) -> Tensor:
            if seg == "before":
                return self.before_embeds
            if seg == "before_img":
                return self.before_img_embeds
            if seg == "before_suffix":
                return self.before_suffix_embeds
            if seg == "image":
                return image
            if seg == "optim":
                return emb(sampled_ids)
            if seg == "after":
                return self.after_embeds
            if seg == "target":
                return self.target_embeds
            raise ValueError(f"Unknown segment '{seg}'")

        # pick sequence of segments
        if mode == "pgd":
            assert single, "PGD mode only supports single=True"
            seq = (
                ["before_img", "optim", "before_suffix", "image", "after", "target"]
                if mt in ("gemma3", "gemma3n")
                else [
                    "before_img",
                    "image",
                    "before_suffix",
                    "optim",
                    "after",
                    "target",
                ]
            )

        elif mode == "gcg":
            if single:
                seq = (
                    ["before_img", "optim", "before_suffix", "after", "target"]
                    if mt in ("gemma3", "gemma3n")
                    else ["before_img", "before_suffix", "optim", "after", "target"]
                )
            elif no_joint_eval:
                seq = ["before", "optim", "after", "target"]
            elif no_target:
                seq = ["before", "optim", "after"]
            else:
                raise ValueError("Invalid flags for BimodalAttack mode")

        elif mode == "gcg_pgd":
            if single:
                seq = (
                    ["before_img", "optim", "before_suffix", "image", "after", "target"]
                    if mt in ("gemma3", "gemma3n")
                    else [
                        "before_img",
                        "image",
                        "before_suffix",
                        "optim",
                        "after",
                        "target",
                    ]
                )
            elif no_target:
                seq = (
                    ["before_img", "optim", "before_suffix", "image", "after"]
                    if mt in ("gemma3", "gemma3n")
                    else ["before_img", "image", "before_suffix", "optim", "after"]
                )
            else:
                # full non-single
                seq = (
                    ["before_img", "optim", "before_suffix", "image", "after", "target"]
                    if mt in ("gemma3", "gemma3n")
                    else [
                        "before_img",
                        "image",
                        "before_suffix",
                        "optim",
                        "after",
                        "target",
                    ]
                )

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # gather & repeat
        parts: List[Tensor] = []
        for seg in seq:
            t = get(seg)  # shape (1, L, D) or (sw, L, D) for 'optim'
            if sw is not None and t.shape[0] == 1:
                # replicate any 1-batch element to match search_width
                t = t.repeat(sw, 1, 1)
            parts.append(t)
        return torch.cat(parts, dim=1)

    # backward compatibility wrappers
    def _build_input_embeds_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width: int,
        single: bool = False,
    ) -> Tensor:
        return self._build_input_embeds(
            sampled_ids,
            image=image,
            search_width=search_width,
            mode="pgd",
            single=single,
        )

    def _build_input_embeds_gcg(
        self,
        sampled_ids: Tensor,
        search_width: int,
        single: bool = False,
        no_joint_eval: bool = False,
        no_target: bool = False,
    ) -> Tensor:
        return self._build_input_embeds(
            sampled_ids,
            image=None,
            search_width=search_width,
            mode="gcg",
            single=single,
            no_joint_eval=no_joint_eval,
            no_target=no_target,
        )

    def _build_input_embeds_gcg_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width: Optional[int] = None,
        single: bool = False,
        no_target: bool = False,
    ) -> Tensor:
        return self._build_input_embeds(
            sampled_ids,
            image=image,
            search_width=search_width,
            mode="gcg_pgd",
            single=single,
            no_target=no_target,
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

    # def _compute_candidates_loss_original(
    #     self, search_batch_size: int, input_embeds: Tensor
    # ) -> Tensor:
    #     """
    #     Computes the loss for candidate prompts using the negative margin loss.

    #     This updated implementation is based on the recommendations from the papers
    #     "Adversarial Training Should be Cast as a Non-Zero-Sum Game" and "Improving SAM"
    #     It replaces the surrogate cross-entropy loss with a loss that
    #     directly maximizes the misclassification error, which is the objective of the
    #     adversary in the proposed non-zero-sum / bilevel game.

    #     The loss to be MINIMIZED by the optimizer is:
    #     Loss = logit_of_correct_token - logit_of_best_incorrect_token

    #     This is equivalent to MAXIMIZING the negative margin, which is the attacker's goal
    #     """
    #     all_loss = []
    #     for i in range(0, input_embeds.shape[0], search_batch_size):
    #         with torch.no_grad():
    #             input_embeds_batch = input_embeds[i : i + search_batch_size]
    #             current_batch_size = input_embeds_batch.shape[0]

    #             outputs = self.model(inputs_embeds=input_embeds_batch)
    #             logits = outputs.logits

    #             # Align logits and labels for next-token prediction loss
    #             prompt_len = input_embeds.shape[1] - self.target_ids.shape[1]
    #             shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
    #             shift_labels = self.target_ids.repeat(current_batch_size, 1)

    #             # --- Early stopping check (must be done with original logits) ---
    #             if self.config.early_stop:
    #                 # Check if the target sequence is perfectly predicted
    #                 if torch.any(
    #                     torch.all(
    #                         torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
    #                     )
    #                 ).item():
    #                     self.stop_flag = True

    #             # --- BETA / Negative Margin Loss Calculation ---
    #             # This implements the objective from Algorithm 1 (BETA) in the first paper
    #             # and is the core idea of the BiSAM formulation in the second paper.

    #             # Get the logits of the correct target tokens (shape: [batch_size, seq_len])
    #             correct_token_logits = torch.gather(
    #                 shift_logits, -1, shift_labels.unsqueeze(-1)
    #             ).squeeze(-1)

    #             # To find the best incorrect logit, we set the correct logit to -inf
    #             # We clone the logits tensor to avoid modifying it in-place before the early stopping check
    #             temp_logits = shift_logits.clone()
    #             temp_logits.scatter_(-1, shift_labels.unsqueeze(-1), float('-inf'))

    #             # Get the max logit among all incorrect tokens
    #             max_incorrect_token_logits = torch.max(temp_logits, dim=-1).values

    #             # The loss is the negative of the margin, which the optimizer will minimize.
    #             loss = correct_token_logits - max_incorrect_token_logits

    #             # Average the loss over the sequence dimension, similar to the original implementation
    #             loss = loss.mean(dim=-1)
    #             all_loss.append(loss)

    #             # Memory management
    #             del outputs, logits, shift_logits, shift_labels, temp_logits, correct_token_logits, max_incorrect_token_logits
    #             gc.collect()
    #             torch.cuda.empty_cache()

    #     return torch.cat(all_loss, dim=0)


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
    config: Optional[BimodalAttackConfig] = None,
    normalize=None,
) -> BimodalAttackResult:
    if config is None:
        config = BimodalAttackConfig()
    logger.setLevel(getattr(logging, config.verbosity))
    bimodalattack = BimodalAttack(model, tokenizer, processor, config, normalize)
    return bimodalattack.run(messages, goal, target, image)
