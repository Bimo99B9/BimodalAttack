import copy
import logging
import time

from dataclasses import dataclass
import numpy as np
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
)

from PIL import Image
import torchvision.transforms.functional as F
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

        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
        logger.info(f"Messages 0: {messages}")

        # Ensure optim placeholder
        if (
            isinstance(messages[-1]["content"], str)
            and "{optim_str}" not in messages[-1]["content"]
        ):
            messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"
            logger.info(f"Messages 2: {messages}")

        # Insert image token for PGD
        if config.pgd_attack:
            cont = messages[-1]["content"]
            if isinstance(cont, str):
                messages[-1]["content"] = [
                    {"type": "text", "text": cont},
                    {"type": "image"},
                ]
            elif isinstance(cont, list) and not any(
                item.get("type") == "image" for item in cont
            ):
                messages[-1]["content"].append({"type": "image"})
        logger.info(f"Messages 4: {messages}")

        # Build prompt
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        logger.info(f"Prompt after applying chat template: {prompt}")

        # Strip BOS
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt.replace(tokenizer.bos_token, "")
        logger.info(f"Prompt after removing BOS token: {prompt}")

        # Tokenize before/after segments
        if config.pgd_attack:
            # Split prompt into before_img_str, before_suffix_str, after_str
            if self.processor.__class__.__name__ == "Gemma3Processor":
                before_str, after_temp = prompt.split("{optim_str}", 1)
                before_img_str = before_str.strip()
                if "<start_of_image>" in after_temp:
                    before_suffix, sep, after_str = after_temp.partition(
                        "<start_of_image>"
                    )
                    before_suffix_str = (before_suffix + sep).strip()
                    after_str = after_str.strip()
                else:
                    raise ValueError(
                        "Expected <start_of_image> token in Gemma PGD prompt."
                    )
            else:
                if "<start_of_image>" in prompt:
                    before_img_str, after_temp = prompt.split("<start_of_image>", 1)
                elif "<image>" in prompt:
                    before_img_str, after_temp = prompt.split("<image>", 1)
                else:
                    raise ValueError("No image token found in prompt for PGD attack")
                before_suffix_str, after_str = after_temp.split("{optim_str}", 1)

            # Log segments
            logger.info(f"Before image str: {before_img_str}")
            logger.info(f"Before suffix str: {before_suffix_str}")
            logger.info(f"After str: {after_str}")
            logger.info(f"Target: {target}")

            # Token IDs
            before_img_ids = tokenizer(before_img_str, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            before_suffix_ids = tokenizer(before_suffix_str, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            target_ids = tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)

            # Embeddings
            self.before_img_ids = before_img_ids
            self.before_suffix_ids = before_suffix_ids
            self.after_ids = after_ids
            self.target_ids = target_ids
            (
                self.before_img_embeds,
                self.before_suffix_embeds,
                self.after_embeds,
                self.target_embeds,
            ) = [
                self.embedding_layer(x)
                for x in (before_img_ids, before_suffix_ids, after_ids, target_ids)
            ]
        else:
            before_str, after_str = prompt.split("{optim_str}")
            logger.info(f"Before str: {before_str}")
            logger.info(f"After str: {after_str}")
            logger.info(f"Target: {target}")

            before_ids = tokenizer(before_str, return_tensors="pt")["input_ids"].to(
                model.device
            )
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            target_ids = tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)

            self.before_embeds, self.after_embeds, self.target_embeds = [
                self.embedding_layer(x) for x in (before_ids, after_ids, target_ids)
            ]
            self.target_ids = target_ids

        # Initialize buffer & starting candidate
        buffer = self.init_buffer(image)
        optim_ids = buffer.get_best_ids()

        # Containers for metrics
        losses, optim_strings, adv_suffixes, model_outputs = [], [], [], []
        gradient_times, sampling_times, loss_times, pgd_times, total_times = (
            [],
            [],
            [],
            [],
            [],
        )
        total_gradient_time = total_sampling_time = total_loss_time = total_pgd_time = (
            0.0
        )

        best_loss = float("inf")
        best_optim_ids = best_image = None
        current_loss = None

        if config.pgd_attack:
            logger.warning(f"Using alpha: {config.alpha}, eps: {config.eps}")
            image.requires_grad_(True)
            image_original = image.clone()

        # Log attack mode
        if config.pgd_attack and config.gcg_attack:
            logger.info("Running PGD and GCG")
        elif config.pgd_attack:
            logger.info("Running only PGD")
        else:
            logger.info("Running only GCG")

        # Main loop
        for i in tqdm(range(config.num_steps)):
            logger.info(
                f"[Iteration {i}] Starting (pgd={config.pgd_attack}, gcg={config.gcg_attack})"
            )

            # Phase A: compute gradient
            start = time.perf_counter()
            if config.pgd_attack:
                optim_grad, image_grad = self.compute_gradient(optim_ids, image)
            else:
                optim_grad, _ = self.compute_gradient(optim_ids)
            grad_time = time.perf_counter() - start
            gradient_times.append(grad_time)
            total_gradient_time += grad_time
            logger.info(f"[{i}] Phase A (GRADS) in {grad_time:.4f}s")

            # Phase B/C: PGD before GCG
            if config.pgd_attack:
                logger.info(f"[{i}] Phase B (PGD before GCG)")
                start = time.perf_counter()
                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, image_grad, image_original
                )
                pgd_t = time.perf_counter() - start
                pgd_times.append(pgd_t)
                total_pgd_time += pgd_t
                logger.info(f"[{i}] Phase B in {pgd_t:.4f}s")

                if config.gcg_attack and not config.joint_eval:
                    logger.info(f"[{i}] Phase C (GRADS after PGD)")
                    start = time.perf_counter()
                    optim_grad, image_grad = self.compute_gradient(optim_ids, image)
                    g2 = time.perf_counter() - start
                    gradient_times.append(g2)
                    total_gradient_time += g2
                    logger.info(f"[{i}] Phase C in {g2:.4f}s")
            else:
                pgd_t = 0.0
                logger.info(f"[{i}] Skipping PGD")

            # Phase D: sample & partial loss
            logger.info(f"[{i}] Phase D (GCG sampling)")
            sampled_ids, new_w, samp_t, total_sampling_time = self.candidate_sampling(
                config,
                i,
                optim_ids,
                optim_grad,
                tokenizer,
                sampling_times,
                total_sampling_time,
            )
            sampling_times.append(samp_t)
            logger.info(f"[{i}] Sampled {new_w} candidates in {samp_t:.4f}s")

            with torch.no_grad():
                start = time.perf_counter()
                batch_size = new_w if config.batch_size is None else config.batch_size

                if config.pgd_attack:
                    # extract image features
                    px = self.normalize(image)
                    if self.processor.__class__.__name__ == "Gemma3Processor":
                        img_feats = model.get_image_features(pixel_values=px)
                    else:
                        img_feats = model.get_image_features(
                            pixel_values=px,
                            vision_feature_layer=-2,
                            vision_feature_select_strategy="default",
                        )

                    # Standard combined PGD+GCG evaluation
                    if config.joint_eval:
                        cand_emb = self._build_input_embeds(
                            sampled_ids,
                            image=img_feats,
                            search_width=new_w,
                            mode="pgd",
                            single=True,
                        )
                    elif config.gcg_attack:
                        cand_emb = self._build_input_embeds(
                            sampled_ids,
                            image=None,
                            search_width=new_w,
                            mode="gcg",
                            single=True,
                        )
                    else:
                        cand_emb = None

                    if cand_emb is not None:
                        loss_vec = find_executable_batch_size(
                            self._compute_candidates_loss_original, batch_size
                        )(cand_emb)
                        best_before = loss_vec.min().item()
                        best_idx = loss_vec.argmin()
                    else:
                        best_before, best_idx = 0.0, 0

                    logger.info(f"[{i}] Best loss before image eval: {best_before:.4f}")

                    # Full evaluation with image
                    full_emb = self._build_input_embeds(
                        sampled_ids[best_idx].unsqueeze(0),
                        image=img_feats,
                        mode="gcg_pgd",
                    )
                    current_loss = self._compute_candidates_loss_original(
                        1, full_emb
                    ).item()
                    optim_ids = sampled_ids[best_idx].unsqueeze(0)

                    # record
                    losses.append(current_loss)
                    s = tokenizer.batch_decode(optim_ids)[0]
                    optim_strings.append(s)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)
                    if current_loss < best_loss:
                        best_loss, best_optim_ids, best_image = (
                            current_loss,
                            optim_ids.clone(),
                            image.clone(),
                        )

                    logger.info(f"[{i}] Final loss with image: {current_loss:.4f}")
                else:
                    # GCG only
                    cand_emb = self._build_input_embeds(
                        sampled_ids,
                        image=None,
                        search_width=new_w,
                        mode="gcg",
                        no_joint_eval=True,
                    )
                    loss_vec = find_executable_batch_size(
                        self._compute_candidates_loss_original, batch_size
                    )(cand_emb)
                    best_idx = loss_vec.argmin()
                    current_loss = loss_vec.min().item()
                    optim_ids = sampled_ids[best_idx].unsqueeze(0)

                    losses.append(current_loss)
                    s = tokenizer.batch_decode(optim_ids)[0]
                    optim_strings.append(s)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)
                    if current_loss < best_loss:
                        best_loss, best_optim_ids = current_loss, optim_ids.clone()

                    logger.info(f"[{i}] Final loss (GCG only): {current_loss:.4f}")

                loss_t = time.perf_counter() - start
                loss_times.append(loss_t)
                total_loss_time += loss_t
                logger.info(f"[{i}] Phase D loss in {loss_t:.4f}s")

            # Save image
            if config.pgd_attack:
                self._save_image(image, os.path.join(images_folder, f"{i}.png"))

            # Debug output
            if config.debug_output and i % 10 == 0:
                with torch.no_grad():
                    if config.pgd_attack:
                        px = self.normalize(image)
                        if self.processor.__class__.__name__ == "Gemma3Processor":
                            img_feats = model.get_image_features(pixel_values=px)
                        else:
                            img_feats = model.get_image_features(
                                pixel_values=px,
                                vision_feature_layer=-2,
                                vision_feature_select_strategy="default",
                            )
                        input_embeds = self._build_input_embeds(
                            sampled_ids,
                            image=img_feats,
                            search_width=new_w,
                            mode="gcg_pgd",
                            no_target=True,
                        )
                    else:
                        input_embeds = self._build_input_embeds(
                            sampled_ids,
                            image=None,
                            search_width=new_w,
                            mode="gcg",
                            no_target=True,
                        )
                    gen_ids = model.generate(
                        inputs_embeds=input_embeds, max_new_tokens=120
                    )
                    gen_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    logger.info(f"Output generated at iteration {i}: {gen_output}")
                model_outputs.append(gen_output)
            else:
                model_outputs.append("")

            adv_suffixes.append(tokenizer.batch_decode(optim_ids)[0])
            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to perfect match.")
                break

            iter_total = grad_time + samp_t + pgd_t + loss_t
            total_times.append(iter_total)
            logger.info(
                f"[{i}] Total time: {iter_total:.4f}s "
                f"(Grad: {grad_time:.4f}s, Samp: {samp_t:.4f}s, PGD: {pgd_t:.4f}s, Loss: {loss_t:.4f}s)"
            )

        num_iters = i + 1
        logger.warning(f"Avg grad time: {total_gradient_time/num_iters:.4f}s")
        logger.warning(f"Avg PGD time: {total_pgd_time/num_iters:.4f}s")
        logger.warning(f"Avg sampling time: {total_sampling_time/num_iters:.4f}s")
        logger.warning(f"Avg loss time: {total_loss_time/num_iters:.4f}s")

        min_idx = losses.index(min(losses))
        return BimodalAttackResult(
            best_loss=losses[min_idx],
            best_string=optim_strings[min_idx],
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
            if self.processor.__class__.__name__ == "Gemma3Processor":
                image_features = model.get_image_features(pixel_values=pixel_values)
            else:
                image_features = model.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=-2,
                    vision_feature_select_strategy="default",
                )
            init_buffer_embeds = self._build_input_embeds(
                init_buffer_ids,
                image=image_features,
                search_width=true_buffer_size,
                mode="gcg_pgd",
                single=True,
            )
        else:
            init_buffer_embeds = self._build_input_embeds(
                init_buffer_ids,
                search_width=true_buffer_size,
                mode="gcg",
                no_joint_eval=True,
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
            if self.processor.__class__.__name__ == "Gemma3Processor":
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
                if mt == "gemma3"
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
                    if mt == "gemma3"
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
                    if mt == "gemma3"
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
                    if mt == "gemma3"
                    else ["before_img", "image", "before_suffix", "optim", "after"]
                )
            else:
                # full non-single
                seq = (
                    ["before_img", "optim", "before_suffix", "image", "after", "target"]
                    if mt == "gemma3"
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

    def _compute_candidates_loss_original(
        self, search_batch_size: int, input_embeds: Tensor
    ) -> Tensor:
        with torch.no_grad():
            outputs = self.model(inputs_embeds=input_embeds)
            logits = outputs.logits
            seq_len = input_embeds.size(1)
            tgt_len = self.target_ids.numel()
            offset = seq_len - tgt_len
            shift_logits = logits[..., offset - 1 : -1, :].contiguous()
            tgt = self.target_ids.squeeze()
            if tgt.dim() != 1:
                tgt = tgt.view(-1)
            batch_size = shift_logits.size(0)
            shift_labels = tgt.unsqueeze(0).expand(batch_size, -1).contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            loss_flat = torch.nn.functional.cross_entropy(
                flat_logits, flat_labels, reduction="none"
            )
            losses = loss_flat.view(batch_size, -1).mean(dim=1)
            if self.config.early_stop:
                correct = torch.argmax(shift_logits, dim=-1) == shift_labels
                if torch.any(correct.all(dim=-1)):
                    self.stop_flag = True
        return losses

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
    config: Optional[BimodalAttackConfig] = None,
    normalize=None,
) -> BimodalAttackResult:
    if config is None:
        config = BimodalAttackConfig()
    logger.setLevel(getattr(logging, config.verbosity))
    bimodalattack = BimodalAttack(model, tokenizer, processor, config, normalize)
    return bimodalattack.run(messages, goal, target, image)
