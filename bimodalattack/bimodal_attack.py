# bimodal_attack.py

import copy
import logging
import time
import gc
import os
import json

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from bimodalattack.utils import (
    INIT_CHARS,
    get_nonascii_toks,
    sample_ids_from_grad,
    filter_ids,
    save_image,
    configure_pad_token,
    should_reduce_batch_size,
)


# ---------------------------
# Logging configuration
# ---------------------------
# The logger is now configured in the main experiment script for better control.
# This ensures consistency and allows directing logs to both console and file.
logger = logging.getLogger("bimodalattack")


# ---------------------------
# Dataclass definitions
# ---------------------------
@dataclass
class BimodalAttackConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = (
        None  # This will be the starting batch size for evaluation
    )
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
            if loss < self.buffer[-1][0]:
                self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "AttackBuffer content:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.decode(ids.squeeze(0), skip_special_tokens=True)
            optim_str = optim_str.replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\n  - Loss: {loss:.4f} | String: '{optim_str}'"
        logger.debug(message)


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
        self.start_anchor = self.tokenizer.eos_token
        self.end_anchor = self.tokenizer.eos_token
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.stop_flag = False
        self.executable_batch_size = config.batch_size

        configure_pad_token(self.tokenizer)
        self.tokenizer.padding_side = "left"

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type for faster optimization."
            )

    def _prepare_message_template(self, messages: List[dict]) -> Tuple[List[dict], str]:
        messages = copy.deepcopy(messages)
        last_user_idx = -1
        for i, msg in reversed(list(enumerate(messages))):
            if msg.get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx == -1:
            raise ValueError("No user message found to insert optimization string.")

        injection_placeholder = "{optim_str}"
        attack_template = f"{self.start_anchor}{{optim_str}}{self.end_anchor}"

        content = messages[last_user_idx].get("content", "")
        if isinstance(content, str):
            content = (
                content.replace(injection_placeholder, attack_template)
                if injection_placeholder in content
                else content + f" {attack_template}"
            )
            messages[last_user_idx]["content"] = content
        elif isinstance(content, list):
            text_part = next((p for p in content if p.get("type") == "text"), None)
            if text_part and injection_placeholder in text_part.get("text", ""):
                text_part["text"] = text_part["text"].replace(
                    injection_placeholder, attack_template
                )
            elif text_part:
                text_part["text"] += f" {attack_template}"
            else:
                content.append({"type": "text", "text": attack_template})

        if self.config.pgd_attack:
            content_list = messages[last_user_idx]["content"]
            if isinstance(content_list, str):
                content_list = [{"type": "text", "text": content_list}]
            if not any(item.get("type") == "image" for item in content_list):
                content_list.insert(0, {"type": "image"})
            messages[last_user_idx]["content"] = content_list

        prompt_template = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.debug(f"Final prompt template string prepared:\n{prompt_template}")
        return messages, prompt_template

    def run(
        self,
        messages: Union[str, List[dict]],
        goal: str,
        target: str,
        image: torch.Tensor = None,
    ) -> BimodalAttackResult:
        config = self.config
        os.makedirs(config.images_folder, exist_ok=True)

        if config.seed is not None:
            set_seed(config.seed)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Prepare templates and log initial parameters
        self.messages_template, self.prompt_template_str = (
            self._prepare_message_template(messages)
        )
        config_dict = {k: str(v) for k, v in self.config.__dict__.items()}

        logger.info("=" * 80)
        logger.info(f"{'BIMODAL ATTACK STARTED':^80}")
        logger.info("=" * 80)
        logger.info(f"Configuration:\n{json.dumps(config_dict, indent=2, default=str)}")
        logger.info("-" * 80)
        logger.info(f"Goal: {goal}")
        logger.info(f"Target: '{target}'")
        logger.info(
            f"Input Messages Template:\n{json.dumps(self.messages_template, indent=2)}"
        )
        logger.info("=" * 80)

        if config.pgd_attack and image is not None:
            logger.info("Processing initial image for PGD attack.")
            image = self.processor(text="", images=image, return_tensors="pt").to(
                self.model.device
            )["pixel_values"]
            logger.debug(f"Initial image tensor shape: {image.shape}")

        self.target_ids = self.tokenizer(
            target, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        self.initial_goal_text = goal
        self.start_anchor_id = self.tokenizer.convert_tokens_to_ids(self.start_anchor)
        self.end_anchor_id = self.tokenizer.convert_tokens_to_ids(self.end_anchor)
        logger.debug(f"Target IDs: {self.target_ids.squeeze().tolist()}")

        buffer = self.init_buffer(image)
        optim_ids = buffer.get_best_ids()

        losses, optim_strings, adv_suffixes, model_outputs = [], [], [], []
        gradient_times, sampling_times, loss_times, pgd_times, total_times = (
            [],
            [],
            [],
            [],
            [],
        )
        best_loss = float("inf")
        best_optim_ids = optim_ids.clone()
        image_original, best_image = (
            (image.clone(), image.clone()) if config.pgd_attack else (None, None)
        )

        for i in tqdm(range(config.num_steps), desc="Bimodal Attack"):
            iter_start_time = time.perf_counter()
            total_grad_time = 0.0
            current_pgd_time = 0.0

            logger.debug(f"\n{'='*20} Iteration {i+1}/{config.num_steps} {'='*20}")
            # GCG Step
            if config.gcg_attack:
                logger.debug("--- GCG GRAD START ---")
                grad_start_time = time.perf_counter()
                optim_ids_onehot_grad, grad_optim_ids = self.compute_text_gradient(
                    optim_ids, image
                )
                total_grad_time += time.perf_counter() - grad_start_time
                logger.debug("--- GCG GRAD END ---")

                logger.debug("--- CANDIDATE SAMPLING START ---")
                sample_start_time = time.perf_counter()
                sampled_ids = self.candidate_sampling(
                    grad_optim_ids, optim_ids_onehot_grad
                )
                sampling_time = time.perf_counter() - sample_start_time
                logger.debug(f"Sampled {len(sampled_ids)} candidates.")
                logger.debug("--- CANDIDATE SAMPLING END ---")
            else:
                sampled_ids = optim_ids
                sampling_time = 0.0

            # PGD Step
            if config.pgd_attack:
                logger.debug("--- PGD STEP START ---")
                pgd_grad_start_time = time.perf_counter()
                image_grad = self.compute_image_gradient(optim_ids, image)
                total_grad_time += time.perf_counter() - pgd_grad_start_time

                pgd_step_start_time = time.perf_counter()
                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, image_grad, image_original
                )
                current_pgd_time += time.perf_counter() - pgd_step_start_time
                logger.debug("--- PGD STEP END ---")

            gradient_times.append(total_grad_time)
            sampling_times.append(sampling_time)
            pgd_times.append(current_pgd_time)

            # Evaluation Step
            logger.debug("--- CANDIDATE EVALUATION START ---")
            loss_start_time = time.perf_counter()
            candidate_losses = self._compute_candidates_loss(sampled_ids, image)
            current_loss = candidate_losses.min().item()
            best_candidate_idx = candidate_losses.argmin()
            optim_ids = sampled_ids[best_candidate_idx].unsqueeze(0)
            loss_time = time.perf_counter() - loss_start_time
            loss_times.append(loss_time)
            logger.debug(
                f"Evaluated {len(sampled_ids)} candidates. Best loss in batch: {current_loss:.4f}"
            )
            logger.debug("--- CANDIDATE EVALUATION END ---")

            losses.append(current_loss)
            current_optim_str = self.tokenizer.decode(
                optim_ids.squeeze(0), skip_special_tokens=True
            )
            optim_strings.append(current_optim_str)
            adv_suffixes.append(current_optim_str)

            if buffer.size > 0:
                buffer.add(current_loss, optim_ids)
                if i % 10 == 0 or i == config.num_steps - 1:
                    buffer.log_buffer(self.tokenizer)

            if current_loss < best_loss:
                logger.info(
                    f"New best loss found: {current_loss:.4f} (previously {best_loss:.4f})"
                )
                best_loss = current_loss
                best_optim_ids = optim_ids.clone()
                if config.pgd_attack:
                    best_image = image.clone()
                    logger.debug("Updated best_image with current perturbed image.")

            gen_output = ""
            if config.debug_output and (i % 10 == 0 or i == config.num_steps - 1):
                gen_output = self.generate_test_output(optim_ids, image)
            model_outputs.append(gen_output)

            iter_total_time = time.perf_counter() - iter_start_time
            total_times.append(iter_total_time)
            self._log_iteration_summary(
                i,
                current_loss,
                best_loss,
                current_optim_str,
                iter_total_time,
                total_grad_time,
                sampling_time,
                loss_time,
                current_pgd_time,
            )

            if config.pgd_attack:
                save_image(image, os.path.join(config.images_folder, f"{i}.png"))
            if self.stop_flag:
                logger.info(
                    "Early stopping triggered: Found an exact match for the target."
                )
                break

        return BimodalAttackResult(
            best_loss=best_loss,
            best_string=self.tokenizer.decode(
                best_optim_ids.squeeze(0), skip_special_tokens=True
            ),
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

    def candidate_sampling(
        self, optim_ids: Tensor, optim_ids_onehot_grad: Optional[Tensor]
    ) -> Tensor:
        if not self.config.gcg_attack or optim_ids_onehot_grad is None:
            logger.debug("Skipping candidate sampling as GCG attack is disabled.")
            return optim_ids

        sampled_ids = sample_ids_from_grad(
            optim_ids.squeeze(0),
            optim_ids_onehot_grad.squeeze(0),
            self.config.search_width,
            self.config.topk,
            self.config.n_replace,
            not_allowed_ids=self.not_allowed_ids,
        )
        if self.config.filter_ids:
            num_before_filter = len(sampled_ids)
            sampled_ids = filter_ids(sampled_ids, self.tokenizer)
            num_after_filter = len(sampled_ids)
            logger.debug(
                f"Filtered candidates: {num_before_filter} -> {num_after_filter}"
            )

        return sampled_ids if len(sampled_ids) > 0 else optim_ids

    def _get_prompt_texts_for_candidates(self, optim_ids_batch: Tensor) -> List[str]:
        optim_strings = self.tokenizer.batch_decode(
            optim_ids_batch, skip_special_tokens=True
        )
        return [self.prompt_template_str.format(optim_str=s) for s in optim_strings]

    def _compute_candidates_loss(
        self, sampled_ids: Tensor, image: Optional[Tensor]
    ) -> Tensor:
        # If executable batch size is not yet determined, find it by starting with a high value.
        if self.executable_batch_size is None:
            self.executable_batch_size = self.config.search_width
            logger.info(
                f"Determining executable batch size, starting with {self.executable_batch_size}..."
            )

        # Inner function to compute loss for a given batch size.
        def _compute_loss_for_batch(batch_size: int):
            all_losses = []
            logger.debug(f"Computing candidate loss with batch size: {batch_size}")
            for i in range(0, sampled_ids.shape[0], batch_size):
                ids_batch = sampled_ids[i : i + batch_size]
                current_batch_size = ids_batch.shape[0]
                prompt_texts = self._get_prompt_texts_for_candidates(ids_batch)

                # logger.debug(
                #     f"Processing batch of {len(prompt_texts)}. Example prompt text:\n{prompt_texts[0]}"
                # )
                logger.debug(
                    f"Processing batch of {len(prompt_texts)}."
                )

                images_to_process = (
                    [image] * current_batch_size
                    if self.config.pgd_attack and image is not None
                    else None
                )
                if images_to_process:
                    logger.debug(f"Image tensor attached with shape: {image.shape}")

                inputs = self.processor(
                    text=prompt_texts,
                    images=images_to_process,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)

                prompt_ids = inputs["input_ids"]
                prompt_attention_mask = inputs["attention_mask"]
                prompt_len = prompt_ids.shape[1]
                target_ids_batch = self.target_ids.repeat(current_batch_size, 1)
                target_attention_mask = torch.ones_like(target_ids_batch)
                full_input_ids = torch.cat([prompt_ids, target_ids_batch], dim=1)
                full_attention_mask = torch.cat(
                    [prompt_attention_mask, target_attention_mask], dim=1
                )

                model_kwargs = {
                    "input_ids": full_input_ids,
                    "attention_mask": full_attention_mask,
                }
                if "pixel_values" in inputs:
                    model_kwargs["pixel_values"] = inputs["pixel_values"]

                with torch.no_grad():
                    outputs = self.model(**model_kwargs)

                logits = outputs.logits
                shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
                shift_labels = target_ids_batch
                loss = (
                    torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )
                    .view(current_batch_size, -1)
                    .mean(dim=-1)
                )
                all_losses.append(loss)

                if self.config.early_stop and torch.any(
                    torch.all(
                        torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                    )
                ):
                    self.stop_flag = True
                    break

            if self.stop_flag:
                num_processed = sum(len(l) for l in all_losses)
                remaining = sampled_ids.shape[0] - num_processed
                if remaining > 0:
                    all_losses.append(
                        torch.full((remaining,), float("inf"), device=self.model.device)
                    )

            return torch.cat(all_losses, dim=0)

        # Loop to find and set the executable batch size, which persists in self.executable_batch_size
        while self.executable_batch_size > 0:
            try:
                gc.collect()
                torch.cuda.empty_cache()

                # Try to compute loss with the current batch size
                losses = _compute_loss_for_batch(self.executable_batch_size)

                # If successful for the first time, log the found batch size
                if "batch_size_found" not in self.__dict__:
                    logger.info(
                        f"Successfully set executable batch size to: {self.executable_batch_size}. This will be used for all subsequent steps."
                    )
                    self.__dict__["batch_size_found"] = True

                return losses
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.executable_batch_size //= 2
                    logger.warning(
                        f"Resource error caught. Halving batch size to: {self.executable_batch_size}"
                    )
                else:
                    raise

        raise RuntimeError(
            "Could not find an executable batch size, even with a size of 1."
        )

    def init_buffer(self, image: Optional[Tensor]) -> AttackBuffer:
        buffer = AttackBuffer(self.config.buffer_size)
        logger.info(f"Initializing attack with string: '{self.config.optim_str_init}'")
        init_optim_ids = self.tokenizer(
            self.config.optim_str_init, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        initial_loss = self._compute_candidates_loss(init_optim_ids, image)
        buffer.add(initial_loss.item(), init_optim_ids)
        logger.info(f"Initialized buffer with loss {initial_loss.item():.4f}")
        return buffer

    def get_optim_slicing_indices(self, prompt_ids: Tensor) -> Tuple[int, int]:
        start_anchor_indices = (prompt_ids == self.start_anchor_id).nonzero()
        if len(start_anchor_indices) < 2:
            raise RuntimeError(
                f"Could not locate both start and end anchor tokens ('{self.start_anchor}'). Found {len(start_anchor_indices)}."
            )
        start_idx, end_idx = (
            start_anchor_indices[0].item(),
            start_anchor_indices[1].item(),
        )
        logger.debug(
            f"Found optimization slice indices: start={start_idx}, end={end_idx}"
        )
        return start_idx, end_idx

    def compute_text_gradient(
        self, optim_ids: Tensor, image: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if not self.config.gcg_attack:
            return None, optim_ids

        optim_str = self.tokenizer.decode(
            optim_ids.squeeze(0), skip_special_tokens=True
        )
        prompt_text = self.prompt_template_str.format(optim_str=optim_str)
        logger.debug(f"Computing text gradient for prompt: {prompt_text[:100]}...")

        images_to_process = image if self.config.pgd_attack else None
        inputs = self.processor(
            text=prompt_text, images=images_to_process, return_tensors="pt"
        ).to(self.model.device)
        prompt_ids_batch = inputs["input_ids"]

        start_idx, end_idx = self.get_optim_slicing_indices(prompt_ids_batch.squeeze(0))

        prefix_ids = prompt_ids_batch[:, : start_idx + 1]
        differentiable_tokens = prompt_ids_batch[:, start_idx + 1 : end_idx]
        postfix_ids = prompt_ids_batch[:, end_idx:]
        logger.debug(f"Differentiating {differentiable_tokens.shape[1]} tokens.")

        prefix_embeds = self.embedding_layer(prefix_ids)
        postfix_embeds = self.embedding_layer(postfix_ids)

        one_hot = torch.nn.functional.one_hot(
            differentiable_tokens, num_classes=self.embedding_layer.num_embeddings
        ).to(self.model.dtype)
        one_hot.requires_grad_()

        differentiable_embeds = one_hot @ self.embedding_layer.weight

        prompt_embeds = torch.cat(
            [prefix_embeds, differentiable_embeds, postfix_embeds], dim=1
        )
        logger.debug("Text gradient calculation is unimodal (text-only embeddings).")

        target_embeds = self.embedding_layer(self.target_ids)
        final_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        target_attention_mask = torch.ones_like(self.target_ids)
        final_attention_mask = torch.cat(
            [inputs["attention_mask"], target_attention_mask], dim=1
        )

        outputs = self.model(
            inputs_embeds=final_embeds, attention_mask=final_attention_mask
        )
        logits = outputs.logits

        prompt_len = prompt_embeds.shape[1]
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        shift_labels = self.target_ids.view(-1)

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels
        )
        logger.debug(f"Text gradient loss: {loss.item():.4f}")
        optim_grad = torch.autograd.grad(loss, [one_hot])[0]

        return optim_grad, differentiable_tokens

    def compute_image_gradient(
        self, optim_ids: Tensor, image: Optional[Tensor]
    ) -> Optional[Tensor]:
        if not self.config.pgd_attack or image is None:
            return None

        optim_str = self.tokenizer.decode(
            optim_ids.squeeze(0), skip_special_tokens=True
        )
        prompt_text = self.prompt_template_str.format(optim_str=optim_str)
        logger.debug(f"Computing image gradient for prompt: {prompt_text[:100]}...")

        image.requires_grad_()
        if not image.grad is None:
            image.grad.zero_()

        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(
            self.model.device
        )
        logger.debug("Image is attached to the input for gradient computation.")

        prompt_len = inputs["input_ids"].shape[1]

        full_input_ids = torch.cat([inputs["input_ids"], self.target_ids], dim=1)
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones_like(self.target_ids)], dim=1
        )

        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"],
        )
        logits = outputs.logits

        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        shift_labels = self.target_ids.view(-1)

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels
        )
        logger.debug(f"Image gradient loss: {loss.item():.4f}")
        image_grad = torch.autograd.grad(loss, [image])[0]
        logger.debug(
            f"Computed image gradient. Grad norm: {image_grad.norm().item():.4f}"
        )

        return image_grad

    def perform_pgd_step(
        self,
        image: Tensor,
        eps: float,
        alpha: float,
        image_grad: Optional[Tensor],
        image_original: Tensor,
    ) -> Tensor:
        if image_grad is None:
            return image

        logger.debug(f"Performing PGD step with eps={eps}, alpha={alpha}")
        perturbation = alpha * eps * image_grad.sign()
        perturbed_image = image.detach() - perturbation
        total_perturbation = torch.clamp(perturbed_image - image_original, -eps, eps)
        final_image = torch.clamp(image_original + total_perturbation, 0, 1)

        update_norm = (final_image - image).norm().item()
        total_pert_norm = (final_image - image_original).norm().item()
        logger.debug(
            f"Image updated. Update L2 norm: {update_norm:.4f}. Total perturbation L2 norm: {total_pert_norm:.4f}"
        )

        return final_image

    def generate_test_output(self, optim_ids: Tensor, image: Optional[Tensor]) -> str:
        with torch.no_grad():
            prompt_text = self._get_prompt_texts_for_candidates(optim_ids)[0]
            images_to_process = image if self.config.pgd_attack else None
            logger.info(f"Generating debug output for prompt: {prompt_text[:250]}...")
            if images_to_process is not None:
                logger.info("Attaching current perturbed image to generation.")

            inputs = self.processor(
                text=prompt_text, images=images_to_process, return_tensors="pt"
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs, max_new_tokens=120, do_sample=False
            )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[:, input_len:]
            gen_output = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            logger.info(f"--- DEBUG OUTPUT ---\n{gen_output}\n--------------------")
            return gen_output

    def _log_iteration_summary(
        self, i, loss, best_loss, suffix, total_t, grad_t, sample_t, loss_t, pgd_t
    ):
        summary_header = f" Iteration {i+1}/{self.config.num_steps} Summary "
        log_message = (
            f"\n{summary_header:=^80}"
            f"\n{'Loss':<12} | Current: {loss:<8.4f} | Best: {best_loss:<8.4f}"
            f"\n{'Suffix':<12} | '{suffix.replace(chr(10), ' ')}'"
            f"\n{'Timings (s)':<12} | Total: {total_t:<5.2f} | Grad: {grad_t:<5.2f} | "
            f"Sample: {sample_t:<5.2f} | Loss Eval: {loss_t:<5.2f} | PGD: {pgd_t:<5.2f}"
            f"\n{'='*80}"
        )
        logger.info(log_message)


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
) -> BimodalAttackResult:
    if config is None:
        config = BimodalAttackConfig()
    bimodalattack = BimodalAttack(model, tokenizer, processor, config)
    return bimodalattack.run(messages, goal, target, image)
