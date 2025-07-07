# bimodal_attack.py

import copy
import logging
import time
import gc
import os

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
    configure_pad_token,
)


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
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.decode(ids.squeeze(0), skip_special_tokens=True)
            optim_str = optim_str.replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\nloss: {loss:.4f} | string: {optim_str}"
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
        self.start_anchor = self.tokenizer.eos_token
        self.end_anchor = self.tokenizer.eos_token
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.stop_flag = False

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
            content = content.replace(injection_placeholder, attack_template) if injection_placeholder in content else content + f" {attack_template}"
            messages[last_user_idx]["content"] = content
        elif isinstance(content, list):
            text_part = next((p for p in content if p.get("type") == "text"), None)
            if text_part and injection_placeholder in text_part.get("text", ""):
                text_part["text"] = text_part["text"].replace(injection_placeholder, attack_template)
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

        if config.pgd_attack and image is not None:
            image = self.processor(text="", images=image, return_tensors="pt").to(
                self.model.device
            )["pixel_values"]

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        self.messages_template, self.prompt_template_str = (
            self._prepare_message_template(messages)
        )
        self.target_ids = self.tokenizer(
            target, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        self.initial_goal_text = goal
        self.start_anchor_id = self.tokenizer.convert_tokens_to_ids(self.start_anchor)
        self.end_anchor_id = self.tokenizer.convert_tokens_to_ids(self.end_anchor)

        buffer = self.init_buffer(image)
        optim_ids = buffer.get_best_ids()

        losses, optim_strings, adv_suffixes, model_outputs = [], [], [], []
        gradient_times, sampling_times, loss_times, pgd_times, total_times = (
            [], [], [], [], [],
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

            # GCG Step
            grad_start_time = time.perf_counter()
            optim_ids_onehot_grad, grad_optim_ids = self.compute_text_gradient(optim_ids, image)
            total_grad_time += time.perf_counter() - grad_start_time
            
            sample_start_time = time.perf_counter()
            sampled_ids = self.candidate_sampling(grad_optim_ids, optim_ids_onehot_grad)
            sampling_time = time.perf_counter() - sample_start_time

            # PGD Step
            if config.pgd_attack:
                pgd_grad_start_time = time.perf_counter()
                image_grad = self.compute_image_gradient(optim_ids, image)
                total_grad_time += time.perf_counter() - pgd_grad_start_time
                
                pgd_step_start_time = time.perf_counter()
                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, image_grad, image_original
                )
                current_pgd_time += time.perf_counter() - pgd_step_start_time

            gradient_times.append(total_grad_time)
            sampling_times.append(sampling_time)
            pgd_times.append(current_pgd_time)

            # Evaluation Step
            loss_start_time = time.perf_counter()
            candidate_losses = self._compute_candidates_loss(sampled_ids, image)
            current_loss = candidate_losses.min().item()
            optim_ids = sampled_ids[candidate_losses.argmin()].unsqueeze(0)
            loss_time = time.perf_counter() - loss_start_time
            loss_times.append(loss_time)

            losses.append(current_loss)
            current_optim_str = self.tokenizer.decode(
                optim_ids.squeeze(0), skip_special_tokens=True
            )
            optim_strings.append(current_optim_str)
            adv_suffixes.append(current_optim_str)

            if buffer.size > 0:
                buffer.add(current_loss, optim_ids)

            if current_loss < best_loss:
                best_loss = current_loss
                best_optim_ids = optim_ids.clone()
                if config.pgd_attack:
                    best_image = image.clone()

            gen_output = ""
            if config.debug_output and (i % 10 == 0 or i == config.num_steps - 1):
                gen_output = self.generate_test_output(optim_ids, image)
            model_outputs.append(gen_output)

            iter_total_time = time.perf_counter() - iter_start_time
            total_times.append(iter_total_time)
            self._log_iteration_summary(
                i, current_loss, best_loss, current_optim_str,
                iter_total_time, total_grad_time, sampling_time, loss_time, current_pgd_time,
            )

            if config.pgd_attack:
                save_image(image, os.path.join(config.images_folder, f"{i}.png"))
            if self.stop_flag:
                logger.info("Early stopping triggered.")
                break

        return BimodalAttackResult(
            best_loss=best_loss,
            best_string=self.tokenizer.decode(
                best_optim_ids.squeeze(0), skip_special_tokens=True
            ),
            losses=losses, strings=optim_strings, adversarial_suffixes=adv_suffixes,
            model_outputs=model_outputs, gradient_times=gradient_times,
            sampling_times=sampling_times, loss_times=loss_times,
            pgd_times=pgd_times, total_times=total_times,
        )

    def candidate_sampling(
        self, optim_ids: Tensor, optim_ids_onehot_grad: Optional[Tensor]
    ) -> Tensor:
        if not self.config.gcg_attack or optim_ids_onehot_grad is None:
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
            sampled_ids = filter_ids(sampled_ids, self.tokenizer)

        return sampled_ids if len(sampled_ids) > 0 else optim_ids

    def _get_prompt_texts_for_candidates(self, optim_ids_batch: Tensor) -> List[str]:
        optim_strings = self.tokenizer.batch_decode(
            optim_ids_batch, skip_special_tokens=True
        )
        return [self.prompt_template_str.format(optim_str=s) for s in optim_strings]

    def _compute_candidates_loss(
        self, sampled_ids: Tensor, image: Optional[Tensor]
    ) -> Tensor:
        config = self.config
        starting_batch_size = (
            config.batch_size if config.batch_size is not None else config.search_width
        )

        @find_executable_batch_size(starting_batch_size=starting_batch_size)
        def _compute_loss_in_batches(batch_size: int):
            all_losses = []
            for i in range(0, sampled_ids.shape[0], batch_size):
                ids_batch = sampled_ids[i : i + batch_size]
                current_batch_size = ids_batch.shape[0]
                prompt_texts = self._get_prompt_texts_for_candidates(ids_batch)

                images_to_process = [image] * current_batch_size if config.pgd_attack and image is not None else None

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
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(current_batch_size, -1).mean(dim=-1)
                all_losses.append(loss)

                if config.early_stop and torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)):
                    self.stop_flag = True
                    break
            
            if self.stop_flag:
                # Pad losses for remaining batches if early stopping
                num_processed = len(all_losses) * batch_size
                remaining = sampled_ids.shape[0] - num_processed
                if remaining > 0:
                     all_losses.append(torch.full((remaining,), float("inf"), device=self.model.device))

            return torch.cat(all_losses, dim=0)

        return _compute_loss_in_batches()

    def init_buffer(self, image: Optional[Tensor]) -> AttackBuffer:
        buffer = AttackBuffer(self.config.buffer_size)
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
            raise RuntimeError("Could not locate both start and end anchor tokens.")
        return start_anchor_indices[0].item(), start_anchor_indices[1].item()
    
    def compute_text_gradient(self, optim_ids: Tensor, image: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if not self.config.gcg_attack:
            return None, optim_ids

        optim_str = self.tokenizer.decode(optim_ids.squeeze(0), skip_special_tokens=True)
        prompt_text = self.prompt_template_str.format(optim_str=optim_str)
        
        # Note: We pass the image here to ensure prompt tokenization is consistent, but PGD grad is computed separately.
        images_to_process = image if self.config.pgd_attack else None
        inputs = self.processor(text=prompt_text, images=images_to_process, return_tensors="pt").to(self.model.device)
        prompt_ids_batch = inputs["input_ids"]

        start_idx, end_idx = self.get_optim_slicing_indices(prompt_ids_batch.squeeze(0))
        
        prefix_ids = prompt_ids_batch[:, :start_idx + 1]
        differentiable_tokens = prompt_ids_batch[:, start_idx + 1:end_idx]
        postfix_ids = prompt_ids_batch[:, end_idx:]

        prefix_embeds = self.embedding_layer(prefix_ids)
        postfix_embeds = self.embedding_layer(postfix_ids)

        one_hot = torch.nn.functional.one_hot(
            differentiable_tokens, num_classes=self.embedding_layer.num_embeddings
        ).to(self.model.dtype)
        one_hot.requires_grad_()
        
        differentiable_embeds = one_hot @ self.embedding_layer.weight
        
        # Reconstruct embeddings without real image features for unimodal GCG gradient
        prompt_embeds = torch.cat([prefix_embeds, differentiable_embeds, postfix_embeds], dim=1)
        
        target_embeds = self.embedding_layer(self.target_ids)
        final_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
        
        target_attention_mask = torch.ones_like(self.target_ids)
        final_attention_mask = torch.cat([inputs["attention_mask"], target_attention_mask], dim=1)

        # Forward pass is unimodal (text-only) to get GCG gradient
        outputs = self.model(inputs_embeds=final_embeds, attention_mask=final_attention_mask)
        logits = outputs.logits
        
        prompt_len = prompt_embeds.shape[1]
        shift_logits = logits[:, prompt_len - 1:-1, :].contiguous()
        shift_labels = self.target_ids.view(-1)
        
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels)
        optim_grad = torch.autograd.grad(loss, [one_hot])[0]

        return optim_grad, differentiable_tokens

    def compute_image_gradient(self, optim_ids: Tensor, image: Optional[Tensor]) -> Optional[Tensor]:
        if not self.config.pgd_attack or image is None:
            return None

        optim_str = self.tokenizer.decode(optim_ids.squeeze(0), skip_special_tokens=True)
        prompt_text = self.prompt_template_str.format(optim_str=optim_str)
        
        image.requires_grad_()
        
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(self.model.device)
        
        prompt_len = inputs["input_ids"].shape[1]
        target_len = self.target_ids.shape[1]
        
        full_input_ids = torch.cat([inputs["input_ids"], self.target_ids], dim=1)
        full_attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(self.target_ids)], dim=1)

        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"]
        )
        logits = outputs.logits
        
        shift_logits = logits[:, prompt_len - 1:-1, :].contiguous()
        shift_labels = self.target_ids.view(-1)

        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels)
        image_grad = torch.autograd.grad(loss, [image])[0]
        
        return image_grad

    def perform_pgd_step(
        self, image: Tensor, eps: float, alpha: float, image_grad: Optional[Tensor], image_original: Tensor
    ) -> Tensor:
        if image_grad is None:
            return image
        perturbation = alpha * eps * image_grad.sign()
        perturbed_image = image.detach() - perturbation
        total_perturbation = torch.clamp(perturbed_image - image_original, -eps, eps)
        final_image = torch.clamp(image_original + total_perturbation, 0, 1)
        return final_image

    def generate_test_output(self, optim_ids: Tensor, image: Optional[Tensor]) -> str:
        with torch.no_grad():
            prompt_text = self._get_prompt_texts_for_candidates(optim_ids)[0]
            images_to_process = image if self.config.pgd_attack else None
            inputs = self.processor(
                text=prompt_text, images=images_to_process, return_tensors="pt"
            ).to(self.model.device)
            
            output_ids = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[:, input_len:]
            gen_output = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            logger.info(f"--- DEBUG OUTPUT ---\n{gen_output}\n--------------------")
            return gen_output

    def _log_iteration_summary(
        self, i, loss, best_loss, suffix, total_t, grad_t, sample_t, loss_t, pgd_t
    ):
        summary_msg = (
            f"[Iter {i+1}/{self.config.num_steps}] "
            f"Loss: {loss:.4f} | Best Loss: {best_loss:.4f} | "
            f"Suffix: '{suffix}'\n"
            f"                 Timings (s): Total={total_t:.2f} | Grad={grad_t:.2f} | "
            f"Sample={sample_t:.2f} | Loss Eval={loss_t:.2f} | PGD={pgd_t:.2f}"
        )
        logger.info(summary_msg)

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
    logger.setLevel(getattr(logging, config.verbosity))
    bimodalattack = BimodalAttack(model, tokenizer, processor, config)
    return bimodalattack.run(messages, goal, target, image)