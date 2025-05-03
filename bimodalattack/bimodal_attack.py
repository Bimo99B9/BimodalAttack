import copy
import gc
import logging
import time

from dataclasses import dataclass, field
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
logger = logging.getLogger("bimodalattack")
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
    optim_str_init: Union[str, List[str]] = "x " * 19 + "x"
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
    total_times: List[float] = field(default_factory=list)


# ---------------------------
# AttackBuffer definition
# ---------------------------
class AttackBuffer:
    def __init__(self, size: int):
        self.buffer: List[Tuple[float, Tensor]] = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            # replace the worst
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer: transformers.PreTrainedTokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\").replace("\n", "\\n")
            message += f"\nloss: {loss:.4f} | string: {optim_str}"
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
) -> Tensor:
    """
    Returns `search_width` combinations of token ids based on the token gradient.
    """
    n_optim_tokens = ids.size(0)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices  # (search_width, vocab)

    # Randomly pick positions to replace
    pos = torch.argsort(torch.rand(search_width, n_optim_tokens, device=grad.device))[
        :, :n_replace
    ]  # (search_width, n_replace)

    # For each candidate+pos choose one of the topk
    val = torch.gather(
        topk_ids[pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, pos, val)
    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer) -> Tensor:
    """
    Filters out sequences of token ids that change after retokenization.
    """
    decoded = tokenizer.batch_decode(ids)
    kept = []
    for i, text in enumerate(decoded):
        re_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].to(ids.device)
        if torch.equal(ids[i], re_ids):
            kept.append(ids[i])
    if not kept:
        raise RuntimeError(
            "All candidates changed under re-tokenization; try `filter_ids=False` or new init."
        )
    return torch.stack(kept)


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

        # model metadata logging
        if hasattr(model.config, "model_type"):
            logger.info(f"Model type: {model.config.model_type}")
        if hasattr(model.config, "_name_or_path"):
            logger.info(f"Model id/path: {model.config._name_or_path}")

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model in {model.dtype}, consider lower precision.")
        if model.device == torch.device("cpu"):
            logger.warning("Model on CPU; a GPU will be faster.")

        # ensure chat template
        if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
            if config.pgd_attack:
                logger.warning("No chat_template: adding custom PGD+GCG template.")
                tmpl = "USER: <image>\n{{ messages[0]['content'][0]['text'] }} \nASSISTANT: "
            else:
                logger.warning("No chat_template: adding custom GCG template.")
                tmpl = (
                    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                )
            tokenizer.chat_template = tmpl
            self.processor.chat_template = tmpl

    def run(
        self,
        messages: Union[str, List[dict]],
        goal: str,
        target: str,
        image: torch.Tensor = None,
    ) -> BimodalAttackResult:
        model, tokenizer, config = self.model, self.tokenizer, self.config
        self.initial_prompt = goal

        os.makedirs(config.images_folder, exist_ok=True)

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        # normalize messages to list-of-dicts
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
        logger.info(f"Messages[0]: {messages}")

        # ensure the "{optim_str}" placeholder
        if (
            isinstance(messages[-1]["content"], str)
            and "{optim_str}" not in messages[-1]["content"]
        ):
            messages[-1]["content"] += " {optim_str}"
            logger.info(f"Appended optim placeholder: {messages}")

        # if PGD, ensure an image token is present
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
        logger.info(f"Messages w/ image: {messages}")

        # build the chat prompt
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        logger.info(f"Full prompt: {prompt}")

        # strip BOS if present
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token) :]
        logger.info(f"Prompt sans BOS: {prompt}")

        # tokenize before/after segments
        if config.pgd_attack:
            # split around optim_str & image tokens
            if self.processor.__class__.__name__ == "Gemma3Processor":
                before, after_temp = prompt.split("{optim_str}", 1)
                before_img_str = before.strip()
                if "<start_of_image>" not in after_temp:
                    raise ValueError("Expected <start_of_image> in Gemma3 prompt")
                suffix_part, sep, after_str = after_temp.partition("<start_of_image>")
                before_suffix_str = (suffix_part + sep).strip()
            else:
                if "<start_of_image>" in prompt:
                    before_img_str, rest = prompt.split("<start_of_image>", 1)
                elif "<image>" in prompt:
                    before_img_str, rest = prompt.split("<image>", 1)
                else:
                    raise ValueError("No image token for PGD attack")
                before_suffix_str, after_str = rest.split("{optim_str}", 1)

            logger.info(f"Before_img: {before_img_str}")
            logger.info(f"Before_suffix: {before_suffix_str}")
            logger.info(f"After: {after_str}; Target: {target}")

            # encode to ids
            bimg_ids = tokenizer(before_img_str, return_tensors="pt")["input_ids"].to(
                model.device
            )
            bsuf_ids = tokenizer(before_suffix_str, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            tgt_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device)

            # save both ids & embeddings
            self.before_img_ids, self.before_suffix_ids = bimg_ids, bsuf_ids
            self.after_ids, self.target_ids = after_ids, tgt_ids
            (
                self.before_img_embeds,
                self.before_suffix_embeds,
                self.after_embeds,
                self.target_embeds,
            ) = [
                self.embedding_layer(x)
                for x in (bimg_ids, bsuf_ids, after_ids, tgt_ids)
            ]
        else:
            before_str, after_str = prompt.split("{optim_str}")
            logger.info(f"Before: {before_str}")
            logger.info(f"After: {after_str}; Target: {target}")
            b_ids = tokenizer(before_str, return_tensors="pt")["input_ids"].to(
                model.device
            )
            after_ids = tokenizer(
                after_str, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            tgt_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device)

            self.before_embeds, self.after_embeds, self.target_embeds = [
                self.embedding_layer(x) for x in (b_ids, after_ids, tgt_ids)
            ]
            self.target_ids = tgt_ids

        # initialize buffer + starting candidate
        buffer = self.init_buffer(image)
        optim_ids = buffer.get_best_ids()

        # metric containers
        losses, opt_strs, adv_sufs, outs = [], [], [], []
        grad_ts, samp_ts, loss_ts, pgd_ts, total_ts = [], [], [], [], []
        tg, ts, tl, tp = 0.0, 0.0, 0.0, 0.0
        best_loss = float("inf")
        best_ids, best_img = None, None

        if config.pgd_attack:
            logger.warning(f"PGD α={config.alpha}, ε={config.eps}")
            image.requires_grad_(True)
            img_orig = image.clone()

        # log mode
        if config.pgd_attack and config.gcg_attack:
            logger.info("Running PGD then GCG")
        elif config.pgd_attack:
            logger.info("Running PGD only")
        else:
            logger.info("Running GCG only")

        for i in tqdm(range(config.num_steps)):
            logger.info(
                f"[Iter {i}] Start (pgd={config.pgd_attack}, gcg={config.gcg_attack})"
            )
            # Phase A: gradient
            t0 = time.perf_counter()
            if config.pgd_attack:
                grad_onehot, img_grad = self.compute_gradient(optim_ids, image)
            else:
                grad_onehot, _ = self.compute_gradient(optim_ids)
            dg = time.perf_counter() - t0
            grad_ts.append(dg)
            tg += dg
            logger.info(f"[{i}] Phase A (grad) {dg:.4f}s")

            # Phase B & C: PGD before GCG
            if config.pgd_attack:
                logger.info(f"[{i}] Phase B (PGD)")
                t1 = time.perf_counter()
                image = self.perform_pgd_step(
                    image, config.eps, config.alpha, img_grad, img_orig
                )
                dp = time.perf_counter() - t1
                pgd_ts.append(dp)
                tp += dp
                logger.info(f"[{i}] Phase B {dp:.4f}s")

                if config.gcg_attack and not config.joint_eval:
                    logger.info(f"[{i}] Phase C (re-grad)")
                    t2 = time.perf_counter()
                    grad_onehot, img_grad = self.compute_gradient(optim_ids, image)
                    dg2 = time.perf_counter() - t2
                    grad_ts.append(dg2)
                    tg += dg2
                    logger.info(f"[{i}] Phase C {dg2:.4f}s")
            else:
                dp = 0.0
                logger.info(f"[{i}] Skipping PGD")

            # Phase D: GCG sampling & partial eval
            logger.info(f"[{i}] Phase D (sampling)")
            sampled_ids, w2, ds, ts = self.candidate_sampling(
                config, i, optim_ids, grad_onehot, tokenizer, samp_ts, ts
            )
            samp_ts.append(ds)
            ts += ds
            logger.info(f"[{i}] Sampled {w2} in {ds:.4f}s")

            with torch.no_grad():
                t3 = time.perf_counter()
                bs = w2 if config.batch_size is None else config.batch_size

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

                    # pre-image selection
                    if config.joint_eval:
                        cand_emb = self._build_input_embeds_pgd(
                            sampled_ids, img_feats, w2, single=True
                        )
                    elif config.gcg_attack:
                        cand_emb = self._build_input_embeds_gcg(
                            sampled_ids, w2, single=True
                        )
                    else:
                        cand_emb = None

                    if cand_emb is not None:
                        lv = find_executable_batch_size(
                            self._compute_candidates_loss_original, bs
                        )(cand_emb)
                        best_pre = lv.min().item()
                        bi = lv.argmin()
                    else:
                        best_pre, bi = 0.0, 0
                    logger.info(f"[{i}] best pre-image loss {best_pre:.4f}")

                    # full evaluation
                    full_emb = self._build_input_embeds_gcg_pgd(
                        sampled_ids[bi].unsqueeze(0), img_feats
                    )
                    cl = self._compute_candidates_loss_original(1, full_emb).item()
                    optim_ids = sampled_ids[bi].unsqueeze(0)
                    losses.append(cl)
                    s = tokenizer.batch_decode(optim_ids)[0]
                    opt_strs.append(s)
                    if buffer.size == 0 or cl < buffer.get_highest_loss():
                        buffer.add(cl, optim_ids)
                    if cl < best_loss:
                        best_loss, best_ids, best_img = (
                            cl,
                            optim_ids.clone(),
                            image.clone(),
                        )
                    logger.info(f"[{i}] Final loss {cl:.4f}")
                else:
                    # GCG-only
                    cand_emb = self._build_input_embeds_gcg(
                        sampled_ids, w2, no_joint_eval=True
                    )
                    lv = find_executable_batch_size(
                        self._compute_candidates_loss_original, bs
                    )(cand_emb)
                    bi = lv.argmin()
                    cl = lv.min().item()
                    optim_ids = sampled_ids[bi].unsqueeze(0)
                    losses.append(cl)
                    opt_strs.append(tokenizer.batch_decode(optim_ids)[0])
                    if buffer.size == 0 or cl < buffer.get_highest_loss():
                        buffer.add(cl, optim_ids)
                    if cl < best_loss:
                        best_loss, best_ids = cl, optim_ids.clone()
                    logger.info(f"[{i}] GCG-only loss {cl:.4f}")

                dl = time.perf_counter() - t3
                loss_ts.append(dl)
                tl += dl
                logger.info(f"[{i}] Phase D (loss) {dl:.4f}s")

            # save image
            if config.pgd_attack:
                self._save_image(image, os.path.join(config.images_folder, f"{i}.png"))

            # optional debug generation
            if config.debug_output and i % 10 == 0:
                with torch.no_grad():
                    if config.pgd_attack:
                        px = self.normalize(image)
                        if self.processor.__class__.__name__ == "Gemma3Processor":
                            feats = model.get_image_features(pixel_values=px)
                        else:
                            feats = model.get_image_features(
                                pixel_values=px,
                                vision_feature_layer=-2,
                                vision_feature_select_strategy="default",
                            )
                        ib = self._build_input_embeds_gcg_pgd(
                            sampled_ids, feats, w2, no_target=True
                        )
                    else:
                        ib = self._build_input_embeds_gcg(
                            sampled_ids, w2, no_target=True
                        )
                    gen = model.generate(inputs_embeds=ib, max_new_tokens=120)
                    out = tokenizer.decode(gen[0], skip_special_tokens=True)
                    logger.info(f"[{i}] debug gen: {out}")
            else:
                out = ""
            outs.append(out)
            adv_sufs.append(tokenizer.batch_decode(optim_ids)[0])
            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping on perfect match.")
                break

            total_i = dg + ds + dp + dl
            total_ts.append(total_i)
            logger.info(
                f"[{i}] Total {total_i:.4f}s (G:{dg:.4f}, S:{ds:.4f}, P:{dp:.4f}, L:{dl:.4f})"
            )

        iters = i + 1
        logger.warning(f"Avg grad: {tg/iters:.4f}s")
        logger.warning(f"Avg PGD: {tp/iters:.4f}s")
        logger.warning(f"Avg samp: {ts/iters:.4f}s")
        logger.warning(f"Avg loss: {tl/iters:.4f}s")

        mi = losses.index(min(losses))
        return BimodalAttackResult(
            best_loss=losses[mi],
            best_string=opt_strs[mi],
            losses=losses,
            strings=opt_strs,
            adversarial_suffixes=adv_sufs,
            model_outputs=outs,
            gradient_times=grad_ts,
            sampling_times=samp_ts,
            loss_times=loss_ts,
            pgd_times=pgd_ts,
            total_times=total_ts,
        )

    def init_buffer(self, image) -> AttackBuffer:
        model, tokenizer, config = self.model, self.tokenizer, self.config
        logger.info(f"Initializing buffer, size={config.buffer_size}…")
        buffer = AttackBuffer(config.buffer_size)

        # build initial IDs
        if isinstance(config.optim_str_init, str):
            init_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                pool = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                idx = torch.randint(
                    0, pool.size(0), (config.buffer_size - 1, init_ids.size(1))
                )
                init_ids = torch.cat([init_ids, pool[idx]], dim=0)
        else:
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning("Mismatch init-list vs buffer_size")
            init_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)

        true_bs = max(1, config.buffer_size)

        # embed
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
            emb = self._build_input_embeds_gcg_pgd(
                init_ids, img_feats, search_width=true_bs, single=True
            )
        else:
            emb = self._build_input_embeds_gcg(
                init_ids, search_width=true_bs, no_joint_eval=True
            )

        losses0 = find_executable_batch_size(
            self._compute_candidates_loss_original, true_bs
        )(emb)
        for j in range(true_bs):
            buffer.add(losses0[j].item(), init_ids[[j]])
        buffer.log_buffer(tokenizer)
        logger.info("Buffer initialization complete.")
        return buffer

    def candidate_sampling(
        self,
        config: BimodalAttackConfig,
        i: int,
        optim_ids: Tensor,
        grad_onehot: Tensor,
        tokenizer,
        sampling_times: List[float],
        total_sampling_time: float,
    ) -> Tuple[Tensor, int, float, float]:
        if config.dynamic_search:
            cw = max(
                config.min_search_width,
                int(config.search_width * (1 - i / config.num_steps)),
            )
            logger.info(f"[{i}] dynamic_search → {cw}")
        else:
            cw = config.search_width

        if config.gcg_attack:
            t0 = time.perf_counter()
            sids = sample_ids_from_grad(
                optim_ids.squeeze(0),
                grad_onehot.squeeze(0),
                cw,
                config.topk,
                config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )
            if config.filter_ids:
                sids = filter_ids(sids, tokenizer)
            nw = sids.size(0)
            dt = time.perf_counter() - t0
            sampling_times.append(dt)
            total_sampling_time += dt
        else:
            sids, nw, dt = optim_ids, 1, 0.0

        return sids, nw, dt, total_sampling_time

    def compute_gradient(
        self, optim_ids: Tensor, image: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        model = self.model
        emb_layer = self.embedding_layer

        onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=emb_layer.num_embeddings
        )
        onehot = onehot.to(model.device, model.dtype)
        if self.config.gcg_attack:
            onehot.requires_grad_()
        else:
            onehot.requires_grad_(False)

        token_embeds = onehot @ emb_layer.weight  # (B, L, D)

        if self.config.pgd_attack:
            px = self.normalize(image)
            if self.processor.__class__.__name__ == "Gemma3Processor":
                img_feats = model.get_image_features(pixel_values=px)
            else:
                img_feats = model.get_image_features(
                    pixel_values=px,
                    vision_feature_layer=-2,
                    vision_feature_select_strategy="default",
                )
            inp = torch.cat(
                [
                    self.before_img_embeds,
                    img_feats,
                    self.before_suffix_embeds,
                    token_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
        else:
            inp = torch.cat(
                [
                    self.before_embeds,
                    token_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

        logits = model(inputs_embeds=inp).logits
        shift = inp.size(1) - self.target_ids.size(1)
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()
        labels = self.target_ids

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        if self.config.pgd_attack:
            if self.config.gcg_attack:
                grad_onehot, grad_img = torch.autograd.grad(loss, (onehot, image))
            else:
                grad_img = torch.autograd.grad(loss, image)[0]
                grad_onehot = None
            return grad_onehot, grad_img
        else:
            grad_onehot = (
                torch.autograd.grad(loss, onehot)[0] if self.config.gcg_attack else None
            )
            return grad_onehot, None

    def perform_pgd_step(
        self,
        image: Tensor,
        eps: float,
        alpha: float,
        img_grad: Tensor,
        img_orig: Tensor,
    ) -> Tensor:
        adv = (image - alpha * eps * torch.sign(img_grad)).detach().requires_grad_()
        adv = torch.clamp(adv, img_orig - eps, img_orig + eps).clamp(0, 1)
        return adv

    def perform_autopgd_step(
        self,
        image: Tensor,
        eps: float,
        img_grad: Tensor,
        img_orig: Tensor,
        current_loss: Optional[float],
        iter_idx: int,
    ) -> Tensor:
        # (unchanged from working commit)
        alpha = 0.75
        checkpoint = 10
        rho = 0.75
        if not hasattr(self, "pgd_state_initialized"):
            self.pgd_prev_image = image.clone()
            self.pgd_best_image = image.clone()
            self.pgd_best_loss = current_loss or float("inf")
            self.pgd_current_eta = 2 * eps
            self.pgd_improvement_count = 0
            self.pgd_last_best_loss = self.pgd_best_loss
            self.pgd_state_initialized = True

        grad_sign = torch.sign(img_grad)
        z = image - self.pgd_current_eta * grad_sign
        z = torch.max(torch.min(z, img_orig + eps), img_orig - eps).clamp(0, 1)
        new_img = (
            image + alpha * (z - image) + (1 - alpha) * (image - self.pgd_prev_image)
        )
        new_img = torch.max(torch.min(new_img, img_orig + eps), img_orig - eps).clamp(
            0, 1
        )
        self.pgd_prev_image = image.clone()

        if current_loss is not None:
            if current_loss < self.pgd_best_loss:
                self.pgd_best_loss = current_loss
                self.pgd_best_image = new_img.clone()
            if current_loss < self.pgd_last_best_loss:
                self.pgd_improvement_count += 1
            if (iter_idx + 1) % checkpoint == 0:
                frac = self.pgd_improvement_count / checkpoint
                if frac < rho or self.pgd_best_loss == self.pgd_last_best_loss:
                    self.pgd_current_eta /= 2
                    new_img = self.pgd_best_image.clone()
                    self.pgd_prev_image = self.pgd_best_image.clone()
                self.pgd_improvement_count = 0
                self.pgd_last_best_loss = self.pgd_best_loss

        return new_img

    def _build_input_embeds_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width: int,
        single: bool = False,
    ) -> Tensor:
        mt = self.model.config.model_type
        if single:
            if mt == "gemma3":
                parts = [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ]
            else:
                parts = [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ]
            return torch.cat(parts, dim=1)
        else:
            raise ValueError("PGD mode only supports single=True")

    def _build_input_embeds_gcg(
        self,
        sampled_ids: Tensor,
        search_width: int,
        single: bool = False,
        no_joint_eval: bool = False,
        no_target: bool = False,
    ) -> Tensor:
        mt = self.model.config.model_type
        if single:
            if mt == "gemma3":
                parts = [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ]
            else:
                parts = [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                    self.target_embeds.repeat(search_width, 1, 1),
                ]
            return torch.cat(parts, dim=1)

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
        raise ValueError("Invalid flags for GCG embed builder")

    def _build_input_embeds_gcg_pgd(
        self,
        sampled_ids: Tensor,
        image: Tensor,
        search_width: Optional[int] = None,
        single: bool = False,
        no_target: bool = False,
    ) -> Tensor:
        mt = self.model.config.model_type
        if single:
            return self._build_input_embeds_pgd(
                sampled_ids, image, search_width or 1, single=True
            )

        if no_target:
            # same as single but without the final target_embeds
            parts = (
                [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.after_embeds.repeat(search_width, 1, 1),
                ]
                if mt == "gemma3"
                else [
                    self.before_img_embeds.repeat(search_width, 1, 1),
                    image.repeat(search_width, 1, 1),
                    self.before_suffix_embeds.repeat(search_width, 1, 1),
                    self.embedding_layer(sampled_ids),
                    self.after_embeds.repeat(search_width, 1, 1),
                ]
            )
            return torch.cat(parts, dim=1)

        # full batch of size>1
        parts = (
            [
                self.before_img_embeds,
                self.embedding_layer(sampled_ids),
                self.before_suffix_embeds,
                image,
                self.after_embeds,
                self.target_embeds,
            ]
            if mt == "gemma3"
            else [
                self.before_img_embeds,
                image,
                self.before_suffix_embeds,
                self.embedding_layer(sampled_ids),
                self.after_embeds,
                self.target_embeds,
            ]
        )
        return torch.cat(parts, dim=1)

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

    def _save_image(self, image: Tensor, path: str):
        arr = (image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        Image.fromarray(arr).save(path)


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
    attacker = BimodalAttack(model, tokenizer, processor, config, normalize)
    return attacker.run(messages, goal, target, image)
