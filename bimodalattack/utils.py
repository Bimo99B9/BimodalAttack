# utils.py
import functools
import gc
import inspect
import torch
import transformers
from torch import Tensor
from transformers import PreTrainedTokenizerBase
import numpy as np
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger("bimodalattack")

INIT_CHARS = [
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "w",
    "x",
    "y",
    "z",
]


def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = [
        i for i in range(tokenizer.vocab_size) if not is_ascii(tokenizer.decode([i]))
    ]

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)
    
    logger.debug(f"Found {len(nonascii_toks)} non-ASCII or special tokens to disallow.")
    return torch.tensor(nonascii_toks, device=device)


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if an exception is related to resource limitations (e.g., OOM)
    that can be resolved by reducing the batch size.
    """
    _statements = [
        # Standard CUDA OOM errors
        "CUDA out of memory.",
        "DefaultCPUAllocator: can't allocate memory",
        # CUDNN errors
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        # New error for 32-bit index math limitations
        "Expected canUse32BitIndexMath",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Sets the pad token for a tokenizer if it is not already set."""
    if tokenizer.pad_token:
        return tokenizer

    logger.warning("Tokenizer does not have a pad token. Trying to set one.")
    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        logger.info(f"Set pad_token_id to unk_token_id: {tokenizer.unk_token_id}")
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    else:
        logger.info("Adding a new pad token: <|pad|>")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
) -> Tensor:
    """Samples a batch of new token ids from the gradient."""
    grad = grad.cpu()
    ids = ids.cpu()

    if torch.isnan(grad).any() or torch.isinf(grad).any():
        logger.warning("NaN or Inf found in gradient tensor. Sanitizing.")
        grad = torch.nan_to_num(grad, nan=-torch.inf)

    L, V = grad.shape
    scores = torch.gather(grad, 1, ids.unsqueeze(1)).squeeze(1)
    top_indices = torch.topk(grad, topk, dim=-1).indices
    top_scores = torch.gather(grad, 1, top_indices)
    new_ids = top_indices.flatten()
    new_scores = (top_scores - scores.unsqueeze(1)).flatten()

    if not_allowed_ids is not None:
        mask = torch.isin(new_ids, not_allowed_ids.to(new_ids.device))
        new_ids = new_ids[~mask]
        new_scores = new_scores[~mask]

    num_samples = min(search_width, new_scores.shape[0])
    if num_samples == 0:
        logger.warning("No valid candidate tokens found after filtering. Returning original IDs.")
        return ids.unsqueeze(0).to(grad.device)

    logger.debug(f"Sampling {num_samples} new candidates from {new_scores.shape[0]} potential replacements (top-k={topk}).")
    sampled_indices = torch.multinomial(
        torch.softmax(new_scores, dim=0),
        num_samples,
        replacement=False,
    )

    sampled_ids = ids.repeat(num_samples, 1)
    rows_to_update = torch.arange(num_samples)
    cols_to_update = sampled_indices // topk
    new_token_ids = new_ids[sampled_indices]

    sampled_ids[rows_to_update, cols_to_update] = new_token_ids

    return sampled_ids.to(grad.device)


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out token sequences that are not stable after re-tokenization."""
    logger.debug(f"Filtering {len(ids)} candidate IDs for tokenization stability.")
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []
    
    original_device = ids.device

    for i, text in enumerate(ids_decoded):
        current_text = str(text)
        ids_encoded = tokenizer(current_text, return_tensors="pt", add_special_tokens=False).to(
            original_device
        )["input_ids"][0]
        
        original_ids = ids[i]
        
        if torch.equal(original_ids[:len(ids_encoded)], ids_encoded):
            filtered_ids.append(original_ids)

    if not filtered_ids:
        logger.error(
            "No token sequences are the same after decoding and re-encoding. "
            "This can happen with complex tokens. Consider setting filter_ids=False or using a simpler init string."
        )
        return ids

    return torch.stack(filtered_ids)


def save_image(image, path):
    """Saves a tensor image to a file."""
    logger.debug(f"Saving image to {path}")
    image = image.squeeze(0).detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(path)