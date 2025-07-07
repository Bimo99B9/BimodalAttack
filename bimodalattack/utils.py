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


def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128
):
    """
    A decorator that automatically finds an executable batch size by halving it
    on resource-related runtime errors.
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size, starting_batch_size=starting_batch_size
        )

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(
                        f"Resource error caught. Decreasing batch size to: {batch_size}"
                    )
                else:
                    raise

    return decorator


def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Sets the pad token for a tokenizer if it is not already set."""
    if tokenizer.pad_token:
        return tokenizer

    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
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
        print("Warning: NaN or Inf found in gradient tensor. Sanitizing.")
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
        return ids.unsqueeze(0).to(grad.device)

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
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i, text in enumerate(ids_decoded):
        ids_encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            ids.device
        )["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting filter_ids=False or trying a different optim_str_init."
        )

    return torch.stack(filtered_ids)


def save_image(image, path):
    """Saves a tensor image to a file."""
    image = image.squeeze(0).detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(path)
