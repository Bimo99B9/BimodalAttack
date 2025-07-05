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

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
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
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join(
                [f"{arg}={value}" for arg, value in zip(params[1:], args[1:])]
            )
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
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
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator


def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.

    Borrowed from https://github.com/EleutherAI/lm-evaluation-harness/blob/5c006ed417a2f4d01248d487bcbd493ebe3e5edd/lm_eval/models/utils.py#L624
    """
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
    """
    Samples a batch of new token ids from the gradient.
    """
    grad = grad.cpu()
    ids = ids.cpu()

    # === FIX: Sanitize the gradient tensor to prevent invalid indices ===
    # Replace any NaN or Inf values with a large negative number
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("Warning: NaN or Inf found in gradient tensor. Sanitizing.")
        grad = torch.nan_to_num(grad, nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)
    # =================================================================

    L, V = grad.shape
    scores = torch.gather(grad, 1, ids.unsqueeze(1)).squeeze(1)
    top_indices = torch.topk(grad, topk, dim=-1).indices
    top_scores = torch.gather(grad, 1, top_indices)
    new_ids = top_indices.flatten()
    new_scores = (top_scores - scores.unsqueeze(1)).flatten()

    if not_allowed_ids is not None:
        # filter out not allowed ids
        # --- FIX STARTS HERE ---
        # Ensure both tensors are on the same device (CPU) before comparison
        mask = torch.isin(new_ids, not_allowed_ids.to(new_ids.device))
        # --- FIX ENDS HERE ---
        new_ids = new_ids[~mask]
        new_scores = new_scores[~mask]

    # Sample without replacement
    # TODO: this can be slow
    sampled_indices = torch.multinomial(
        torch.softmax(new_scores, dim=0),
        min(search_width, new_scores.shape[0]),
        replacement=False,
    )
    # (search_width, L)
    sampled_ids = torch.zeros(
        (sampled_indices.shape[0], L), dtype=torch.long
    )
    for i in range(sampled_indices.shape[0]):
        # The index of the token to replace
        idx_to_replace = sampled_indices[i] // topk
        # The new token id
        new_id = new_ids[sampled_indices[i]]
        new_ids_i = ids.clone()
        new_ids_i[idx_to_replace] = new_id
        sampled_ids[i] = new_ids_i

    return sampled_ids.to(ids.device)


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


def save_image(image, path):
    image = image.squeeze(0).detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(path)
