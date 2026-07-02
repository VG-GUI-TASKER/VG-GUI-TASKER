"""
Multimodal API utilities for the VLM-based VideoQA evaluation.

Wraps the generic, reproducible :class:`OpenAICompatibleModel` (see ``model.py``)
so that every evaluation script can talk to any OpenAI-compatible endpoint
(OpenAI, Azure OpenAI, or a self-hosted vLLM / SGLang / LMDeploy server).

Configure the endpoint through environment variables::

    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o-2024-11-20"
    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"   # optional
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

# Make sure this directory is importable so that ``model`` can be found.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model import build_model


# ============================================================
#  Global model instance (singleton, thread-safe)
# ============================================================

_model_instance = None
_model_lock = threading.Lock()


def get_model(max_try: int = 3, timeout: float = 120.0, max_tokens: int = 1024, temperature: float = 0.0):
    """Return a process-wide :class:`OpenAICompatibleModel` singleton.

    The endpoint is read from the ``OPENAI_MODEL`` / ``OPENAI_API_KEY`` /
    ``OPENAI_BASE_URL`` environment variables.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = build_model(
                    max_try=max_try,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
    return _model_instance


# ============================================================
#  Call interface
# ============================================================

def call_vlm(
    question: str,
    image_paths: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> Optional[str]:
    """Query the VLM with an optional list of images.

    Args:
        question: The text prompt.
        image_paths: List of local image paths (may be empty for text-only).
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum number of output tokens.
        max_retries: Number of retries handled by the underlying model.

    Returns:
        The model reply, or ``None`` on failure.
    """
    model = get_model(max_try=max_retries, max_tokens=max_tokens, temperature=temperature)
    try:
        if image_paths and len(image_paths) > 0:
            response = model(
                img_path_or_list=image_paths,
                question=question,
                system_prompt=system_prompt,
                image_first=True,
            )
        else:
            response = model(
                img_path_or_list=None,
                question=question,
                system_prompt=system_prompt,
            )
        if response is None or response == "":
            return None
        return response
    except Exception as e:  # noqa: BLE001
        print(f"  [API Error] {e}")
        return None


# Backward-compatible alias used throughout the evaluation scripts.
call_qwen_vl = call_vlm


def call_vlm_batch(
    tasks: List[Dict[str, Any]],
    max_workers: int = 16,
    progress_bar: bool = True,
) -> List[Optional[str]]:
    """Run :func:`call_vlm` over a list of tasks in parallel.

    Each task is a dict with keys ``question``, ``image_paths`` and optional
    ``system_prompt`` / ``temperature`` / ``max_tokens``. Results preserve the
    input order.
    """
    results: List[Optional[str]] = [None] * len(tasks)

    pbar = None
    if progress_bar:
        from tqdm import tqdm
        pbar = tqdm(total=len(tasks), desc="API Calls")

    def process_task(idx, task):
        result = call_vlm(
            question=task["question"],
            image_paths=task["image_paths"],
            system_prompt=task.get("system_prompt"),
            temperature=task.get("temperature", 0.0),
            max_tokens=task.get("max_tokens", 1024),
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_task, idx, task): idx
            for idx, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


# Backward-compatible alias.
call_qwen_vl_batch = call_vlm_batch
