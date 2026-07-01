"""
Generic OpenAI-compatible Vision-Language Model interface for VG-GUI-Bench.

This module provides a single, fully reproducible way to call any
Vision-Language Model (VLM) that exposes an OpenAI-compatible
``/chat/completions`` endpoint. This covers:

* The official OpenAI API (e.g. ``gpt-4o``, ``gpt-4.1``, ``gpt-5``);
* Azure OpenAI (set ``OPENAI_BASE_URL`` to your Azure endpoint);
* Any open-source model served locally through an OpenAI-compatible server
  such as `vLLM`, `SGLang` or `LMDeploy`
  (e.g. ``python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-7B-Instruct``).

Everything is configured through standard environment variables (recommended)
or explicit arguments, so results can be reproduced by anyone::

    export OPENAI_API_KEY="sk-..."                       # your API key
    export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your own server
    export OPENAI_MODEL="gpt-4o"                          # model name / path

Usage::

    from api.model import build_model

    model = build_model()                       # everything from env vars
    model = build_model(model_name="gpt-4o")     # explicit model name

    response = model(
        img_path_or_list=["step1.png", "target.png"],
        question="What is the next action?",
        system_prompt="You are a GUI agent.",
        image_first=True,
    )
"""

import os
import base64
import time
import logging
from typing import List, Optional, Union

import requests

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `openai` package is required. Install it with `pip install openai`."
    ) from e


# ============================================================================
# Image encoding
# ============================================================================

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image(image_path: str) -> Optional[str]:
    """Encode an image into a base64 ``data:`` URL.

    Accepts a local file path or an ``http(s)`` URL. Returns ``None`` on
    failure so callers can decide how to handle a missing/invalid image.
    """
    try:
        if isinstance(image_path, str) and image_path.startswith("http"):
            content = requests.get(image_path, timeout=30).content
            ext = os.path.splitext(image_path.split("?")[0])[1].lower()
            mime = _MIME_MAP.get(ext, "image/png")
        else:
            if not os.path.exists(image_path):
                logging.warning(f"[model] Image not found: {image_path}")
                return None
            with open(image_path, "rb") as f:
                content = f.read()
            ext = os.path.splitext(image_path)[1].lower()
            mime = _MIME_MAP.get(ext, "image/png")

        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:  # noqa: BLE001
        logging.warning(f"[model] Failed to encode image {image_path}: {e}")
        return None


# ============================================================================
# OpenAI-compatible model
# ============================================================================

class OpenAICompatibleModel:
    """A thin, reproducible wrapper around any OpenAI-compatible VLM endpoint.

    The ``__call__`` signature is kept identical to the one used throughout the
    benchmark so it can be dropped in anywhere a model callable is expected::

        model(img_path_or_list=None, question='', img_tnx=None,
              image_first=False, system_prompt=None) -> str
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_try: int = 3,
        timeout: float = 300.0,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        extra_body: Optional[dict] = None,
    ):
        """
        Args:
            model_name: Model name/path served by the endpoint. Falls back to
                the ``OPENAI_MODEL`` environment variable.
            api_key: API key. Falls back to ``OPENAI_API_KEY`` (use any
                non-empty placeholder such as ``"EMPTY"`` for local servers
                that do not check keys).
            base_url: OpenAI-compatible base URL. Falls back to
                ``OPENAI_BASE_URL`` (defaults to the official OpenAI endpoint).
            max_try: Maximum number of retries per request.
            timeout: Per-request timeout in seconds.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            extra_body: Optional extra parameters forwarded to the server
                (e.g. ``{"top_p": 0.8}``).
        """
        self.model_name = model_name or os.environ.get("OPENAI_MODEL")
        if not self.model_name:
            raise ValueError(
                "No model name provided. Pass `model_name=` or set the "
                "`OPENAI_MODEL` environment variable."
            )

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.max_try = max_try
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logging.info(
            f"[model] Initialised OpenAICompatibleModel "
            f"(model={self.model_name}, base_url={self.base_url})"
        )

    def __call__(
        self,
        img_path_or_list: Union[str, List[str], None] = None,
        question: str = "",
        img_tnx=None,  # kept for interface compatibility (unused)
        image_first: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        if question == "":
            raise ValueError("Question cannot be empty")

        # ---- 1. Encode images ----
        image_urls: List[str] = []
        if img_path_or_list:
            if isinstance(img_path_or_list, (list, tuple)):
                for single_img in img_path_or_list:
                    encoded = encode_image(single_img)
                    if encoded is None:
                        return ""
                    image_urls.append(encoded)
            elif isinstance(img_path_or_list, str):
                encoded = encode_image(img_path_or_list)
                if encoded is None:
                    return ""
                image_urls = [encoded]
            else:
                raise TypeError(
                    f"img_path_or_list has invalid type: {type(img_path_or_list)}"
                )

        # ---- 2. Build OpenAI-format messages ----
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if not image_urls:
            user_content.append({"type": "text", "text": question})
        elif image_first:
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
            user_content.append({"type": "text", "text": question})
        else:
            user_content.append({"type": "text", "text": question})
            for url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": "user", "content": user_content})

        # ---- 3. Call the endpoint with retries ----
        num_request = 0
        while num_request < self.max_try:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **self.extra_body,
                )
                return completion.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                num_request += 1
                logging.warning(
                    f"[model] API call failed (attempt {num_request}/{self.max_try}): {e}"
                )
                if num_request >= self.max_try:
                    logging.error("[model] Exceeded maximum retries, returning empty string.")
                    return ""
                time.sleep(2 * num_request)
        return ""

    def __repr__(self):
        return f"OpenAICompatibleModel(model_name='{self.model_name}', base_url='{self.base_url}')"


# ============================================================================
# Factory
# ============================================================================

def build_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_try: int = 3,
    timeout: float = 300.0,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    extra_body: Optional[dict] = None,
) -> OpenAICompatibleModel:
    """Build an :class:`OpenAICompatibleModel`.

    All arguments are optional; anything omitted is read from environment
    variables (``OPENAI_MODEL`` / ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``).

    Examples::

        # Official OpenAI
        export OPENAI_API_KEY=sk-...
        model = build_model(model_name="gpt-4o")

        # Local vLLM server
        export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
        export OPENAI_API_KEY=EMPTY
        model = build_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    """
    return OpenAICompatibleModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        max_try=max_try,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
