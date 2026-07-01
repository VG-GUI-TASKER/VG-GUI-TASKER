"""
Minimal example showing how to call a Vision-Language Model through the
generic OpenAI-compatible interface (see ``api/model.py``).

Configure the endpoint with environment variables, then run this file::

    # Official OpenAI
    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o"

    # ...or a local vLLM server hosting an open-source VLM
    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

    python -m api.use_api
"""

import concurrent.futures

from api.model import build_model


def batch_api_call(model, tasks, num_threads=16):
    """Run multiple model calls concurrently.

    Args:
        model: A callable model built with ``build_model()``.
        tasks: A list of kwargs dicts forwarded to ``model()``, e.g.::

            [
                {"img_path_or_list": "test_images/test1.jpg", "question": "Describe the image."},
                {"img_path_or_list": "test_images/test2.jpg", "question": "How many people are there?"},
            ]
        num_threads: Number of concurrent workers.

    Returns:
        A list of results aligned with ``tasks`` (``None`` for failed calls).
    """
    results = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {executor.submit(model, **task): i for i, task in enumerate(tasks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                print(f"Task {idx} failed: {e}")
                results[idx] = None
    return results


if __name__ == "__main__":
    # Reads OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL from the environment.
    model = build_model(max_try=5)

    # ---- Single (serial) call ----
    ans = model(
        img_path_or_list="test_images/test1.jpg",
        question="Describe the content of the image and list the objects in it.",
        system_prompt="Please answer in English.",
    )
    print("answer:", ans)

    # ---- Batch (concurrent) calls ----
    tasks = [
        {"img_path_or_list": "test_images/test1.jpg", "question": "Describe the image in English."},
        {"img_path_or_list": "test_images/test1.jpg", "question": "How many people are in the image?"},
    ]
    results = batch_api_call(model, tasks, num_threads=16)
    for i, res in enumerate(results):
        print(f"Task {i}: {res}")
