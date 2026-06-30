"""
Aiping.cn OpenAI 兼容格式模型适配器

通过 aiping.cn 的 OpenAI 兼容 API 调用 GPT-5 和 GPT-5-Mini。
接口格式为标准 OpenAI chat/completions，支持 vision (多图输入)。

支持的模型：
- gpt5: GPT-5 (旗舰，支持 vlm)
- gpt5_mini: GPT-5 Mini (经济旗舰，支持 vlm)

用法:
    model = build_aiping_model("gpt5")
    response = model(
        img_path_or_list=["img1.png", "img2.png"],
        question="What do you see?",
        system_prompt="You are a GUI assistant.",
        image_first=True
    )
"""

import os
import base64
import time
import logging
import requests
from typing import List, Optional, Union


# ============================================================================
# 配置
# ============================================================================

AIPING_API_BASE = "https://aiping.cn/api/v1/chat/completions"

# API Key 需要自行填写
AIPING_API_KEY = "QC-62f427539af11019afba2b2e6fe6be48-479e41a5b60dda67be1adc711792d654"  # <-- 请替换为你的 aiping.cn API Key

AIPING_MODEL_CONFIGS = {
    "gpt5": {
        "model_name": "GPT-5",
        "display_name": "GPT-5",
        "max_tokens": 8192,
    },
    "gpt5_mini": {
        "model_name": "GPT-5-Mini",
        "display_name": "GPT-5-Mini",
        "max_tokens": 8192,
    },
}


# ============================================================================
# 工具函数
# ============================================================================

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图片文件编码为 base64 data URL"""
    if not os.path.exists(image_path):
        logging.warning(f"[AipingAdapter] 图片不存在: {image_path}")
        return None

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_map.get(ext, 'image/png')

    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        b64_str = base64.b64encode(img_data).decode('utf-8')
        return f"data:{mime_type};base64,{b64_str}"
    except Exception as e:
        logging.warning(f"[AipingAdapter] 图片编码失败 {image_path}: {e}")
        return None


# ============================================================================
# 适配器类
# ============================================================================

class AipingModelAdapter:
    """
    Aiping.cn OpenAI 兼容模型适配器。
    
    将 aiping.cn 的 OpenAI 格式 API 包装为与 VLLMModel.__call__ 完全兼容的接口。
    支持多图 vision 输入（GPT-5/GPT-5-Mini 均为 VLM）。
    """

    def __init__(self, model_type: str, api_key: str = None, max_try: int = 3, 
                 timeout: int = 300, max_tokens: int = 8192, temperature: float = 0.6):
        """
        Args:
            model_type: 模型类型，必须是 AIPING_MODEL_CONFIGS 中的 key
            api_key: aiping.cn API Key（不传则使用模块级默认值）
            max_try: 最大重试次数
            timeout: 请求超时(秒)
            max_tokens: 最大生成 token 数
            temperature: 生成温度
        """
        if model_type not in AIPING_MODEL_CONFIGS:
            raise ValueError(
                f"未知的模型类型: {model_type}\n"
                f"支持的类型: {list(AIPING_MODEL_CONFIGS.keys())}"
            )

        cfg = AIPING_MODEL_CONFIGS[model_type]
        self.model_type = model_type
        self.model_name = cfg["model_name"]
        self.display_name = cfg["display_name"]
        self.max_try = max_try
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

        # API Key
        self.api_key = api_key or AIPING_API_KEY
        if self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "[AipingAdapter] 请设置 API Key！\n"
                "方式1: 修改 api/aiping_adapter.py 中的 AIPING_API_KEY\n"
                "方式2: 设置环境变量 AIPING_API_KEY\n"
                "方式3: 调用时传入 api_key 参数"
            )

        logging.info(f"[AipingAdapter] 初始化完成: {model_type} ({self.model_name})")

    def __call__(self, img_path_or_list=None, question='', img_tnx=None,
                 image_first=False, system_prompt=None) -> str:
        """
        与 VLLMModel.__call__ 完全兼容的调用接口。

        Args:
            img_path_or_list: 单个图片路径(str)或图片路径列表(list)，或 None
            question: 文本 prompt
            img_tnx: 兼容参数(忽略)
            image_first: 是否图片放在文本前面
            system_prompt: 系统 prompt

        Returns:
            str: 模型响应文本
        """
        if question == '':
            raise ValueError("Question cannot be empty")

        # ---- 1. 编码图片为 base64 data URL ----
        image_data_urls = []
        if img_path_or_list is not None:
            if isinstance(img_path_or_list, (list, tuple)):
                for img_path in img_path_or_list:
                    data_url = encode_image_to_base64(img_path)
                    if data_url is None:
                        logging.warning(f"[AipingAdapter] 图片编码失败: {img_path}")
                        return ""
                    image_data_urls.append(data_url)
            elif isinstance(img_path_or_list, str):
                data_url = encode_image_to_base64(img_path_or_list)
                if data_url is None:
                    return ""
                image_data_urls = [data_url]
            else:
                raise TypeError(f"img_path_or_list 类型错误: {type(img_path_or_list)}")

        # ---- 2. 构建 OpenAI 格式 messages ----
        messages = []

        # System prompt
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # User message (多模态)
        user_content = []

        if image_data_urls:
            if image_first:
                # 先图片后文本
                for data_url in image_data_urls:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                user_content.append({"type": "text", "text": question})
            else:
                # 先文本后图片
                user_content.append({"type": "text", "text": question})
                for data_url in image_data_urls:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
        else:
            user_content.append({"type": "text", "text": question})

        messages.append({"role": "user", "content": user_content})

        # ---- 3. 构建请求体 ----
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        # ---- 4. 发送请求，带重试和退避 ----
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response_text = None
        num_request = 0

        while response_text is None and num_request <= self.max_try:
            try:
                resp = requests.post(
                    url=AIPING_API_BASE,
                    headers=headers,
                    json=request_body,
                    timeout=self.timeout,
                )

                # HTTP 状态码检查
                if resp.status_code == 413:
                    logging.error(f"[AipingAdapter] {self.model_type} 请求体过大 (413)，跳过")
                    return ""

                if resp.status_code == 429:
                    # 限频
                    backoff = 5 * (2 ** min(num_request, 4))
                    logging.warning(f"[AipingAdapter] {self.model_type} 限频 (429), 等待 {backoff}s")
                    time.sleep(backoff)
                    num_request += 1
                    continue

                if resp.status_code != 200:
                    logging.error(
                        f"[AipingAdapter] {self.model_type} HTTP {resp.status_code}: "
                        f"{resp.text[:300]}"
                    )
                    num_request += 1
                    time.sleep(3 * num_request)
                    continue

                # 解析响应
                resp_data = resp.json()

                # 检查 API 错误
                if "error" in resp_data:
                    error_msg = resp_data["error"].get("message", str(resp_data["error"]))
                    logging.error(f"[AipingAdapter] {self.model_type} API错误: {error_msg[:300]}")

                    # 不可重试的错误
                    if "too large" in error_msg.lower() or "413" in error_msg:
                        return ""

                    num_request += 1
                    time.sleep(3 * num_request)
                    continue

                # 成功解析
                choices = resp_data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                else:
                    logging.warning(f"[AipingAdapter] {self.model_type} 响应无 choices")
                    num_request += 1
                    time.sleep(2)
                    continue

            except requests.exceptions.Timeout:
                logging.warning(f"[AipingAdapter] {self.model_type} 请求超时 ({self.timeout}s)")
                num_request += 1
                time.sleep(5 * num_request)

            except requests.exceptions.ConnectionError as e:
                logging.warning(f"[AipingAdapter] {self.model_type} 连接错误: {str(e)[:200]}")
                num_request += 1
                time.sleep(5 * num_request)

            except Exception as e:
                logging.error(f"[AipingAdapter] {self.model_type} 未知异常: {str(e)[:200]}")
                num_request += 1
                if num_request > self.max_try:
                    return ""
                time.sleep(3 * num_request)

        if response_text is None:
            logging.error(f"[AipingAdapter] {self.model_type} 超过最大重试次数 ({self.max_try})")
            return ""

        return response_text

    def __repr__(self):
        return f"AipingModelAdapter(model_type='{self.model_type}', model_name='{self.model_name}')"


# ============================================================================
# 工厂函数
# ============================================================================

def build_aiping_model(model_type: str, api_key: str = None, max_try: int = 3,
                       timeout: int = 300, max_tokens: int = 8192, 
                       temperature: float = 0.6):
    """
    工厂函数：构建 Aiping 模型适配器实例。

    Args:
        model_type: 模型类型，可选:
            - "gpt5": GPT-5 (旗舰)
            - "gpt5_mini": GPT-5 Mini (经济旗舰)
        api_key: aiping.cn API Key
        max_try: 最大重试次数
        timeout: 请求超时(秒)
        max_tokens: 最大生成 token 数
        temperature: 生成温度

    Returns:
        AipingModelAdapter 实例
    """
    # 支持从环境变量读取 API Key
    if api_key is None:
        api_key = os.environ.get("AIPING_API_KEY", AIPING_API_KEY)

    return AipingModelAdapter(
        model_type=model_type,
        api_key=api_key,
        max_try=max_try,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )
