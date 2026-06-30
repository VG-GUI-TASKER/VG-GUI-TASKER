"""
闭源模型适配器 (Closed-Source Model Adapter)

将 xtools 的 DistillationAPI 包装成与 VLLMModel.__call__ 完全兼容的接口，
使得 eval_qwen.py 中的 get_seeclick_response() 可以无缝调用闭源模型。

主要解决的问题：
1. DistillationAPI.__call__ 只支持单张图片，本适配器支持多张图片列表
2. DistillationAPI 使用 NACI 消息格式，与 OpenAI 格式不同
3. 统一 __call__ 签名为: (img_path_or_list, question, system_prompt, image_first) -> str

支持的模型：
- gemini_flash: Gemini 3.1 Flash
- gemini_pro: Gemini 3.1 Pro
- claude_sonnet: Claude Sonnet 4.6
- seed2: Seed 2.0 Pro
- kimi: Kimi K2.5
"""

import copy
import json
import time
import uuid
import logging
import requests

from api.gpt_new import DistillationAPI, encode_image, get_simple_auth


# ============================================================================
# 模型配置（与 api_xtools/examples.py 中的 build_api_simple 保持一致）
# ============================================================================

CLOSED_MODEL_CONFIGS = {
    "gemini_flash": {
        "model_name": "gemini-3.1-flash-lite-preview",
        "api_key": "SaxlBWvqHGKWucuy",
        "model_marker": "api_naci_default_gemini-3.1-flash-lite-preview",
        "user": "gzafhJR5_richardlai",
    },
    "gemini_pro": {
        "model_name": "gemini-3.1-pro-preview",
        "api_key": "SaxlBWvqHGKWucuy",
        "model_marker": "api_naci_default_gemini-3.1-pro-preview",
        "user": "gzafhJR5_richardlai",
    },
    "claude_sonnet": {
        "model_name": "anthropic.claude-sonnet-4-6",
        "api_key": "SaxlBWvqHGKWucuy",
        "model_marker": "api_aws_third_anthropic.claude-sonnet-4-6",
        "user": "gzafhJR5_richardlai",
        "params": {"thinking": {"type": "adaptive"}},
    },
    "seed2": {
        "model_name": "doubao-seed-2-0-pro-260215",
        "api_key": "SaxlBWvqHGKWucuy",
        "model_marker": "api_doubao_doubao-seed-2-0-pro-260215",
        "user": "gzafhJR5_richardlai",
        "params": {"reasoning_effort": "minimal"},
    },
    "kimi": {
        "model_name": "kimi-k2.5",
        "api_key": "C8zEDuQJlqVu0C9d",
        "model_marker": "api_moonshot_kimi-k2.5",
        "user": "pEfDjZtY_zhytang",
    },
}


class ClosedModelAdapter:
    """
    闭源模型适配器：将 DistillationAPI (NACI 格式) 包装为与 VLLMModel 兼容的接口。
    
    核心区别：
    - VLLMModel 使用 OpenAI 格式: {"type": "image_url", "image_url": {"url": "..."}}
    - DistillationAPI 使用 NACI 格式: {"type": "image_url", "value": "..."}
    - VLLMModel.__call__ 支持 img_path_or_list (多图), DistillationAPI.__call__ 只支持单图
    - NACI 格式中 system prompt 不单独发 role=system，而是通过 json_data["system"] 字段传递
    """

    def __init__(self, model_type, max_try=3, timeout=300):
        """
        Args:
            model_type: 模型类型名称，必须是 CLOSED_MODEL_CONFIGS 中的 key
            max_try: 最大重试次数
            timeout: 请求超时(秒)
        """
        if model_type not in CLOSED_MODEL_CONFIGS:
            raise ValueError(
                f"未知的模型类型: {model_type}\n"
                f"支持的类型: {list(CLOSED_MODEL_CONFIGS.keys())}"
            )
        
        cfg = CLOSED_MODEL_CONFIGS[model_type].copy()
        self.model_type = model_type
        self.model_name = cfg["model_name"]
        self.max_try = max_try
        self.timeout = timeout
        
        # 构建底层 DistillationAPI 实例
        extra_kwargs = {}
        if "params" in cfg:
            extra_kwargs["params"] = cfg["params"]
        
        self.api = DistillationAPI(
            model_name=cfg["model_name"],
            api_key=cfg["api_key"],
            model_marker=cfg["model_marker"],
            user=cfg["user"],
            max_try=max_try,
            **extra_kwargs,
        )
        # 覆盖 timeout
        self.api.timeout = timeout
        self.api.json_data["timeout"] = timeout
        
        logging.info(f"[ClosedModelAdapter] 初始化完成: {model_type} ({cfg['model_name']})")

    def __call__(self, img_path_or_list=None, question='', img_tnx=None, 
                 image_first=False, system_prompt=None) -> str:
        """
        与 VLLMModel.__call__ 完全兼容的调用接口。
        
        Args:
            img_path_or_list: 单个图片路径(str)或图片路径列表(list)，或 None
            question: 文本 prompt
            img_tnx: 图片事务对象(兼容参数，本适配器忽略)
            image_first: 是否图片放在文本前面
            system_prompt: 系统 prompt
            
        Returns:
            str: 模型响应文本
        """
        if question == '':
            raise ValueError("Question cannot be empty")
        
        # ---- 1. 编码图片 ----
        image_data_urls = []
        if img_path_or_list is not None and len(img_path_or_list) > 0:
            if isinstance(img_path_or_list, (list, tuple)):
                for single_img in img_path_or_list:
                    encoded = encode_image(single_img, img_tnx)
                    if encoded is None:
                        logging.warning(f"[ClosedModelAdapter] 图片编码失败: {single_img}")
                        return ""
                    image_data_urls.append(encoded)
            elif isinstance(img_path_or_list, str):
                encoded = encode_image(img_path_or_list, img_tnx)
                if encoded is None:
                    logging.warning(f"[ClosedModelAdapter] 图片编码失败: {img_path_or_list}")
                    return ""
                image_data_urls = [encoded]
            else:
                raise TypeError(f"img_path_or_list 类型错误: {type(img_path_or_list)}")
        
        # ---- 2. 构建 NACI 格式消息 ----
        content = []
        
        if image_data_urls:
            if image_first:
                # 先图片后文本
                for img_url in image_data_urls:
                    content.append({"type": "image_url", "value": img_url})
                content.append({"type": "text", "value": question})
            else:
                # 先文本后图片
                content.append({"type": "text", "value": question})
                for img_url in image_data_urls:
                    content.append({"type": "image_url", "value": img_url})
        else:
            content.append({"type": "text", "value": question})
        
        messages = [{"role": "user", "content": content}]
        
        # ---- 3. 构建请求 JSON ----
        data_json = copy.deepcopy(self.api.json_data)
        data_json["messages"] = messages
        data_json["request_id"] = str(uuid.uuid4())
        
        # NACI 格式: system prompt 通过 json_data["system"] 字段传递
        if system_prompt is not None and system_prompt.strip():
            data_json["system"] = system_prompt
        
        # ---- 4. 发送请求，带重试和限频退避 ----
        response = None
        num_request = 0
        base_backoff = 3.0  # 限频退避基础秒数
        
        while response is None:
            try:
                headers = dict(self.api.get_header())
                original_resp = requests.post(
                    url=self.api.base_url,
                    headers=headers,
                    json=data_json,
                    timeout=self.timeout,
                )
                
                response_data = original_resp.json()
                
                # 检查平台级错误码
                error_code = response_data.get("code", 0)
                
                if error_code == 1005:
                    # 限频：指数退避后重试（不计入 max_try）
                    backoff = base_backoff * (2 ** min(num_request, 4))  # 3s, 6s, 12s, 24s, 48s
                    logging.warning(f"[ClosedModelAdapter] {self.model_type} 触发限频, 等待 {backoff:.0f}s 后重试...")
                    time.sleep(backoff)
                    num_request += 1
                    if num_request > self.max_try * 3:  # 限频最多重试 3x max_try 次
                        logging.error(f"[ClosedModelAdapter] {self.model_type} 限频重试次数耗尽")
                        return ""
                    continue
                
                if error_code == 2000 and "413" in response_data.get("msg", ""):
                    # 413 Request Entity Too Large：请求体过大，无法重试
                    logging.error(f"[ClosedModelAdapter] {self.model_type} 请求体过大 (413)，跳过此请求")
                    return ""
                
                if error_code != 0 and response_data.get("answer") is None:
                    # 其他平台错误
                    logging.error(f"[ClosedModelAdapter] {self.model_type} 平台错误 code={error_code}: {response_data.get('msg', '')[:200]}")
                    num_request += 1
                    if num_request > self.max_try:
                        return ""
                    time.sleep(2)
                    continue
                
                # 解析响应（与原 DistillationAPI.__call__ 一致）
                if len(response_data['answer']) == 2:
                    # 有 think + answer 两部分（如 Claude adaptive thinking）
                    think = response_data['answer'][0]['value']
                    answer = response_data['answer'][1]['value']
                    # 只返回 answer 部分（与 VLLMModel 返回格式对齐）
                    response = answer
                else:
                    response = response_data['answer'][0]['value']
                    
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"[ClosedModelAdapter] {self.model_type} 网络异常: {error_msg[:200]}")
                response = None
                
                # 不可恢复的错误直接放弃
                if any(kw in error_msg for kw in [
                    'UnsupportedImageFormat', 'Image dimensions are too small',
                    'Maximum allowed: 36000000 pixels'
                ]):
                    return ""
                    
            num_request += 1
            if num_request > self.max_try:
                logging.error(f"[ClosedModelAdapter] {self.model_type} 超过最大重试次数 ({self.max_try})")
                return ""
            
            if response is None:
                time.sleep(2 * num_request)  # 递增退避
        
        return response

    def __repr__(self):
        return f"ClosedModelAdapter(model_type='{self.model_type}', model_name='{self.model_name}')"


def build_closed_model(model_type, max_try=3, timeout=300):
    """
    工厂函数：构建闭源模型适配器实例。
    
    用法:
        model = build_closed_model("gemini_flash")
        response = model(img_path_or_list=["img1.png", "img2.png"], 
                         question="What do you see?",
                         system_prompt="You are a GUI assistant.",
                         image_first=True)
    
    Args:
        model_type: 模型类型，可选:
            - "gemini_flash": Gemini 3.1 Flash
            - "gemini_pro": Gemini 3.1 Pro
            - "claude_sonnet": Claude Sonnet 4.6
            - "seed2": Seed 2.0 Pro
            - "kimi": Kimi K2.5
        max_try: 最大重试次数
        timeout: 请求超时(秒)
    
    Returns:
        ClosedModelAdapter 实例
    """
    return ClosedModelAdapter(model_type=model_type, max_try=max_try, timeout=timeout)
