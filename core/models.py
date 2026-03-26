from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import requests
import openai

# ----------------------------
# 抽象基类
# ----------------------------
class BaseChatModel(ABC):
    """
    所有模型的基类，定义统一接口
    """
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             images: Optional[List[str]] = None,
             **kwargs) -> str:
        """
        多模态 chat 接口
        :param messages: [{"role": "user", "content": "..."}]
        :param images: 可选，图像 URL 或 base64 列表
        :return: 模型生成的文字
        """
        pass


# ----------------------------
# OpenAI API 模型
# ----------------------------
class OpenAIModel(BaseChatModel):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model_name = model_name
        openai.api_key = api_key
        openai.base_url = base_url

    def chat(self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs) -> str:
        # 构造 multi-modal message
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "user" and images:
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        *[{"type": "image_url", "image_url": {"url": img}} for img in images]
                    ]
                })
            else:
                formatted_messages.append(msg)

        response = openai.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            **kwargs
        )
        return response.choices[0].message.content


# ----------------------------
# vLLM Serve 模型 (OpenAI-compatible API)
# ----------------------------
class VLLMModel(BaseChatModel):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{self.base_url}/chat/completions"

    def chat(self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs) -> str:
        # vLLM 的 API 也是 OpenAI 格式，但通常不支持 image
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        resp = requests.post(self.api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    # OpenAI
    openai_model = OpenAIModel(model_name="gpt-4o-mini", api_key="YOUR_KEY")
    out1 = openai_model.chat([{"role": "user", "content": "解释这张图像"},],
                             images=["https://example.com/test.png"])
    print("OpenAI:", out1)

    # vLLM
    # vllm_model = VLLMModel(model_name="Qwen2.5-7B-Instruct")
    # out2 = vllm_model.chat([{"role": "user", "content": "你好，请介绍一下你自己"}])
    # print("vLLM:", out2)
