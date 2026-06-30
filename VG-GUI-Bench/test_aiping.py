#!/usr/bin/env python3
"""
快速测试 aiping.cn GPT-5 连通性
用法: python test_aiping.py
"""
import os
import base64

# pip install openai
from openai import OpenAI

API_KEY = os.environ.get("AIPING_API_KEY", "YOUR_API_KEY_HERE")

client = OpenAI(
    base_url="https://aiping.cn/api/v1",
    api_key=API_KEY,
)

# ============================================================
# 测试 1: GPT-5 纯文本
# ============================================================
print("=" * 50)
print("[Test 1] GPT-5 纯文本")
print("=" * 50)
try:
    completion = client.chat.completions.create(
        model="GPT-5",
        max_completion_tokens=64,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print(f"  ✅ 成功! Response: {completion.choices[0].message.content[:100]}")
    print(f"  Usage: input={completion.usage.prompt_tokens}, output={completion.usage.completion_tokens}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# ============================================================
# 测试 2: GPT-5 图片输入 (VLM)
# ============================================================
print()
print("=" * 50)
print("[Test 2] GPT-5 图片输入 (VLM)")
print("=" * 50)

# 找一张测试图
test_img = None
dataset_root = os.environ.get("DATASET_ROOT", "/root/projects/lql/VG-GUI-Bench/MONDAY")
img_dir = os.path.join(dataset_root, "images", "origin")
if os.path.isdir(img_dir):
    for d in sorted(os.listdir(img_dir))[:1]:
        subdir = os.path.join(img_dir, d)
        if os.path.isdir(subdir):
            for f in sorted(os.listdir(subdir))[:1]:
                if f.endswith(".png") or f.endswith(".jpg"):
                    test_img = os.path.join(subdir, f)
                    break

if test_img and os.path.exists(test_img):
    print(f"  使用测试图片: {test_img}")
    with open(test_img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    try:
        completion = client.chat.completions.create(
            model="GPT-5",
            max_completion_tokens=128,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Describe what you see in this screenshot in one sentence."},
                ]
            }],
        )
        print(f"  ✅ 成功! Response: {completion.choices[0].message.content[:150]}")
        print(f"  Usage: input={completion.usage.prompt_tokens}, output={completion.usage.completion_tokens}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
else:
    print(f"  ⚠️ 未找到测试图片，跳过 VLM 测试")
    print(f"  (检查路径: {img_dir})")

print()
print("测试完成！")
