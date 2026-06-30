#!/usr/bin/env python3
"""
VG-GUI-Bench 模型连通性检测脚本

检测所有模型（开源 + 闭源）是否能正确调用，包括：
1. 纯文本调用
2. 单图调用
3. 多图调用（模拟 uniform10 模式）

用法：
    python test_models.py                # 测试全部模型
    python test_models.py qwen3vl        # 只测 Qwen3-VL
    python test_models.py gemini_flash   # 只测 Gemini Flash
    python test_models.py --closed-only  # 只测闭源模型
    python test_models.py --open-only    # 只测开源模型
"""

import os
import sys
import time
import glob
import traceback

# 清除代理
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# 确保 api 目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.closed_model_adapter import build_closed_model, CLOSED_MODEL_CONFIGS
from api.vllm_tool import get_api


# ============================================================================
# 配置
# ============================================================================

# 用一张实际的测试图片（从 MONDAY 数据集中取）
DATASET_ROOT = "/root/projects/lql/VG-GUI-Bench/MONDAY"
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "images", "origin")

OPEN_MODELS = {
    "qwen3vl": {
        "model_type": "qwen3vl",
        "description": "Qwen3-VL-235B-A22B-Instruct (localhost:8000)",
    },
}

CLOSED_MODELS = list(CLOSED_MODEL_CONFIGS.keys())

# 测试用的简单 prompt
TEST_QUESTION = "Describe what you see in this screenshot in one sentence."
TEST_SYSTEM_PROMPT = "You are a GUI automation assistant. Respond concisely."


# ============================================================================
# 工具函数
# ============================================================================

def find_test_image():
    """从数据集中找一张测试图片"""
    if os.path.isdir(TEST_IMG_DIR):
        imgs = glob.glob(os.path.join(TEST_IMG_DIR, "**", "*.png"), recursive=True)
        if imgs:
            return imgs[0]
    
    # fallback: 当前目录找
    for ext in ["*.png", "*.jpg"]:
        imgs = glob.glob(ext)
        if imgs:
            return imgs[0]
    
    return None


def find_multi_test_images(count=3):
    """找多张测试图片（模拟 ref_imgs + target）"""
    if os.path.isdir(TEST_IMG_DIR):
        imgs = glob.glob(os.path.join(TEST_IMG_DIR, "**", "*.png"), recursive=True)
        if len(imgs) >= count:
            return imgs[:count]
        elif imgs:
            return imgs * ((count // len(imgs)) + 1)[:count]
    return None


def print_result(model_name, test_name, success, response=None, error=None, elapsed=0):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} [{test_name}] ({elapsed:.1f}s)")
    if success and response:
        preview = str(response)[:100].replace('\n', ' ')
        print(f"       响应: {preview}...")
    if error:
        print(f"       错误: {error}")


# ============================================================================
# 测试开源模型 (Qwen3-VL via vLLM)
# ============================================================================

def test_open_model(model_type="qwen3vl"):
    """测试开源模型"""
    print(f"\n{'='*60}")
    print(f"  测试开源模型: {model_type}")
    print(f"{'='*60}")
    
    # 初始化
    try:
        model = get_api(
            ['all'], model_type=model_type, max_try=2,
            EXTRA_PARAMS={"max_tokens": 256, "temperature": 0.6, "timeout": 60},
            split=['gui_video', 'ocr']
        )
        print(f"  ✅ 模型初始化成功")
    except Exception as e:
        print(f"  ❌ 模型初始化失败: {e}")
        traceback.print_exc()
        return False
    
    all_pass = True
    test_img = find_test_image()
    
    # Test 1: 纯文本
    try:
        t0 = time.time()
        resp = model(img_path_or_list=None, question="Say hello in one word.", system_prompt=None)
        elapsed = time.time() - t0
        success = resp is not None and len(str(resp)) > 0
        print_result(model_type, "纯文本", success, resp, elapsed=elapsed)
        if not success:
            all_pass = False
    except Exception as e:
        print_result(model_type, "纯文本", False, error=str(e))
        all_pass = False
    
    # Test 2: 单图
    if test_img:
        try:
            t0 = time.time()
            resp = model(
                img_path_or_list=test_img,
                question=TEST_QUESTION,
                system_prompt=TEST_SYSTEM_PROMPT,
                image_first=True
            )
            elapsed = time.time() - t0
            success = resp is not None and len(str(resp)) > 0
            print_result(model_type, "单图", success, resp, elapsed=elapsed)
            if not success:
                all_pass = False
        except Exception as e:
            print_result(model_type, "单图", False, error=str(e))
            all_pass = False
    else:
        print(f"  ⚠️  跳过图片测试（未找到测试图片）")
    
    # Test 3: 多图（模拟 uniform10 的场景）
    multi_imgs = find_multi_test_images(3)
    if multi_imgs:
        try:
            t0 = time.time()
            resp = model(
                img_path_or_list=multi_imgs,
                question=TEST_QUESTION,
                system_prompt=TEST_SYSTEM_PROMPT,
                image_first=True
            )
            elapsed = time.time() - t0
            success = resp is not None and len(str(resp)) > 0
            print_result(model_type, "多图(3张)", success, resp, elapsed=elapsed)
            if not success:
                all_pass = False
        except Exception as e:
            print_result(model_type, "多图(3张)", False, error=str(e))
            all_pass = False
    
    return all_pass


# ============================================================================
# 测试闭源模型 (ClosedModelAdapter)
# ============================================================================

def test_closed_model(model_type):
    """测试闭源模型"""
    print(f"\n{'='*60}")
    print(f"  测试闭源模型: {model_type}")
    print(f"{'='*60}")
    
    # 初始化
    try:
        model = build_closed_model(model_type=model_type, max_try=2, timeout=120)
        print(f"  ✅ 模型初始化成功")
    except Exception as e:
        print(f"  ❌ 模型初始化失败: {e}")
        return False
    
    all_pass = True
    test_img = find_test_image()
    
    # Test 1: 纯文本
    try:
        t0 = time.time()
        resp = model(img_path_or_list=None, question="Say hello in one word.", system_prompt=None)
        elapsed = time.time() - t0
        success = resp is not None and len(str(resp)) > 0
        print_result(model_type, "纯文本", success, resp, elapsed=elapsed)
        if not success:
            all_pass = False
    except Exception as e:
        print_result(model_type, "纯文本", False, error=str(e))
        all_pass = False
    
    # Test 2: 单图
    if test_img:
        try:
            t0 = time.time()
            resp = model(
                img_path_or_list=test_img,
                question=TEST_QUESTION,
                system_prompt=TEST_SYSTEM_PROMPT,
                image_first=True
            )
            elapsed = time.time() - t0
            success = resp is not None and len(str(resp)) > 0
            print_result(model_type, "单图", success, resp, elapsed=elapsed)
            if not success:
                all_pass = False
        except Exception as e:
            print_result(model_type, "单图", False, error=str(e))
            all_pass = False
    else:
        print(f"  ⚠️  跳过图片测试（未找到测试图片）")
    
    # Test 3: 多图（模拟 uniform10，用 3 张快速验证）
    multi_imgs = find_multi_test_images(3)
    if multi_imgs:
        try:
            t0 = time.time()
            resp = model(
                img_path_or_list=multi_imgs,
                question=TEST_QUESTION,
                system_prompt=TEST_SYSTEM_PROMPT,
                image_first=True
            )
            elapsed = time.time() - t0
            success = resp is not None and len(str(resp)) > 0
            print_result(model_type, "多图(3张)", success, resp, elapsed=elapsed)
            if not success:
                all_pass = False
        except Exception as e:
            print_result(model_type, "多图(3张)", False, error=str(e))
            all_pass = False
    
    return all_pass


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("  VG-GUI-Bench 模型连通性检测")
    print("=" * 60)
    print(f"  数据集目录: {DATASET_ROOT}")
    
    test_img = find_test_image()
    if test_img:
        print(f"  测试图片: {test_img}")
    else:
        print(f"  ⚠️  未找到测试图片，将跳过图片相关测试")
    
    # 解析参数
    args = sys.argv[1:]
    test_open = True
    test_closed = True
    specific_models = []
    
    for arg in args:
        if arg == "--closed-only":
            test_open = False
        elif arg == "--open-only":
            test_closed = False
        else:
            specific_models.append(arg)
    
    results = {}
    
    # 测试开源模型
    if test_open:
        for model_key in OPEN_MODELS:
            if specific_models and model_key not in specific_models:
                continue
            ok = test_open_model(model_key)
            results[model_key] = ok
    
    # 测试闭源模型
    if test_closed:
        for model_key in CLOSED_MODELS:
            if specific_models and model_key not in specific_models:
                continue
            ok = test_closed_model(model_key)
            results[model_key] = ok
    
    # 汇总
    print("\n")
    print("=" * 60)
    print("  汇总结果")
    print("=" * 60)
    
    all_ok = True
    for model, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {model}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("  🎉 全部通过！可以放心启动 run_leaderboard.sh")
    else:
        print("  ⚠️  有模型未通过，请检查对应错误后再启动评测")
    
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
