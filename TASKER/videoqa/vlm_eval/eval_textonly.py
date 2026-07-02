"""
方法0: 不看视频 (Text-Only / Blind Baseline)
仅用问题文本和选项让 VLM 回答，完全不提供视频帧。
用于评估"问题本身是否有捷径可循"的下界参考。
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAX_CONCURRENT_REQUESTS, RESULTS_BASE_DIR
from api_utils import call_qwen_vl
from dataset_utils import (
    load_egoschema_dataset, load_nextqa_dataset,
    build_vqa_prompt, parse_answer,
    evaluate_egoschema, evaluate_nextqa,
    save_results, export_egoschema_submission
)


def run_single_item(item):
    """处理单个样本（不看视频，纯文本回答）"""
    question = item["question"]
    options = item["options"]
    uid = item.get("uid") or item.get("quid")
    
    # 构建 prompt（告知是 video question 但不给帧）
    option_labels = ['A', 'B', 'C', 'D', 'E']
    options_text = "\n".join([f"({label}) {opt}" for label, opt in zip(option_labels, options)])
    
    prompt = (
        f"Answer the following multiple-choice question about a video. "
        f"You do NOT have access to the video, so use your best judgment based on the question and options alone.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Please respond with ONLY the letter of the correct answer (A, B, C, D, or E). "
        f"Do not provide any explanation."
    )
    
    response = call_qwen_vl(
        question=prompt,
        image_paths=[],  # 不提供任何图片
        system_prompt=(
            "You are an expert at answering questions. "
            "Answer based solely on the question text and your knowledge."
        ),
        temperature=0.0,
        max_tokens=64,
    )
    
    pred = parse_answer(response)
    
    return {
        "uid": uid,
        "pred": pred,
        "response": response,
        "num_frames": 0,
    }


def run_egoschema(args, subset_only=True):
    """在 EgoSchema 上评测"""
    split_name = "subset (500)" if subset_only else "full (5031)"
    split_tag = "egoschema_subset" if subset_only else "egoschema_full"
    
    print("=" * 60)
    print(f"方法: Text-Only (不看视频, blind baseline)")
    print(f"数据集: EgoSchema {split_name}")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_egoschema_dataset(subset_only=subset_only)
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Checkpoint 支持
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "textonly_checkpoint", split_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "predictions.json")
    
    processed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed = json.load(f)
        print(f"  已有 checkpoint: {len(processed)} 个样本")
    
    remaining = [item for item in dataset if item["uid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    if not remaining:
        print("  所有样本已处理完毕！")
    else:
        pbar = tqdm(total=len(remaining), desc=f"EgoSchema Text-Only ({split_name})")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_single_item, item): item
                for item in remaining
            }
            
            for future in as_completed(futures):
                result = future.result()
                processed[result["uid"]] = result
                pbar.update(1)
                
                if len(processed) % 10 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(processed, f, ensure_ascii=False)
        
        pbar.close()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(processed, f, ensure_ascii=False)
    
    # 评估
    predictions = {uid: r["pred"] for uid, r in processed.items()}
    ground_truth = {item["uid"]: item["answer"] for item in dataset}
    
    eval_result = evaluate_egoschema(predictions, ground_truth)
    eval_result["method"] = "textonly_0frames"
    eval_result["split"] = "Sub." if subset_only else "Full"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"  EgoSchema {split_name} Text-Only 结果:")
    print(f"  准确率: {eval_result['accuracy_pct']}")
    print(f"  正确/总数: {eval_result['correct']}/{eval_result['total']}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, "textonly_0frames", split_tag)
    
    # Full set: 导出提交文件
    if not subset_only:
        export_egoschema_submission(predictions, "textonly_0frames")
    
    return eval_result


def run_nextqa(args):
    """在 NExT-QA 上评测"""
    print("=" * 60)
    print(f"方法: Text-Only (不看视频, blind baseline)")
    print(f"数据集: NExT-QA MC (test)")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_nextqa_dataset()
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Checkpoint 支持
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "textonly_checkpoint", "nextqa")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "predictions.json")
    
    processed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed = json.load(f)
        print(f"  已有 checkpoint: {len(processed)} 个样本")
    
    remaining = [item for item in dataset if item["quid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    if not remaining:
        print("  所有样本已处理完毕！")
    else:
        pbar = tqdm(total=len(remaining), desc="NExTQA Text-Only")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_single_item, item): item
                for item in remaining
            }
            
            for future in as_completed(futures):
                result = future.result()
                processed[result["uid"]] = result
                pbar.update(1)
                
                if len(processed) % 20 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(processed, f, ensure_ascii=False)
        
        pbar.close()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(processed, f, ensure_ascii=False)
    
    # 评估
    predictions = [{"quid": uid, "pred": r["pred"]} for uid, r in processed.items()]
    eval_result = evaluate_nextqa(predictions, dataset)
    eval_result["method"] = "textonly_0frames"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*40}")
    print(f"  NExT-QA Text-Only 结果:")
    print(f"  Avg: {eval_result['Avg']}")
    print(f"  Temporal: {eval_result['Temporal']}")
    print(f"  Causal: {eval_result['Causal']}")
    print(f"  Descriptive: {eval_result['Descriptive']}")
    print(f"{'='*40}")
    
    save_results(eval_result, "textonly_0frames", "nextqa")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Text-Only (不看视频) 评测")
    parser.add_argument("--dataset", type=str, choices=["egoschema", "nextqa", "both"], default="both")
    parser.add_argument("--max_workers", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help="并行处理数")
    parser.add_argument("--egoschema_split", type=str, choices=["subset", "full", "both"], default="both",
                        help="EgoSchema 评测范围")
    args = parser.parse_args()
    
    if args.dataset in ["egoschema", "both"]:
        if args.egoschema_split in ["subset", "both"]:
            run_egoschema(args, subset_only=True)
        if args.egoschema_split in ["full", "both"]:
            run_egoschema(args, subset_only=False)
    
    if args.dataset in ["nextqa", "both"]:
        run_nextqa(args)
