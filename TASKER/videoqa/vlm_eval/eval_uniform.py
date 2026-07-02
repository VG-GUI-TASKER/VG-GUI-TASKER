"""
方法1: 均匀取帧 + VLM 理解 (baseline)
最基础的 baseline：从视频中均匀采样 N 帧，直接让 VLM 回答问题
"""
import os
import sys
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import UNIFORM_NUM_FRAMES, MAX_CONCURRENT_REQUESTS, RESULTS_BASE_DIR
from api_utils import call_qwen_vl
from dataset_utils import (
    load_egoschema_dataset, load_nextqa_dataset,
    extract_frames_uniform,
    build_vqa_prompt, parse_answer,
    evaluate_egoschema, evaluate_nextqa,
    save_results, export_egoschema_submission
)


def run_single_item(item, num_frames, cache_dir):
    """处理单个样本"""
    video_path = item["video_path"]
    question = item["question"]
    options = item["options"]
    uid = item.get("uid") or item.get("quid")
    
    # 提取均匀帧
    frame_paths = extract_frames_uniform(video_path, num_frames, cache_dir)
    
    if not frame_paths:
        return {"uid": uid, "pred": -1, "response": None, "num_frames": 0, "error": "无法提取帧"}
    
    # 构建 prompt
    prompt = build_vqa_prompt(question, options, len(frame_paths))
    
    # 调用 API
    system_prompt = (
        "You are an expert video understanding assistant. "
        "You are shown frames sampled uniformly from a video. "
        "Analyze the visual content carefully to answer the question."
    )
    
    response = call_qwen_vl(
        question=prompt,
        image_paths=frame_paths,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=64,
    )
    
    # 解析答案
    pred = parse_answer(response)
    
    return {
        "uid": uid,
        "pred": pred,
        "response": response,
        "num_frames": len(frame_paths),
    }


def run_egoschema(num_frames: int, max_workers: int, cache_dir: str, resume_path: str = None, subset_only: bool = True):
    """在 EgoSchema 上评测"""
    split_name = "subset (500)" if subset_only else "full (5031)"
    print("=" * 60)
    print(f"方法: 均匀取帧 ({num_frames} 帧)")
    print(f"数据集: EgoSchema {split_name}")
    print(f"并行数: {max_workers}")
    print("=" * 60)
    
    dataset = load_egoschema_dataset(subset_only=subset_only)
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Resume 支持
    split_tag = "egoschema_subset" if subset_only else "egoschema_full"
    processed = {}
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            processed = json.load(f)
        print(f"  从断点恢复: 已处理 {len(processed)} 个样本")
    
    # 过滤已处理的
    remaining = [item for item in dataset if item["uid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    # 并行处理
    results = []
    pbar = tqdm(total=len(remaining), desc=f"EgoSchema Uniform ({split_name})")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_item, item, num_frames, cache_dir): item
            for item in remaining
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            processed[result["uid"]] = result
            pbar.update(1)
            
            # 定期保存
            if len(results) % 10 == 0:
                _save_checkpoint(processed, "uniform", split_tag)
    
    pbar.close()
    
    # 最终保存
    _save_checkpoint(processed, "uniform", split_tag)
    
    # 评估
    predictions = {uid: r["pred"] for uid, r in processed.items()}
    ground_truth = {item["uid"]: item["answer"] for item in dataset}
    
    eval_result = evaluate_egoschema(predictions, ground_truth)
    eval_result["method"] = f"uniform_{num_frames}frames"
    eval_result["split"] = "Sub." if subset_only else "Full"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"  EgoSchema {split_name} 评测结果:")
    print(f"  准确率 (Table 1 '{eval_result['split']}'): {eval_result['accuracy_pct']}")
    print(f"  正确/总数: {eval_result['correct']}/{eval_result['total']}")
    print(f"  无效回答: {eval_result['invalid']}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"uniform_{num_frames}frames", split_tag)
    
    # Full set: 导出提交文件
    if not subset_only:
        export_egoschema_submission(predictions, f"uniform_{num_frames}frames")
    
    return eval_result


def run_nextqa(num_frames: int, max_workers: int, cache_dir: str, resume_path: str = None):
    """在 NExT-QA 上评测"""
    print("=" * 60)
    print(f"方法: 均匀取帧 ({num_frames} 帧)")
    print(f"数据集: NExT-QA MC (test)")
    print(f"并行数: {max_workers}")
    print("=" * 60)
    
    dataset = load_nextqa_dataset()
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Resume 支持
    processed = {}
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            processed = json.load(f)
        print(f"  从断点恢复: 已处理 {len(processed)} 个样本")
    
    remaining = [item for item in dataset if item["quid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    # 并行处理
    results = []
    pbar = tqdm(total=len(remaining), desc="NExTQA Uniform")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_item, item, num_frames, cache_dir): item
            for item in remaining
        }
        
        for future in as_completed(futures):
            result = future.result()
            uid = result["uid"]
            results.append(result)
            processed[uid] = result
            pbar.update(1)
            
            if len(results) % 50 == 0:
                _save_checkpoint(processed, "uniform", "nextqa")
    
    pbar.close()
    
    _save_checkpoint(processed, "uniform", "nextqa")
    
    # 评估
    pred_list = [{"quid": uid, "pred": r["pred"]} for uid, r in processed.items()]
    eval_result = evaluate_nextqa(pred_list, dataset)
    eval_result["method"] = f"uniform_{num_frames}frames"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"  NExT-QA 评测结果:")
    print(f"  Avg (Table 1): {eval_result['Avg']}")
    if "by_category" in eval_result:
        for cat, info in eval_result["by_category"].items():
            print(f"  {cat}: {info['accuracy_pct']}")
    print(f"  无效回答: {eval_result['invalid']}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"uniform_{num_frames}frames", "nextqa")
    return eval_result


def _save_checkpoint(processed, method, dataset_name):
    """保存中间结果"""
    output_dir = os.path.join(RESULTS_BASE_DIR, f"{method}_checkpoint", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    with open(checkpoint_path, 'w') as f:
        json.dump(processed, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("均匀取帧 Baseline 评测")
    parser.add_argument("--dataset", type=str, choices=["egoschema", "nextqa", "both"], default="both")
    parser.add_argument("--num_frames", type=int, default=UNIFORM_NUM_FRAMES)
    parser.add_argument("--max_workers", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--cache_dir", type=str, default="/tmp/benchmark_frames")
    parser.add_argument("--resume", type=str, default=None, help="从断点恢复的 checkpoint 路径")
    parser.add_argument("--egoschema_split", type=str, choices=["subset", "full", "both"], default="both",
                        help="EgoSchema 评测范围: subset(500), full(5031), 或 both")
    args = parser.parse_args()
    
    if args.dataset in ["egoschema", "both"]:
        splits = []
        if args.egoschema_split in ["subset", "both"]:
            splits.append(True)
        if args.egoschema_split in ["full", "both"]:
            splits.append(False)
        
        for subset_only in splits:
            split_tag = "egoschema_subset" if subset_only else "egoschema_full"
            resume_path = args.resume or os.path.join(
                RESULTS_BASE_DIR, "uniform_checkpoint", split_tag, "checkpoint.json")
            run_egoschema(args.num_frames, args.max_workers, args.cache_dir, resume_path, subset_only=subset_only)
    
    if args.dataset in ["nextqa", "both"]:
        resume_path = args.resume or os.path.join(
            RESULTS_BASE_DIR, "uniform_checkpoint", "nextqa", "checkpoint.json")
        run_nextqa(args.num_frames, args.max_workers, args.cache_dir, resume_path)
