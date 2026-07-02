"""
数据集加载和评估工具
支持 EgoSchema (subset 500) 和 NExT-QA
"""
import os
import json
import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from config import (
    DATA_ROOT, EGOSCHEMA_DIR, NEXTQA_DIR, RESULTS_BASE_DIR, SAMPLE_FPS
)


# ============================================================
#  数据集加载
# ============================================================

def load_egoschema_dataset(subset_only: bool = True) -> List[Dict]:
    """
    加载 EgoSchema 数据集
    
    Args:
        subset_only: True=只加载 subset (500个, 有 ground truth)
                     False=加载 full set (5031个, 无 ground truth, 只能提交到服务器评测)
    
    注意：EgoSchema Full Set (5031) 的 ground truth 不公开！
    只有 Subset (500) 可以本地评测。
    Full set 需要提交至 Kaggle 或 https://validation-server.onrender.com/api/upload/
    
    Returns:
        数据列表，每个元素包含 uid, question, options, answer, video_path
    """
    anno_filename = "subset_anno.json"  # 只有 subset 有 ground truth
    
    if not subset_only:
        print("  [WARNING] EgoSchema Full Set (5031) ground truth 不公开！")
        print("  [WARNING] 将生成预测结果用于提交至官方服务器评测")
        print("  [WARNING] 本地评测将使用 answer=-1 作为占位")
        # full set 使用 questions.json 或 fullset_anno.json
        anno_filename = "questions.json"
    
    # Look for the annotation file. The subset annotation ships with the repo
    # under videoqa/data/egoschema/; videos are downloaded by the user.
    possible_paths = [
        os.path.join(EGOSCHEMA_DIR, anno_filename),
        os.path.join(EGOSCHEMA_DIR, "annotations", anno_filename),
        os.path.join(DATA_ROOT, "egoschema", anno_filename),
    ]

    # The full set annotation may have several possible names.
    if not subset_only:
        extra_paths = [
            os.path.join(EGOSCHEMA_DIR, "fullset_anno.json"),
            os.path.join(EGOSCHEMA_DIR, "questions.json"),
            os.path.join(DATA_ROOT, "egoschema", "fullset_anno.json"),
            os.path.join(DATA_ROOT, "egoschema", "questions.json"),
        ]
        possible_paths.extend(extra_paths)
    
    anno = None
    for p in possible_paths:
        if os.path.exists(p):
            print(f"  加载标注文件: {p}")
            with open(p, 'r') as f:
                anno = json.load(f)
            break
    
    if anno is None:
        raise FileNotFoundError(
            f"无法找到 EgoSchema 标注文件! 尝试过的路径:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    # 寻找视频目录
    video_dirs = [
        os.path.join(EGOSCHEMA_DIR, "videos"),
        os.path.join(EGOSCHEMA_DIR, "video"),
        EGOSCHEMA_DIR,  # 视频可能直接在根目录
    ]
    video_dir = None
    for vd in video_dirs:
        if os.path.exists(vd):
            # 检查是否有 mp4 文件
            if any(f.endswith('.mp4') for f in os.listdir(vd) if os.path.isfile(os.path.join(vd, f))):
                video_dir = vd
                break
    
    if video_dir is None:
        # 递归搜索
        for root, dirs, files in os.walk(EGOSCHEMA_DIR):
            if any(f.endswith('.mp4') for f in files):
                video_dir = root
                break
    
    if video_dir is None:
        print(f"[WARNING] 找不到视频目录，将仅使用视频 ID 作为路径占位符")
        video_dir = EGOSCHEMA_DIR
    
    dataset = []
    # 支持两种格式：
    # 1. dict format (subset_anno.json): {uid: {question, option 0, ..., truth}, ...}
    # 2. list format (官方 questions.json): [{q_uid, question, option 0, ...}, ...]
    if isinstance(anno, list):
        for item in anno:
            uid = item.get("q_uid", item.get("uid", ""))
            video_path = os.path.join(video_dir, f"{uid}.mp4")
            dataset.append({
                "uid": uid,
                "question": item["question"],
                "options": [item[f"option {i}"] for i in range(5)],
                "answer": item.get("truth", -1),
                "video_path": video_path,
            })
    else:
        for uid, item in anno.items():
            video_path = os.path.join(video_dir, f"{uid}.mp4")
            dataset.append({
                "uid": uid,
                "question": item["question"],
                "options": [item[f"option {i}"] for i in range(5)],
                "answer": item.get("truth", -1),
                "video_path": video_path,
            })
    
    return dataset


def load_nextqa_dataset() -> List[Dict]:
    """
    加载 NExT-QA 数据集（MC test split）
    
    Returns:
        数据列表，每个元素包含 quid, video_id, question, options, answer, q_type, video_path
    """
    import pandas as pd
    
    # 查找标注 CSV
    possible_anno_paths = [
        os.path.join(NEXTQA_DIR, "test.csv"),
        os.path.join(NEXTQA_DIR, "val.csv"),
        os.path.join(NEXTQA_DIR, "annotations", "test.csv"),
        os.path.join(NEXTQA_DIR, "dataset", "nextqa", "test.csv"),
    ]
    
    anno_df = None
    
    for p in possible_anno_paths:
        if os.path.exists(p):
            anno_df = pd.read_csv(p)
            print(f"  加载 NExT-QA 标注: {p}")
            break
    
    if anno_df is None:
        # 尝试 parquet
        import glob as glob_module
        
        # 直接搜索 MC 目录下的 test parquet
        mc_test_patterns = [
            os.path.join(NEXTQA_DIR, "MC", "test*.parquet"),
            os.path.join(NEXTQA_DIR, "MC", "test", "*.parquet"),
            os.path.join(NEXTQA_DIR, "data", "MC-test*.parquet"),
        ]
        
        for pattern in mc_test_patterns:
            parquet_files = glob_module.glob(pattern)
            if parquet_files:
                anno_df = pd.concat([pd.read_parquet(f) for f in sorted(parquet_files)])
                print(f"  加载 NExT-QA parquet ({len(parquet_files)} 文件): {pattern}")
                break
    
    if anno_df is None:
        # 最后尝试递归搜索
        for root, dirs, files in os.walk(NEXTQA_DIR):
            for f in files:
                if f.endswith('.csv') and 'test' in f.lower():
                    anno_df = pd.read_csv(os.path.join(root, f))
                    print(f"  加载 NExT-QA: {os.path.join(root, f)}")
                    break
            if anno_df is not None:
                break
    
    if anno_df is None:
        raise FileNotFoundError("无法找到 NExT-QA 标注文件")
    
    # 查找视频目录
    video_dirs = [
        os.path.join(NEXTQA_DIR, "videos"),
        os.path.join(NEXTQA_DIR, "video"),
        os.path.join(NEXTQA_DIR, "NExTVideo"),
        NEXTQA_DIR,
    ]
    video_dir = None
    for vd in video_dirs:
        if os.path.exists(vd):
            if any(f.endswith(('.mp4', '.avi', '.mkv', '.webm')) for f in os.listdir(vd) if os.path.isfile(os.path.join(vd, f))):
                video_dir = vd
                break
    
    if video_dir is None:
        for root, dirs, files in os.walk(NEXTQA_DIR):
            if any(f.endswith(('.mp4', '.avi', '.mkv', '.webm')) for f in files):
                video_dir = root
                break
    
    if video_dir is None:
        print(f"[WARNING] 找不到 NExT-QA 视频目录")
        video_dir = NEXTQA_DIR
    
    dataset = []
    for _, row in anno_df.iterrows():
        video_id = str(row['video'])
        qid = row['qid']
        quid = f"{video_id}_{qid}"
        
        # NExT-QA 视频命名可能有多种格式
        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            candidate = os.path.join(video_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break
        if video_path is None:
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        dataset.append({
            "quid": quid,
            "video_id": video_id,
            "question": row['question'],
            "options": [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']],
            "answer": int(row['answer']),
            "q_type": row['type'],
            "video_path": video_path,
        })
    
    return dataset


# ============================================================
#  视频帧提取
# ============================================================

def extract_frames_uniform(video_path: str, num_frames: int, cache_dir: str = "/tmp/benchmark_frames") -> List[str]:
    """
    从视频中均匀提取指定数量的帧
    
    Args:
        video_path: 视频路径
        num_frames: 需要提取的帧数
        cache_dir: 帧缓存目录
    
    Returns:
        帧图片路径列表
    """
    video_id = Path(video_path).stem
    frame_cache_dir = os.path.join(cache_dir, video_id)
    os.makedirs(frame_cache_dir, exist_ok=True)
    
    # 检查是否已有缓存
    cache_marker = os.path.join(frame_cache_dir, f".uniform_{num_frames}")
    if os.path.exists(cache_marker):
        frame_paths = sorted([
            os.path.join(frame_cache_dir, f) 
            for f in os.listdir(frame_cache_dir) 
            if f.startswith("uniform_") and f.endswith(".jpg")
        ])
        if len(frame_paths) >= num_frames:
            return frame_paths[:num_frames]
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        print(f"  [WARNING] 无法读取视频帧数: {video_path}")
        return []
    
    # 均匀选取帧索引
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frame_paths = []
    for idx in frame_indices:
        frame_path = os.path.join(frame_cache_dir, f"uniform_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    
    cap.release()
    
    # 写入缓存标记
    with open(cache_marker, 'w') as f:
        f.write(str(len(frame_paths)))
    
    return frame_paths


def extract_frames_at_fps(video_path: str, fps: float = 1.0, cache_dir: str = "/tmp/benchmark_frames") -> Tuple[List[str], List[int]]:
    """
    按指定 FPS 从视频提取帧
    
    Args:
        video_path: 视频路径
        fps: 采样 FPS
        cache_dir: 帧缓存目录
    
    Returns:
        (帧路径列表, 帧索引列表)
    """
    video_id = Path(video_path).stem
    frame_cache_dir = os.path.join(cache_dir, video_id)
    os.makedirs(frame_cache_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        return [], []
    
    interval = max(1, int(video_fps / fps))
    frame_indices = list(range(0, total_frames, interval))
    
    frame_paths = []
    for idx in frame_indices:
        frame_path = os.path.join(frame_cache_dir, f"fps_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths, frame_indices


def extract_frames_by_indices(video_path: str, frame_indices: List[int], cache_dir: str = "/tmp/benchmark_frames") -> List[str]:
    """
    按指定帧索引提取帧
    
    Args:
        video_path: 视频路径  
        frame_indices: 帧索引列表
        cache_dir: 帧缓存目录
    
    Returns:
        帧路径列表
    """
    video_id = Path(video_path).stem
    frame_cache_dir = os.path.join(cache_dir, video_id)
    os.makedirs(frame_cache_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    
    for idx in frame_indices:
        frame_path = os.path.join(frame_cache_dir, f"sel_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths


# ============================================================
#  VQA prompt 构建
# ============================================================

def build_vqa_prompt(question: str, options: List[str], num_frames: int) -> str:
    """
    构建 VQA 多选题 prompt
    
    Args:
        question: 问题
        options: 选项列表 (5个)
        num_frames: 输入帧数
    """
    option_labels = ['A', 'B', 'C', 'D', 'E']
    options_text = "\n".join([f"({label}) {opt}" for label, opt in zip(option_labels, options)])
    
    prompt = (
        f"You are shown {num_frames} frames sampled from a video. "
        f"Please answer the following multiple-choice question about the video.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Please respond with ONLY the letter of the correct answer (A, B, C, D, or E). "
        f"Do not provide any explanation."
    )
    return prompt


def parse_answer(response: Optional[str]) -> int:
    """
    从模型回复中解析答案
    
    Returns:
        答案索引 (0-4)，解析失败返回 -1
    """
    if response is None:
        return -1
    
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # 尝试直接匹配第一个字母
    response = response.strip()
    if response and response[0].upper() in mapping:
        return mapping[response[0].upper()]
    
    # 尝试匹配括号格式 (A), (B) 等
    match = re.search(r'\(([A-E])\)', response, re.IGNORECASE)
    if match:
        return mapping[match.group(1).upper()]
    
    # 尝试匹配 "Answer: X" 格式
    match = re.search(r'(?:answer|choice|option)[:\s]*([A-E])', response, re.IGNORECASE)
    if match:
        return mapping[match.group(1).upper()]
    
    # 尝试在整个回复中找到单个字母
    letters_found = re.findall(r'\b([A-E])\b', response)
    if len(letters_found) == 1:
        return mapping[letters_found[0].upper()]
    
    return -1


# ============================================================
#  评估指标
# ============================================================

def evaluate_egoschema(predictions: Dict[str, int], ground_truth: Dict[str, int]) -> Dict:
    """
    评估 EgoSchema 结果
    
    忠实于原始仓库的评估方式：
    - VideoTree/eval.py: acc = num_corrects / len(data)
      pred=-1 的样本不计为正确，但总数仍为全部样本数
    - AKeyS/eval.py: 无答案按 0.2 (5选1随机概率) 计入
    
    这里同时输出两种指标。
    
    Args:
        predictions: {uid: predicted_answer}
        ground_truth: {uid: correct_answer}
    
    Returns:
        评估结果字典
    """
    num_total = len(ground_truth)
    num_corrects = 0
    num_valids = 0
    num_invalid = 0
    
    for uid, gt in ground_truth.items():
        pred = predictions.get(uid, -1)
        if pred == -1:
            num_invalid += 1
        else:
            num_valids += 1
            if pred == gt:
                num_corrects += 1
    
    # 主准确率：与 VideoTree/eval.py 一致
    # acc = num_corrects / len(data)，pred=-1 不算正确但不从分母排除
    accuracy = num_corrects / num_total if num_total > 0 else 0
    
    # AKeyS 方式：无答案按随机概率 0.2 计入
    accuracy_with_random = (num_corrects + num_invalid * 0.2) / num_total if num_total > 0 else 0
    
    return {
        "dataset": "EgoSchema",
        "total": num_total,
        "num_valids": num_valids,
        "correct": num_corrects,
        "invalid": num_invalid,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy*100:.2f}%",
        "accuracy_with_random_guess": accuracy_with_random,
        "accuracy_with_random_guess_pct": f"{accuracy_with_random*100:.2f}%",
    }


def evaluate_nextqa(predictions: List[Dict], dataset: List[Dict]) -> Dict:
    """
    评估 NExT-QA 结果，按问题类型分别计算准确率
    
    忠实于原始 VideoTree/eval.py 的评估方式:
    - TP 合并到 TN: if qtype == 'TP': qtype = 'TN'
    - 子类型: CW, CH, TN(含TP), TC, DC, DL, DO
    - 大类: C(CW+CH), T(TN+TC), D(DC+DL+DO)
    - Avg = all_correct / all_total (加权整体准确率，与原版一致)
    - pred=-1 计入分母（算错），与 EgoSchema 评估一致
    
    Args:
        predictions: [{quid, pred, ...}, ...]
        dataset: 原始数据集
    
    Returns:
        评估结果字典，包含 Table 1 所需的 Tem., Cau., Des., Avg. 指标
    """
    # 构建 quid -> prediction 映射
    pred_map = {p["quid"]: p["pred"] for p in predictions}
    
    # 按类型分组（忠实于原版：TP → TN）
    # 原版 group 定义: {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    group = defaultdict(list)  # q_type -> [(quid, pred, gt), ...]
    
    total_invalid = 0
    
    for item in dataset:
        quid = item["quid"]
        q_type = item["q_type"]
        gt = item["answer"]
        
        # 忠实于原版：TP 合并到 TN
        if q_type == "TP":
            q_type = "TN"
        
        pred = pred_map.get(quid, -1)
        if pred == -1:
            total_invalid += 1
        
        group[q_type].append((quid, pred, gt))
    
    # 计算各子类型准确率
    type_accuracy = {}
    
    # 大类累加器 (忠实于原版: overall_acc/overall_cnt)
    cat_correct = {"C": 0, "T": 0, "D": 0}
    cat_total = {"C": 0, "T": 0, "D": 0}
    all_correct = 0
    all_total = 0
    
    for q_type in ["CW", "CH", "TN", "TC", "DC", "DL", "DO"]:
        items = group.get(q_type, [])
        cnt = len(items)
        acc = sum(1 for _, pred, gt in items if pred == gt)
        
        type_accuracy[q_type] = {
            "total": cnt,
            "correct": acc,
            "accuracy": acc / cnt * 100 if cnt > 0 else 0,
            "accuracy_pct": f"{acc / cnt * 100:.2f}%" if cnt > 0 else "0.00%",
        }
        
        # 大类累加 (用子类型首字母)
        cat_key = q_type[0]  # C, T, or D
        cat_correct[cat_key] += acc
        cat_total[cat_key] += cnt
        all_correct += acc
        all_total += cnt
    
    # 大类准确率 (Table 1: Cau., Tem., Des.)
    category_accuracy = {}
    cat_name_map = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}
    
    for cat_key in ["C", "T", "D"]:
        cat_name = cat_name_map[cat_key]
        if cat_total[cat_key] > 0:
            acc = cat_correct[cat_key] / cat_total[cat_key]
            category_accuracy[cat_name] = {
                "total": cat_total[cat_key],
                "correct": cat_correct[cat_key],
                "accuracy": acc,
                "accuracy_pct": f"{acc*100:.2f}%",
            }
    
    # Avg: 加权整体准确率 (忠实于原版: all_acc / all_cnt)
    overall_acc = all_correct / all_total if all_total > 0 else 0
    
    return {
        "dataset": "NExT-QA",
        "total": len(dataset),
        "evaluated": all_total,
        "correct": all_correct,
        "invalid": total_invalid,
        "accuracy": overall_acc,
        "accuracy_pct": f"{overall_acc*100:.2f}%",
        "by_type": type_accuracy,
        "by_category": category_accuracy,
        # Table 1 指标快捷访问
        "Temporal": category_accuracy.get("Temporal", {}).get("accuracy_pct", "N/A"),
        "Causal": category_accuracy.get("Causal", {}).get("accuracy_pct", "N/A"),
        "Descriptive": category_accuracy.get("Descriptive", {}).get("accuracy_pct", "N/A"),
        "Avg": f"{overall_acc*100:.2f}%",
    }


# ============================================================
#  结果保存
# ============================================================

def save_results(results: Dict, method_name: str, dataset_name: str):
    """保存评测结果"""
    output_dir = os.path.join(RESULTS_BASE_DIR, method_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n  结果已保存: {output_path}")
    return output_path


def export_egoschema_submission(predictions: Dict[str, int], method_name: str):
    """
    导出 EgoSchema Full Set 提交文件。
    
    官方格式: {q_uid: predicted_answer_index}
    提交地址: https://validation-server.onrender.com/api/upload/
    
    Args:
        predictions: {uid: predicted_answer (0-4)}
        method_name: 方法名（用于文件命名）
    
    Returns:
        导出文件路径
    """
    output_dir = os.path.join(RESULTS_BASE_DIR, method_name, "egoschema_full")
    os.makedirs(output_dir, exist_ok=True)
    
    # 官方提交格式: {q_uid: answer_index}
    submission = {}
    for uid, pred in predictions.items():
        # pred=-1 时默认填 0（不能提交-1）
        submission[uid] = pred if pred >= 0 else 0
    
    submission_path = os.path.join(output_dir, "submission.json")
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n  ★ EgoSchema Full 提交文件已保存: {submission_path}")
    print(f"    共 {len(submission)} 个预测")
    print(f"    提交地址: https://validation-server.onrender.com/api/upload/")
    print(f"    或 Kaggle: https://www.kaggle.com/competitions/egoschema-challenge")
    
    return submission_path
