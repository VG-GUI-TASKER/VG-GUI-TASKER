"""
方法4: VideoAgent (VLM-guided iterative frame selection) + VLM 理解

核心算法: 忠实于 ECCV 2024 原版 VideoAgent 的3步迭代策略:
  Step 1: 均匀采样 5 帧 → 让 VLM 回答 → 自我评估置信度
  Step 2: 置信度<3 → VLM描述需补充帧的segment → VLM在segment内选帧 → 再回答+评估
  Step 3: 置信度仍<3 → 再迭代一轮 → 最终回答

与原版的差异(为公平对比):
  - 用 VLM 替代 GPT-4 + 预提取 caption
  - VLM 直接看图做帧检索(替代 CLIP embedding 检索)
  - Final QA 使用统一的 build_vqa_prompt + call_qwen_vl (与 Uniform/VideoTree/TASKER 一致)
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import sys
import json
import re
import argparse
import cv2
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MAX_CONCURRENT_REQUESTS, RESULTS_BASE_DIR, SAMPLE_FPS
)
from api_utils import call_qwen_vl
from dataset_utils import (
    load_egoschema_dataset, load_nextqa_dataset,
    extract_frames_at_fps, extract_frames_by_indices,
    build_vqa_prompt, parse_answer,
    evaluate_egoschema, evaluate_nextqa,
    save_results, export_egoschema_submission
)


# ============================================================
#  VideoAgent 超参数
# ============================================================
VIDEOAGENT_NUM_INIT_FRAMES = 5   # Step 1 初始均匀采样帧数 (原版默认 5)
VIDEOAGENT_MAX_FRAMES = 16       # 最终最多选帧数 (与 Uniform 对齐)
VIDEOAGENT_MAX_ITERATIONS = 2    # 最大迭代轮数 (Step 2 + Step 3)
VIDEOAGENT_MAX_SEG_FRAMES = 16   # 帧检索时 segment 内最多展示帧数


# ============================================================
#  帧提取与缓存
# ============================================================

def extract_and_cache_frames(video_path, frame_indices, cache_dir):
    """懒加载：动态抽取所需帧并存入缓存目录"""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_cache_dir = os.path.join(cache_dir, video_id)
    os.makedirs(frame_cache_dir, exist_ok=True)
    
    img_paths = []
    cap = cv2.VideoCapture(video_path)
    for idx in frame_indices:
        frame_path = os.path.join(frame_cache_dir, f"va_{idx:06d}.jpg")
        if not os.path.exists(frame_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        if os.path.exists(frame_path):
            img_paths.append(frame_path)
    cap.release()
    return img_paths


# ============================================================
#  JSON 解析工具
# ============================================================

def parse_json_response(text):
    """从 LLM 回复中提取 JSON"""
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r"\{.*?\}|\[.*?\]"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return None


def parse_confidence(text):
    """解析置信度 (1-3)"""
    item = parse_json_response(text)
    try:
        c = int(item["confidence"])
        if c in range(1, 4):
            return c
    except:
        pass
    return 1


# ============================================================
#  VideoAgent 核心函数
# ============================================================

def ask_frames(img_paths, frame_labels, question, options_text, num_frames):
    """
    Step 1/2: 让 VLM 看帧图片并回答问题。
    对应原版 ask_gpt_caption / ask_gpt_caption_step。
    """
    frame_label_str = ", ".join([f"Frame {fl}" for fl in frame_labels])
    prompt = (
        f"Given a video that has {num_frames} frames (decoded at 1 fps). "
        f"You are presented with the following sampled frames: {frame_label_str}.\n\n"
        f"#C to denote the sentence is an action done by the camera wearer "
        f"(the person who recorded the video while wearing a camera on their head).\n"
        f"#O to denote that the sentence is an action done by someone other than the camera wearer.\n\n"
        f"Please answer the following question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f'Please think step-by-step and write the best answer index in JSON format '
        f'{{"final_answer": "X"}} where X is one of A, B, C, D, E.'
    )
    response = call_qwen_vl(
        question=prompt,
        image_paths=img_paths,
        system_prompt="You are a helpful video understanding assistant.",
        temperature=0.0,
        max_tokens=512,
    )
    return prompt, response


def self_eval(previous_prompt, answer, img_paths):
    """
    让 VLM 评估回答的置信度。
    对应原版 self_eval。
    """
    prompt = (
        f"Please assess the confidence level in the decision-making process.\n"
        f"The provided information is as follows:\n{previous_prompt}\n"
        f"The decision making process is as follows:\n{answer}\n\n"
        f"Criteria for Evaluation:\n"
        f"Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.\n"
        f"Partial Information (Confidence Level: 2): If information partially supports an informed guess.\n"
        f"Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.\n\n"
        f"Assessment Focus:\n"
        f"Evaluate based on the relevance, completeness, and clarity of the provided information "
        f"in relation to the decision-making context.\n"
        f'Please generate the confidence with JSON format {{"confidence": X}} where X is 1, 2, or 3.'
    )
    response = call_qwen_vl(
        question=prompt,
        image_paths=img_paths,
        system_prompt="You are a helpful assistant designed to output JSON.",
        temperature=0.0,
        max_tokens=64,
    )
    return response


def generate_description_step(img_paths, frame_labels, question, options_text, num_frames, segment_des):
    """
    让 VLM 描述哪些 segment 需要补充帧。
    对应原版 generate_description_step。
    """
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    frame_label_str = ", ".join([f"Frame {fl}" for fl in frame_labels])
    prompt = (
        f"Given a video that has {num_frames} frames (decoded at 1 fps). "
        f"You are presented with the following sampled frames: {frame_label_str}.\n\n"
        f"#C to denote the sentence is an action done by the camera wearer.\n"
        f"#O to denote that the sentence is an action done by someone other than the camera wearer.\n\n"
        f"To answer the following question:\n{question}\n"
        f"Options:\n{options_text}\n\n"
        f"However, the information in the initial frames is not sufficient.\n\n"
        f"Objective:\n"
        f"Identify additional frames that contain crucial information necessary for answering the question. "
        f"These frames should complement the insights from the initial frames.\n\n"
        f"Steps:\n"
        f"1. Divide the video into segments based on the intervals between initial frames. "
        f"Candidate segments: {segment_des}\n"
        f"2. Determine which segments are likely to contain frames most relevant to the question. "
        f"These frames should capture key visual elements (objects, humans, interactions, actions, scenes).\n\n"
        f"For each relevant frame, provide a concise description focusing on essential visual elements. "
        f"Select multiple frames from one segment if necessary.\n"
        f'Note "segment_id" must be smaller than {len(segment_des) + 1}.\n'
        f"Return JSON format:\n{json.dumps(formatted_description, indent=2)}"
    )
    response = call_qwen_vl(
        question=prompt,
        image_paths=img_paths,
        system_prompt="You are a helpful assistant designed to output JSON.",
        temperature=0.0,
        max_tokens=512,
    )
    return response


def frame_retrieval_by_vlm(description, all_frame_indices, all_img_paths, seg_start_pos, seg_end_pos):
    """
    VLM 直接看图选帧 (替代原版 CLIP embedding 检索)。
    在 segment 内均匀选一些候选帧展示给 VLM，让其选最匹配描述的帧。
    """
    # 找到该 segment 区间内的帧
    seg_frame_indices = []
    seg_img_paths = []
    for i, fidx in enumerate(all_frame_indices):
        if seg_start_pos <= fidx < seg_end_pos:
            seg_frame_indices.append(fidx)
            if i < len(all_img_paths):
                seg_img_paths.append(all_img_paths[i])
    
    if len(seg_frame_indices) == 0:
        return None
    if len(seg_frame_indices) == 1:
        return seg_frame_indices[0]
    
    # 限制展示帧数
    max_show = min(len(seg_frame_indices), VIDEOAGENT_MAX_SEG_FRAMES)
    step = max(1, len(seg_frame_indices) // max_show)
    sub_indices = seg_frame_indices[::step][:max_show]
    sub_paths = []
    for si in sub_indices:
        idx_in_all = all_frame_indices.index(si) if si in all_frame_indices else -1
        if idx_in_all >= 0 and idx_in_all < len(all_img_paths):
            sub_paths.append(all_img_paths[idx_in_all])
    
    if len(sub_paths) == 0:
        return seg_frame_indices[len(seg_frame_indices) // 2]
    
    description_text = description.get("description", "")
    frame_labels = [str(si) for si in sub_indices]
    prompt = (
        f"You are given {len(sub_paths)} frames from a video segment. "
        f"The frame numbers are: {', '.join(frame_labels)}. "
        f"Please select the ONE frame that best matches the following description:\n"
        f'"{description_text}"\n'
        f"Return ONLY the frame number as a single integer."
    )
    response = call_qwen_vl(
        question=prompt,
        image_paths=sub_paths,
        system_prompt="You are a helpful assistant that selects frames.",
        temperature=0.0,
        max_tokens=32,
    )
    
    # 解析返回的帧号
    try:
        selected = int(re.search(r'\d+', response).group())
        if selected in sub_indices:
            return selected
        # 选最近的
        return min(sub_indices, key=lambda x: abs(x - selected))
    except:
        return seg_frame_indices[len(seg_frame_indices) // 2]


# ============================================================
#  VideoAgent 主流程
# ============================================================

def videoagent_select_frames(
    video_path: str,
    question: str,
    options: list,
    cache_dir: str,
    num_init_frames: int = VIDEOAGENT_NUM_INIT_FRAMES,
    max_frames: int = VIDEOAGENT_MAX_FRAMES,
    max_iterations: int = VIDEOAGENT_MAX_ITERATIONS,
    sample_fps: float = SAMPLE_FPS,
) -> dict:
    """
    VideoAgent 3 步迭代选帧。
    
    流程:
    Step 1: 均匀采样 num_init_frames 帧 → VLM 回答 → 自我评估置信度
    Step 2: 置信度<3 → VLM 描述需补充帧的 segment → VLM 帧检索 → 再回答+评估
    Step 3: 置信度仍<3 → 再迭代一轮 → 最终回答
    
    Returns:
        dict with selected_frame_paths, selected_frame_indices, num_steps, confidence
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if num_frames_total <= 0 or video_fps <= 0:
        return {
            "selected_frame_paths": [],
            "selected_frame_indices": [],
            "num_frames": 0,
            "num_steps": 0,
            "confidence": 0,
            "stop_reason": "video_error",
        }
    
    # 按 FPS 采样所有候选帧
    interval = max(1, int(video_fps / sample_fps))
    all_frame_indices = list(range(0, num_frames_total, interval))
    num_frames = len(all_frame_indices)  # 对应原版的 num_frames (= len(caps))
    
    # 提取所有候选帧图片（缓存）
    all_img_paths = extract_and_cache_frames(video_path, all_frame_indices, cache_dir)
    
    # 构建 options_text
    option_labels = ['A', 'B', 'C', 'D', 'E']
    options_text = "\n".join([f"({l}) {o}" for l, o in zip(option_labels[:len(options)], options)])
    
    # ---- Step 1: 均匀采样 ----
    sample_idx = np.linspace(0, num_frames - 1, num=min(num_init_frames, num_frames), dtype=int).tolist()
    sample_frame_indices = [all_frame_indices[i] for i in sample_idx]
    sample_img_paths = [all_img_paths[i] for i in sample_idx if i < len(all_img_paths)]
    sample_labels = [i + 1 for i in sample_idx]  # 1-indexed labels
    
    # 回答 + 评估置信度
    previous_prompt, answer_str = ask_frames(
        sample_img_paths, sample_labels, question, options_text, num_frames
    )
    confidence_str = self_eval(previous_prompt, answer_str or "", sample_img_paths)
    confidence = parse_confidence(confidence_str)
    
    num_steps = 1
    
    # ---- Step 2 & Step 3: 迭代补充帧 ----
    for step_num in range(2, 2 + max_iterations):
        if confidence >= 3:
            break
        if len(sample_idx) >= max_frames:
            break
        
        try:
            # Segment 描述
            segment_des = {
                i + 1: f"{sample_labels[i]}-{sample_labels[i + 1]}"
                for i in range(len(sample_labels) - 1)
            }
            
            # VLM 描述需补充帧的 segment
            candidate_descriptions_str = generate_description_step(
                sample_img_paths, sample_labels, question, options_text,
                num_frames, segment_des
            )
            parsed_descriptions = parse_json_response(candidate_descriptions_str)
            
            if parsed_descriptions is None or "frame_descriptions" not in parsed_descriptions:
                num_steps += 1
                continue
            
            # VLM 帧检索
            sample_idx_positions = [all_frame_indices[i] for i in sample_idx]
            new_frame_indices = []
            
            for desc in parsed_descriptions["frame_descriptions"]:
                seg_id_val = None
                for key in desc:
                    if key.lower() == "segment_id":
                        nums = re.findall(r'\d+', str(desc[key]))
                        if nums:
                            seg_id_val = int(nums[0]) - 1  # 0-indexed
                        break
                
                if seg_id_val is None or seg_id_val < 0 or seg_id_val >= len(sample_idx_positions) - 1:
                    continue
                
                seg_start = sample_idx_positions[seg_id_val]
                seg_end = sample_idx_positions[seg_id_val + 1]
                
                retrieved = frame_retrieval_by_vlm(
                    desc, all_frame_indices, all_img_paths, seg_start, seg_end
                )
                if retrieved is not None:
                    new_frame_indices.append(retrieved)
            
            # 合并帧集合
            for nfi in new_frame_indices:
                if nfi in all_frame_indices:
                    pos = all_frame_indices.index(nfi)
                    if pos not in sample_idx:
                        sample_idx.append(pos)
            sample_idx = sorted(list(set(sample_idx)))
            
            # 限制最大帧数
            if len(sample_idx) > max_frames:
                sample_idx = sample_idx[:max_frames]
            
            # 更新
            sample_frame_indices = [all_frame_indices[i] for i in sample_idx]
            sample_img_paths = [all_img_paths[i] for i in sample_idx if i < len(all_img_paths)]
            sample_labels = [i + 1 for i in sample_idx]
            
            # 再次回答 + 评估
            if step_num < 2 + max_iterations - 1:
                previous_prompt, answer_str = ask_frames(
                    sample_img_paths, sample_labels, question, options_text, num_frames
                )
                confidence_str = self_eval(previous_prompt, answer_str or "", sample_img_paths)
                confidence = parse_confidence(confidence_str)
            
            num_steps += 1
        
        except Exception as e:
            num_steps += 1
            continue
    
    # 最终选出的帧
    final_frame_indices = [all_frame_indices[i] for i in sample_idx]
    final_frame_paths = [all_img_paths[i] for i in sample_idx if i < len(all_img_paths)]
    
    return {
        "selected_frame_paths": final_frame_paths,
        "selected_frame_indices": final_frame_indices,
        "num_frames": len(final_frame_paths),
        "num_steps": num_steps,
        "confidence": confidence,
        "stop_reason": "confidence_met" if confidence >= 3 else "max_iterations",
    }


# ============================================================
#  评测入口
# ============================================================

def run_single_item(item, cache_dir, max_frames, num_init_frames, max_iterations, sample_fps):
    """处理单个样本（选帧 + 统一 Final QA 解耦）"""
    video_path = item["video_path"]
    question = item["question"]
    options = item["options"]
    uid = item.get("uid") or item.get("quid")
    
    try:
        # === Stage 1: VideoAgent 选帧 ===
        selection = videoagent_select_frames(
            video_path, question, options, cache_dir,
            num_init_frames=num_init_frames,
            max_frames=max_frames,
            max_iterations=max_iterations,
            sample_fps=sample_fps,
        )
        
        selected_frames = selection["selected_frame_paths"]
        
        if not selected_frames:
            return {
                "uid": uid, "pred": -1, "num_frames": 0,
                "num_steps": selection["num_steps"],
                "stop_reason": selection["stop_reason"],
                "error": "no_frames_selected",
            }
        
        # === Stage 2: 统一 Final QA（与 Uniform/VideoTree/TASKER 一致） ===
        prompt = build_vqa_prompt(question, options, len(selected_frames))
        
        response = call_qwen_vl(
            question=prompt,
            image_paths=selected_frames,
            system_prompt=(
                "You are an expert video understanding assistant. "
                "You are shown key frames adaptively selected from a video. "
                "Analyze these frames carefully to answer the question."
            ),
            temperature=0.0,
            max_tokens=64,
        )
        
        pred = parse_answer(response)
        
        return {
            "uid": uid,
            "pred": pred,
            "num_frames": selection["num_frames"],
            "num_steps": selection["num_steps"],
            "confidence": selection["confidence"],
            "stop_reason": selection["stop_reason"],
            "selected_indices": selection["selected_frame_indices"],
        }
    
    except Exception as e:
        return {
            "uid": uid, "pred": -1, "num_frames": 0,
            "num_steps": 0, "stop_reason": "error", "error": str(e),
        }


def run_egoschema(args):
    """在 EgoSchema 上评测"""
    subset_only = getattr(args, 'egoschema_subset_only', True)
    split_name = "subset (500)" if subset_only else "full (5031)"
    split_tag = "egoschema_subset" if subset_only else "egoschema_full"
    
    print("=" * 60)
    print(f"方法: VideoAgent (init={args.num_init_frames}, max={args.max_frames}, iter={args.max_iterations})")
    print(f"数据集: EgoSchema {split_name}")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_egoschema_dataset(subset_only=subset_only)
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Checkpoint 支持
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "videoagent_checkpoint", split_tag)
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
        effective_workers = args.max_workers
        pbar = tqdm(total=len(remaining), desc="EgoSchema VideoAgent")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    run_single_item, item, args.cache_dir,
                    args.max_frames, args.num_init_frames,
                    args.max_iterations, args.sample_fps
                ): item
                for item in remaining
            }
            
            for future in as_completed(futures):
                result = future.result()
                processed[result["uid"]] = result
                pbar.update(1)
                
                if len(processed) % 5 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(processed, f, ensure_ascii=False)
        
        pbar.close()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(processed, f, ensure_ascii=False)
    
    # 评估
    predictions = {uid: r["pred"] for uid, r in processed.items()}
    gt = {item["uid"]: item["answer"] for item in dataset}
    eval_result = evaluate_egoschema(predictions, gt)
    eval_result["method"] = f"videoagent_{args.max_frames}frames"
    eval_result["config"] = {
        "num_init_frames": args.num_init_frames,
        "max_frames": args.max_frames,
        "max_iterations": args.max_iterations,
        "sample_fps": args.sample_fps,
    }
    
    # 统计
    avg_frames = np.mean([r.get("num_frames", 0) for r in processed.values()])
    avg_steps = np.mean([r.get("num_steps", 0) for r in processed.values()])
    eval_result["avg_frames_used"] = f"{avg_frames:.1f}"
    eval_result["avg_search_steps"] = f"{avg_steps:.1f}"
    
    print(f"\n{'='*40}")
    print(f"  EgoSchema {split_name} 结果:")
    print(f"  准确率: {eval_result['accuracy_pct']}")
    print(f"  平均使用帧数: {avg_frames:.1f}")
    print(f"  平均搜索步数: {avg_steps:.1f}")
    print(f"  无效预测: {eval_result['invalid']}")
    print(f"{'='*40}")
    
    save_results(eval_result, f"videoagent_{args.max_frames}frames", split_tag)
    
    # Full set: 导出提交文件
    if not subset_only:
        export_egoschema_submission(predictions, f"videoagent_{args.max_frames}frames")


def run_nextqa(args):
    """在 NExT-QA 上评测"""
    print("=" * 60)
    print(f"方法: VideoAgent (init={args.num_init_frames}, max={args.max_frames}, iter={args.max_iterations})")
    print(f"数据集: NExT-QA (MC test)")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_nextqa_dataset()
    print(f"  加载了 {len(dataset)} 个样本")
    
    # Checkpoint 支持
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "videoagent_checkpoint", "nextqa")
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
        effective_workers = args.max_workers
        pbar = tqdm(total=len(remaining), desc="NExTQA VideoAgent")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    run_single_item, item, args.cache_dir,
                    args.max_frames, args.num_init_frames,
                    args.max_iterations, args.sample_fps
                ): item
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
    eval_result["method"] = f"videoagent_{args.max_frames}frames"
    eval_result["config"] = {
        "num_init_frames": args.num_init_frames,
        "max_frames": args.max_frames,
        "max_iterations": args.max_iterations,
        "sample_fps": args.sample_fps,
    }
    
    # 统计
    avg_frames = np.mean([r.get("num_frames", 0) for r in processed.values()])
    avg_steps = np.mean([r.get("num_steps", 0) for r in processed.values()])
    eval_result["avg_frames_used"] = f"{avg_frames:.1f}"
    eval_result["avg_search_steps"] = f"{avg_steps:.1f}"
    
    print(f"\n{'='*40}")
    print(f"  NExT-QA 结果:")
    print(f"  Avg: {eval_result['Avg']}")
    print(f"  Temporal: {eval_result['Temporal']}")
    print(f"  Causal: {eval_result['Causal']}")
    print(f"  Descriptive: {eval_result['Descriptive']}")
    print(f"  平均使用帧数: {avg_frames:.1f}")
    print(f"  平均搜索步数: {avg_steps:.1f}")
    print(f"{'='*40}")
    
    save_results(eval_result, f"videoagent_{args.max_frames}frames", "nextqa")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VideoAgent 评测")
    parser.add_argument("--dataset", type=str, choices=["egoschema", "nextqa", "both"], default="both")
    parser.add_argument("--max_frames", type=int, default=VIDEOAGENT_MAX_FRAMES,
                        help="最大帧数上限 (与 Uniform 对齐)")
    parser.add_argument("--num_init_frames", type=int, default=VIDEOAGENT_NUM_INIT_FRAMES,
                        help="Step 1 初始均匀采样帧数 (原版默认 5)")
    parser.add_argument("--max_iterations", type=int, default=VIDEOAGENT_MAX_ITERATIONS,
                        help="最大迭代轮数 (Step 2 + Step 3)")
    parser.add_argument("--max_workers", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help="并行处理的视频数")
    parser.add_argument("--sample_fps", type=float, default=SAMPLE_FPS,
                        help="从视频中采样的 FPS")
    parser.add_argument("--cache_dir", type=str, default="/tmp/benchmark_frames",
                        help="帧缓存目录")
    parser.add_argument("--egoschema_split", type=str, choices=["subset", "full", "both"], default="subset",
                        help="EgoSchema 评测范围")
    args = parser.parse_args()
    
    if args.dataset in ["egoschema", "both"]:
        splits = []
        if args.egoschema_split in ["subset", "both"]:
            splits.append(True)
        if args.egoschema_split in ["full", "both"]:
            splits.append(False)
        
        for subset_only in splits:
            args.egoschema_subset_only = subset_only
            run_egoschema(args)
    
    if args.dataset in ["nextqa", "both"]:
        run_nextqa(args)
