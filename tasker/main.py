import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import re
import concurrent.futures
import threading

# 引入 Qwen API 工具
from qwen.vllm_tool import get_api
# 保留原仓库的视频分割与节点树工具
from video_seg import VideoSeg
from arg_parser import parse_args

# ================= 路径配置（由命令行参数传入，此处声明全局变量） =================
VIDEO_DIR = None
JSON_DIR = None
OUT_ROOT = None
CACHE_DIR = None
RECORD_JSON_PATH = None

def extract_and_cache_frames(video_path, sample_idx, video_id):
    """懒加载：利用 cv2 动态抽取所需帧并存入缓存目录，返回图片路径列表"""
    img_paths = []
    cap = cv2.VideoCapture(video_path)
    for idx in sample_idx:
        frame_path = os.path.join(CACHE_DIR, f"{video_id}_frame_{idx:04d}.jpg")
        if not os.path.exists(frame_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        if os.path.exists(frame_path):
            img_paths.append(frame_path)
    cap.release()
    return img_paths

def parse_json(response):
    """简单的 JSON 提取器"""
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(json_str)
    except:
        return None

# ================= 视觉去重工具 =================
def compute_color_histogram(img_path):
    """计算图片的颜色直方图特征（3通道联合直方图）"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    hist = cv2.calcHist([img], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def frame_similarity(hist1, hist2):
    """计算两个颜色直方图之间的相关系数相似度，返回值范围 [-1, 1]"""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def is_frame_redundant(new_frame_path, existing_frame_paths, hist_cache, threshold=0.985):
    """
    检查新帧是否与已选帧过于相似（视觉去重）
    阈值设为0.985，过滤掉视觉上高度相似的冗余帧
    """
    new_hist = hist_cache.get(new_frame_path)
    if new_hist is None:
        new_hist = compute_color_histogram(new_frame_path)
        hist_cache[new_frame_path] = new_hist
    if new_hist is None:
        return False
    
    for existing_path in existing_frame_paths:
        existing_hist = hist_cache.get(existing_path)
        if existing_hist is None:
            existing_hist = compute_color_histogram(existing_path)
            hist_cache[existing_path] = existing_hist
        if existing_hist is None:
            continue
        sim = frame_similarity(new_hist, existing_hist)
        if sim >= threshold:
            return True
    return False

# ================= 视觉变化分割点（改进版：选过渡后稳定帧 + 模糊度检测） =================

def _frame_hist(frame):
    """计算单帧的归一化颜色直方图"""
    hist = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def _hist_similarity(hist_a, hist_b):
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

def _laplacian_variance(frame):
    """计算拉普拉斯方差作为清晰度指标，值越大越清晰"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _read_frame(cap, idx):
    """读取指定帧"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None

def find_visual_change_split_point(video_path, segment_start, segment_end, video_id):
    """
    改进版分割点选择：
    1. 在segment内粗采样，找到视觉变化最大的相邻对 (idx_a, idx_b)
    2. 在 [idx_a, idx_b] 内密集采样，从 idx_b 往前找"过渡结束后的第一个稳定帧"
    3. 对候选帧做模糊度检测，如果太模糊则向前/后搜索更清晰的帧
    Fallback: 出错或分割点太偏时回退到中点
    """
    midpoint = (segment_start + segment_end) // 2
    try:
        seg_length = segment_end - segment_start
        if seg_length <= 2:
            return midpoint
        
        cap = cv2.VideoCapture(video_path)
        
        # === 阶段1：粗采样找最大变化区间 ===
        num_samples = min(seg_length, 10)
        step = max(1, seg_length // num_samples)
        sample_indices = list(range(segment_start, segment_end, step))
        if sample_indices[-1] != segment_end:
            sample_indices.append(segment_end)
        
        frames = {}
        hists = {}
        for idx in sample_indices:
            f = _read_frame(cap, idx)
            if f is not None:
                frames[idx] = f
                hists[idx] = _frame_hist(f)
        
        if len(frames) < 2:
            cap.release()
            return midpoint
        
        sorted_indices = sorted(frames.keys())
        max_diff = -1
        best_a, best_b = sorted_indices[0], sorted_indices[-1]
        
        for i in range(len(sorted_indices) - 1):
            idx_a, idx_b = sorted_indices[i], sorted_indices[i + 1]
            if idx_a in hists and idx_b in hists:
                diff = 1.0 - _hist_similarity(hists[idx_a], hists[idx_b])
                if diff > max_diff:
                    max_diff = diff
                    best_a, best_b = idx_a, idx_b
        
        # === 阶段2：在最大变化区间内密集采样，从后往前找稳定帧 ===
        transition_len = best_b - best_a
        if transition_len <= 2:
            # 区间太小，直接取 best_b 作为稳定帧
            candidate = best_b
        else:
            # 密集采样：最多20个点
            dense_step = max(1, transition_len // 20)
            dense_indices = list(range(best_a, best_b + 1, dense_step))
            if dense_indices[-1] != best_b:
                dense_indices.append(best_b)
            
            # 读取 best_b 的直方图作为"稳定目标"
            ref_hist = hists.get(best_b)
            if ref_hist is None:
                f = _read_frame(cap, best_b)
                if f is not None:
                    frames[best_b] = f
                    ref_hist = _frame_hist(f)
                    hists[best_b] = ref_hist
            
            if ref_hist is None:
                cap.release()
                return midpoint
            
            # 从后往前找第一个与 best_b 相似度 >= 0.99 的帧（过渡结束点）
            stable_idx = best_b
            for k in range(len(dense_indices) - 1, -1, -1):
                didx = dense_indices[k]
                if didx not in hists:
                    f = _read_frame(cap, didx)
                    if f is not None:
                        frames[didx] = f
                        hists[didx] = _frame_hist(f)
                if didx in hists:
                    sim = _hist_similarity(hists[didx], ref_hist)
                    if sim >= 0.99:
                        stable_idx = didx
                    else:
                        break  # 不再稳定，停止
            candidate = stable_idx
        
        # === 阶段3：模糊度检测，如果候选帧太模糊则搜索附近更清晰的帧 ===
        if candidate not in frames:
            f = _read_frame(cap, candidate)
            if f is not None:
                frames[candidate] = f
        
        if candidate in frames:
            blur_score = _laplacian_variance(frames[candidate])
            BLUR_THRESHOLD = 100  # 低于此值认为模糊
            
            if blur_score < BLUR_THRESHOLD:
                # 在候选帧前后各搜索若干帧，找最清晰的
                search_range = max(3, transition_len // 10)
                best_clarity = blur_score
                best_clear_idx = candidate
                
                for offset in range(-search_range, search_range + 1):
                    check_idx = candidate + offset
                    if check_idx <= segment_start or check_idx >= segment_end:
                        continue
                    if check_idx not in frames:
                        f = _read_frame(cap, check_idx)
                        if f is not None:
                            frames[check_idx] = f
                    if check_idx in frames:
                        clarity = _laplacian_variance(frames[check_idx])
                        if clarity > best_clarity:
                            best_clarity = clarity
                            best_clear_idx = check_idx
                
                if best_clear_idx != candidate:
                    print(f"  [Anti-Blur] Frame {candidate} blur={blur_score:.1f} -> replaced with {best_clear_idx} clarity={best_clarity:.1f}")
                    candidate = best_clear_idx
        
        cap.release()
        
        # === Fallback：分割点太偏则回退中点 ===
        min_pos = segment_start + int(seg_length * 0.15)
        max_pos = segment_start + int(seg_length * 0.85)
        if candidate < min_pos or candidate > max_pos:
            print(f"  [Visual Split] Candidate {candidate} too biased (range [{min_pos},{max_pos}]), using midpoint.")
            return midpoint
        
        return candidate
    except Exception as e:
        print(f"[Visual Split Fallback] Error: {e}, using midpoint.")
        return midpoint

# ================= 多模态大模型调用封装 =================
def qwen_call(model, prompt, img_paths, is_json=True, step_info=""):
    system_prompt = "You are a highly strict UI navigation assistant designed to output JSON." if is_json else "You are a highly strict UI navigation assistant."

    response = model(
        img_path_or_list=img_paths, 
        question=prompt,
        system_prompt=system_prompt,
        image_first=True
    )
    
    # 【新增功能 2】：将模型的推理 response 打印出来，方便跟踪过程
    print(f"\n{'='*20} Qwen Response ({step_info}) {'='*20}")
    print(response)
    print(f"{'='*60}\n")
    
    return response

# ================= 逐帧相关性过滤（替代旧片头片尾检测） =================
def check_frame_relevance(model, goal, frame_path):
    """
    检查单帧是否与 goal 相关。
    返回 True 表示相关（保留），False 表示无关（片头/片尾/广告等，应丢弃）
    """
    prompt = f"""This frame is from a UI tutorial video. Goal: {goal}
Does this frame show any actual GUI/app interface related to the goal?
Answer "no" ONLY if it is clearly an intro/outro/ad/splash/credits/title screen with NO GUI operation.
Output JSON: {{"is_relevant": "yes"}} or {{"is_relevant": "no"}}"""
    response = qwen_call(model, prompt, [frame_path], is_json=True, step_info="Relevance Check")
    result = parse_json(response)
    if result and result.get("is_relevant", "yes").lower() == "no":
        return False
    return True

def filter_irrelevant_frames(model, goal, sample_idx, video_path, video_id, content_bounds, fps, num_frames):
    """
    片头片尾过滤：从首帧开始检查，如果无关则向后跳2秒再检查，直到找到有关帧。
    片尾同理从末帧向前跳2秒。收缩 content_bounds 并重建 sample_idx。
    """
    jump_frames = int(fps * 2)  # 每次跳2秒
    
    # --- 片头检测：从帧0开始，每次向后跳2秒 ---
    head_cursor = 0
    max_head = content_bounds[1] - jump_frames  # 不能跳过中间
    while head_cursor <= max_head:
        frame_paths = extract_and_cache_frames(video_path, [head_cursor], video_id)
        if frame_paths and not check_frame_relevance(model, goal, frame_paths[0]):
            print(f"  [Relevance Filter] Head frame {head_cursor} is irrelevant, jumping +{jump_frames} frames (2s)")
            head_cursor += jump_frames
        else:
            break  # 找到有关帧，停止
    
    # head_cursor 现在指向第一个有关帧（或跳过后的位置）
    if head_cursor > 0:
        content_bounds[0] = head_cursor
        print(f"  [Relevance Filter] Head content starts at frame {head_cursor}")
    
    # --- 片尾检测：从最后一帧开始，每次向前跳2秒 ---
    tail_cursor = num_frames - 1
    min_tail = content_bounds[0] + jump_frames  # 不能跳过片头
    while tail_cursor >= min_tail:
        frame_paths = extract_and_cache_frames(video_path, [tail_cursor], video_id)
        if frame_paths and not check_frame_relevance(model, goal, frame_paths[0]):
            print(f"  [Relevance Filter] Tail frame {tail_cursor} is irrelevant, jumping -{jump_frames} frames (2s)")
            tail_cursor -= jump_frames
        else:
            break
    
    if tail_cursor < num_frames - 1:
        content_bounds[1] = tail_cursor
        print(f"  [Relevance Filter] Tail content ends at frame {tail_cursor}")
    
    # 用收缩后的区间过滤初始 sample_idx，去掉落在区间外的帧
    new_start, new_end = content_bounds
    sample_idx = [idx for idx in sample_idx if new_start <= idx <= new_end]
    
    if not sample_idx:
        # 极端情况：所有初始帧都被过滤掉了，用区间端点
        sample_idx = [new_start, new_end]
    
    print(f"  [Relevance Filter] Final content range: [{new_start}, {new_end}], remaining frames: {len(sample_idx)}")
    return sample_idx

def check_new_frame_relevance(model, goal, new_frame_idx, video_path, video_id, sample_idx, content_bounds):
    """
    检查树搜索中新增的单帧是否相关。
    如果无关，在该帧前后搜索一个有关帧作为替代。
    返回值：
      - (True, new_frame_idx)：原帧有关，保留
      - (True, replacement_idx)：原帧无关，找到了替代帧
      - (False, None)：原帧无关，且找不到替代帧
    """
    def _extract_and_check(idx):
        """提取单帧并检查相关性"""
        path = os.path.join(CACHE_DIR, f"{video_id}_frame_{idx:04d}.jpg")
        if not os.path.exists(path):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(path, frame)
            cap.release()
        if not os.path.exists(path):
            return None  # 无法读取
        return check_frame_relevance(model, goal, path)
    
    # 先检查原帧
    result = _extract_and_check(new_frame_idx)
    if result is None:
        return (True, new_frame_idx)  # 无法读取，默认保留
    if result:
        return (True, new_frame_idx)  # 有关，直接保留
    
    print(f"  [Relevance Filter] Frame {new_frame_idx} is irrelevant, searching nearby for replacement...")
    
    # 原帧无关，在前后的 sample_idx 邻居之间搜索替代帧
    # 找到 new_frame_idx 在 sample_idx 中的左右邻居
    sorted_all = sorted(set(sample_idx + [new_frame_idx]))
    pos = sorted_all.index(new_frame_idx)
    left_bound = sorted_all[pos - 1] + 1 if pos > 0 else content_bounds[0]
    right_bound = sorted_all[pos + 1] - 1 if pos < len(sorted_all) - 1 else content_bounds[1]
    
    # 从中心向两侧交替搜索，步长为区间的 10%（至少5帧）
    search_step = max(5, (right_bound - left_bound) // 10)
    max_search_attempts = 5  # 最多额外检查5帧
    
    for attempt in range(1, max_search_attempts + 1):
        offset = attempt * search_step
        # 向右搜索
        candidate_right = new_frame_idx + offset
        if candidate_right <= right_bound and candidate_right not in sample_idx:
            result = _extract_and_check(candidate_right)
            if result:
                print(f"  [Relevance Filter] Replaced frame {new_frame_idx} -> {candidate_right} (relevant)")
                return (True, candidate_right)
        # 向左搜索
        candidate_left = new_frame_idx - offset
        if candidate_left >= left_bound and candidate_left not in sample_idx:
            result = _extract_and_check(candidate_left)
            if result:
                print(f"  [Relevance Filter] Replaced frame {new_frame_idx} -> {candidate_left} (relevant)")
                return (True, candidate_left)
    
    # 找不到替代帧，丢弃
    print(f"  [Relevance Filter] No replacement found for frame {new_frame_idx}, dropping it.")
    return (False, None)

# ================= AKeyS 树搜索策略 (直接看图) =================
def bfs_select_segments(model, goal, img_paths, num_frames, segment_des):
    prompt = f"""
    You are provided with sequential images sampled from a UI interaction video.
    Each image is labeled with its frame index. The images are shown in chronological order.
    Goal: {goal}
    
    Candidate video segments (gaps between current frames):
    {segment_des}
    
    (BFS Strategy - Breadth Exploration)
    Determine which segments are likely to contain crucial missing UI actions necessary to achieve the Goal. 
    You are ALLOWED and ENCOURAGED to select MULTIPLE segments simultaneously if you believe important actions are missing in different video gaps.
    
    Return JSON format exactly like this (can contain one or multiple items): 
    {{"frame_descriptions": [
        {{"segment_id": "1", "duration": "xxx - xxx", "description": "Missing initial click"}}, 
        {{"segment_id": "3", "duration": "yyy - yyy", "description": "Missing confirmation popup"}}
    ]}}
    """
    return qwen_call(model, prompt, img_paths, is_json=True, step_info="BFS Select")

def gbfs_select_one_segment(model, goal, img_paths, num_frames, segment_des):
    prompt = f"""
    You are provided with sequential images sampled from a UI interaction video.
    Each image is labeled with its frame index. The images are shown in chronological order.
    Goal: {goal}
    
    Candidate segments (gaps between current frames):
    {segment_des}
    
    (GBFS Strategy - Focus on Missing Goal-Critical Actions)
    Determine which SINGLE segment is MOST LIKELY to contain the frames showing crucial missing UI actions needed to achieve the goal.
    Focus on gaps in the operation flow: where does the current frame sequence fail to explain HOW the user got from one state to the next?
    
    Return JSON format: {{"frame_descriptions": [{{"segment_id": "1", "duration": "xxx - xxx", "description": "Contains missing goal-critical actions"}}]}}
    """
    return qwen_call(model, prompt, img_paths, is_json=True, step_info="GBFS Select")

def dijkstra_select_one_segment(model, goal, img_paths, num_frames, segment_des):
    prompt = f"""
    You are provided with sequential images sampled from a UI interaction video.
    Each image is labeled with its frame index. The images are shown in chronological order.
    
    Candidate segments (gaps between current frames):
    {segment_des}
    
    (Dijkstra Strategy - Focus on UI State Changes)
    Identify which SINGLE candidate video segment contains the MOST significant UI state transition between its start frame and end frame.
    Look at the actual images for the start and end frames of each segment to judge the visual difference.
    
    In GUI operations, important UI state changes include (even if visually subtle):
    - Page or screen navigation (e.g., moving from one view to another)
    - Dialog boxes, pop-ups, or dropdown menus appearing or disappearing
    - Button click effects or toggle state changes
    - Text input or form field content changes
    - Loading spinners, progress bars, or status indicator updates
    - Sidebar, tab, or panel switching
    
    Prioritize the segment where the start frame and end frame represent the MOST DISTINCT operational states, even if the pixel-level difference appears small.
    
    Return JSON format: {{"frame_descriptions": [{{"segment_id": "1", "duration": "xxx - xxx", "description": "Most significant UI state transition"}}]}}
    """
    return qwen_call(model, prompt, img_paths, is_json=True, step_info="Dijkstra Select")

def a_star_select_one_segment(model, goal, img_paths, num_frames, segment_des):
    prompt = f"""
    You are provided with sequential images sampled from a UI interaction video.
    Each image is labeled with its frame index. The images are shown in chronological order.
    Goal: {goal}
    
    Candidate segments (gaps between current frames):
    {segment_des}
    
    (A* Strategy - Balance missing goal-relevant info and UI state changes)
    Identify ONE single candidate segment that BEST satisfies BOTH conditions simultaneously:
    1. GOAL PROXIMITY: The segment likely contains crucial missing UI actions that are necessary steps toward achieving the Goal. Without seeing the frames in this segment, the operation flow has an unexplained gap.
    2. STATE CHANGE MAGNITUDE: Look at the start frame and end frame images of each segment. The segment whose boundary frames show the MOST different UI states is more likely to contain important operations.
    
    In GUI operations, even subtle visual differences can represent critical steps (e.g., a single checkbox toggle, a dropdown selection, text typed into a field). Do NOT dismiss segments just because the visual change appears small - focus on whether an operational step is missing.
    
    Return JSON format: {{"frame_descriptions": [{{"segment_id": "1", "duration": "xxx - xxx", "description": "Best A* candidate: missing goal step + UI state change"}}]}}
    """
    return qwen_call(model, prompt, img_paths, is_json=True, step_info="A* Select")

# ================= 问答与自我评价 (适配 UI 任务) =================
def qa_and_reflect(model, goal, img_paths):
    """评估当前的可见帧是否足以让模型理解整个 UI 动作流"""
    # 1. 尝试推理步骤 (要求非常细致)
    prompt_qa = f"Task Goal: {goal}\nLook at these sequential UI frames. Describe the EXACT step-by-step UI actions (clicks, scrolls, typing) that happen transitioning from one frame to the next."
    answer_str = qwen_call(model, prompt_qa, img_paths, is_json=False, step_info="Action Analysis")
    
    # 2. 评估标准
    prompt_eval = f"""
    Task Goal: {goal}
    Your sequential UI analysis: {answer_str}
    
    As a strict UI automation tester, evaluate the VISUAL CONTINUITY of these frames. 
    Can a user replicate this task step-by-step based on these frames?
    
    Evaluate your confidence level strictly:
    1: Severe Jumps (There are completely missing screens or sudden state changes. E.g., jumping from home to settings without seeing the menu. MUST expand.)
    2: Minor Disconnects (The flow makes sense, but some button clicks, typing actions, or intermediate loading states are missing. Should expand.)
    3: Strong Continuity (The frames capture all important UI actions and transitions. The operation flow is clear and a user can follow without confusion. Very minor intermediate states may be missing but no key step is skipped.)
    
    CRITICAL RULE: UI tasks require high precision. If you have to GUESS what major action happened between any two frames, you MUST output 1 or 2. Do NOT output 3 unless the key action flow is clearly continuous with no important gaps.
    
    Output JSON exactly like this: {{"confidence": 1}}
    """
    conf_str = qwen_call(model, prompt_eval, img_paths, is_json=True, step_info="Confidence Eval")
    conf_json = parse_json(conf_str)
    confidence = conf_json.get("confidence", 1) if conf_json else 1
    
    return answer_str, int(confidence)

# ================= 核心工作流 =================
def select_process(model, goal, img_paths, sample_idx, num_frames, video_segments, select_fn, video_path=None, video_id=None, hist_cache=None, content_bounds=None, frozen_segments=None, max_frames=None):
    # frozen_segments: set of (start, end) tuples — segments that have been explored and found redundant
    # max_frames: 帧数上限，BFS 模式下用于限制单轮新增帧数
    if frozen_segments is None:
        frozen_segments = set()
    
    # 构建 segment 描述：标注每个 segment 两端帧对应的图片序号（1-indexed）
    # img_paths 和 sample_idx 是一一对应的，img_paths[i] 对应 sample_idx[i]
    frame_to_img_idx = {frame: i + 1 for i, frame in enumerate(sample_idx)}
    
    segment_des_lines = []
    for i, seg in enumerate(video_segments):
        seg_id = i + 1
        # 跳过已冻结的 segment，不展示给模型
        if (seg.start, seg.end) in frozen_segments:
            continue
        start_img = frame_to_img_idx.get(seg.start, "?")
        end_img = frame_to_img_idx.get(seg.end, "?")
        segment_des_lines.append(
            f"  Segment {seg_id}: frames {seg.start}-{seg.end} "
            f"(Image #{start_img} -> Image #{end_img})"
        )
    segment_des_str = "\n".join(segment_des_lines)
    
    # 如果所有 segment 都被冻结了，无法继续拆分
    if not segment_des_lines:
        print(f"  [select_process] All segments are frozen. Nothing to split.")
        return video_segments, sample_idx, False
    
    candidate_descriptions = select_fn(model, goal, img_paths, num_frames, segment_des_str)
    parsed_candidate = parse_json(candidate_descriptions) if candidate_descriptions else None
    
    # --- Determine which segment IDs to split ---
    selected_seg_ids = set()
    
    if parsed_candidate and "frame_descriptions" in parsed_candidate:
        selected_descriptions = parsed_candidate["frame_descriptions"]
        for desc in selected_descriptions:
            seg_id = None
            # Try to get segment_id from the description
            for key in desc:
                if key.lower() == "segment_id":
                    val = str(desc[key]).strip()
                    # Handle "start-end" format in segment_id field
                    if '-' in val:
                        try:
                            seg_id = int(val.split('-')[0])
                        except (ValueError, IndexError):
                            pass
                    if seg_id is None:
                        nums = re.findall(r'\d+', val)
                        if nums:
                            seg_id = int(nums[0])
                    break
            if seg_id is not None and 1 <= seg_id <= len(video_segments):
                selected_seg_ids.add(seg_id)
                print(f"  [select_process] Model selected segment {seg_id}: {video_segments[seg_id-1].start}-{video_segments[seg_id-1].end}")
            else:
                print(f"  [select_process] WARNING: Could not resolve segment_id from: {desc}")
    else:
        if not candidate_descriptions:
            print(f"  [select_process] WARNING: select_fn returned empty response!")
        elif parsed_candidate is None:
            print(f"  [select_process] WARNING: JSON parse failed! Raw: {str(candidate_descriptions)[:300]}")
        elif "frame_descriptions" not in parsed_candidate:
            print(f"  [select_process] WARNING: No 'frame_descriptions' key! Keys: {list(parsed_candidate.keys())}")
    
    # --- 过滤掉模型选中但已冻结的 segment ---
    unfrozen_selected = set()
    for seg_id in selected_seg_ids:
        seg = video_segments[seg_id - 1]
        if (seg.start, seg.end) in frozen_segments:
            print(f"  [select_process] Segment {seg_id} ({seg.start}-{seg.end}) is frozen, skipping.")
        else:
            unfrozen_selected.add(seg_id)
    selected_seg_ids = unfrozen_selected
    
    # --- FALLBACK: if model failed to select any segment, pick the longest splittable non-frozen segment ---
    if not selected_seg_ids:
        longest_seg_id = None
        longest_len = 0
        for i, seg in enumerate(video_segments):
            seg_len = seg.end - seg.start
            if seg_len > longest_len and seg_len > 1 and (seg.start, seg.end) not in frozen_segments:
                longest_len = seg_len
                longest_seg_id = i + 1
        if longest_seg_id is not None:
            selected_seg_ids.add(longest_seg_id)
            print(f"  [select_process] FALLBACK: auto-selecting longest segment {longest_seg_id} "
                  f"({video_segments[longest_seg_id-1].start}-{video_segments[longest_seg_id-1].end}, len={longest_len})")
        else:
            print(f"  [select_process] FALLBACK: no splittable segments left!")
            return video_segments, sample_idx, False
    
    # Build visual-change-based split point function if video_path is available
    split_point_fn = None
    if video_path is not None:
        split_point_fn = lambda s, e: find_visual_change_split_point(video_path, s, e, video_id)
    
    # --- BFS 配额限制：如果选了太多 segment，截断到不超过剩余配额 ---
    if max_frames is not None and len(selected_seg_ids) > 1:
        remaining_quota = max_frames - len(sample_idx)
        if remaining_quota <= 0:
            print(f"  [select_process] No remaining frame quota ({len(sample_idx)}/{max_frames}). Skipping.")
            return video_segments, sample_idx, False
        if len(selected_seg_ids) > remaining_quota:
            # 按 segment 长度降序排列，优先拆分最长的 segment
            sorted_seg_ids = sorted(selected_seg_ids, 
                                    key=lambda sid: video_segments[sid-1].end - video_segments[sid-1].start, 
                                    reverse=True)
            selected_seg_ids = set(sorted_seg_ids[:remaining_quota])
            print(f"  [select_process] BFS quota limit: truncated to {len(selected_seg_ids)} segments "
                  f"(remaining quota: {remaining_quota}, max_frames: {max_frames})")
    
    # --- Directly split by segment_id (bypass the fragile start/end matching) ---
    # 记录被拆分的原始 segment，用于冻结机制
    split_origin = {}  # {new_frame_idx: (original_seg_start, original_seg_end)}
    new_segments = []
    seg_counter = 0
    for i, seg in enumerate(video_segments):
        seg_id = i + 1
        if seg_id in selected_seg_ids:
            if seg.end - seg.start <= 1:
                seg_counter += 1
                new_segments.append(VideoSeg(seg.start, seg.end, seg_counter, None))
            else:
                if split_point_fn is not None:
                    try:
                        sp = split_point_fn(seg.start, seg.end)
                        if not (seg.start < sp < seg.end):
                            sp = (seg.start + seg.end) // 2
                    except Exception:
                        sp = (seg.start + seg.end) // 2
                else:
                    sp = (seg.start + seg.end) // 2
                split_origin[sp] = (seg.start, seg.end)
                seg_counter += 1
                new_segments.append(VideoSeg(seg.start, sp, seg_counter, None))
                seg_counter += 1
                new_segments.append(VideoSeg(sp, seg.end, seg_counter, None))
        else:
            seg_counter += 1
            new_segments.append(VideoSeg(seg.start, seg.end, seg_counter, None))
    video_segments = new_segments
    
    sample_idx_set = set()
    for seg in video_segments:
        sample_idx_set.add(seg.start)
        sample_idx_set.add(seg.end)
    new_sample_idx = sorted(list(sample_idx_set))
    
    print(f"  [select_process] After split: {len(sample_idx)} -> {len(new_sample_idx)} frames "
          f"(new frames: {sorted(set(new_sample_idx) - set(sample_idx))})")
    
    # Visual deduplication (新帧 vs 旧帧 + 新帧之间互相去重)
    if hist_cache is not None and video_path is not None:
        old_sample_set = set(sample_idx)
        new_frames = [idx for idx in new_sample_idx if idx not in old_sample_set]
        frames_to_remove = []
        accepted_new_frame_paths = []  # 已通过去重的新帧路径，用于新帧间互相比较
        
        for new_idx in new_frames:
            new_frame_path = os.path.join(CACHE_DIR, f"{video_id}_frame_{new_idx:04d}.jpg")
            if not os.path.exists(new_frame_path):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(new_frame_path, frame)
                cap.release()
            
            if not os.path.exists(new_frame_path):
                continue
            
            # 和所有旧帧比较
            old_frame_paths = [
                os.path.join(CACHE_DIR, f"{video_id}_frame_{idx:04d}.jpg") 
                for idx in old_sample_set
            ]
            old_frame_paths = [p for p in old_frame_paths if os.path.exists(p)]
            
            # 合并：旧帧 + 已通过去重的其他新帧（BFS 模式下新帧间互相去重）
            all_compare_paths = old_frame_paths + accepted_new_frame_paths
            
            if is_frame_redundant(new_frame_path, all_compare_paths, hist_cache, threshold=0.985):
                print(f"  [Visual Dedup] Frame {new_idx} is virtually identical to an existing/accepted frame, skipping.")
                frames_to_remove.append(new_idx)
                # 冻结机制：将被拆分的原始 segment 标记为 frozen
                if new_idx in split_origin:
                    orig_seg = split_origin[new_idx]
                    frozen_segments.add(orig_seg)
                    print(f"  [Freeze] Segment {orig_seg[0]}-{orig_seg[1]} frozen (new frame {new_idx} was redundant)")
            else:
                accepted_new_frame_paths.append(new_frame_path)
        
        if frames_to_remove:
            print(f"  [Visual Dedup] Removed {len(frames_to_remove)} redundant frames: {frames_to_remove}")
            new_sample_idx = [idx for idx in new_sample_idx if idx not in frames_to_remove]
            new_sample_idx = sorted(new_sample_idx)
            video_segments = [VideoSeg(new_sample_idx[i-1], new_sample_idx[i], i, None) 
                              for i in range(1, len(new_sample_idx))]
        else:
            print(f"  [Visual Dedup] No redundant frames found.")
    
    # Relevance filter: check new frames, replace irrelevant ones with nearby relevant frames
    if content_bounds is not None and video_path is not None:
        old_sample_set = set(sample_idx)
        new_frames = [idx for idx in new_sample_idx if idx not in old_sample_set]
        dropped_frames = []
        replacements = {}  # {old_idx: new_idx}
        for new_idx in new_frames:
            is_relevant, replacement = check_new_frame_relevance(
                model, goal, new_idx, video_path, video_id, sample_idx, content_bounds)
            if not is_relevant:
                dropped_frames.append(new_idx)
            elif replacement != new_idx:
                replacements[new_idx] = replacement
        
        if replacements or dropped_frames:
            # 替换无关帧
            updated = []
            for idx in new_sample_idx:
                if idx in dropped_frames:
                    continue  # 找不到替代的，丢弃
                elif idx in replacements:
                    updated.append(replacements[idx])
                else:
                    updated.append(idx)
            new_sample_idx = sorted(set(updated))  # 去重+排序
            video_segments = [VideoSeg(new_sample_idx[i-1], new_sample_idx[i], i, None)
                              for i in range(1, len(new_sample_idx))]
            if replacements:
                print(f"  [Relevance Filter] Replaced {len(replacements)} frames: {replacements}")
            if dropped_frames:
                print(f"  [Relevance Filter] Dropped {len(dropped_frames)} frames (no replacement): {dropped_frames}")
    
    print(f"  [select_process] Final: {len(new_sample_idx)} frames")
    actually_added = len(new_sample_idx) > len(sample_idx)
    return video_segments, new_sample_idx, actually_added

def run_one_video(video_id, goal, model, args):
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path): return None

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if num_frames <= 0 or fps <= 0: return None

    # ================= 初始化：全范围采样 + 逐帧相关性过滤（替代旧片头片尾检测） =================
    content_start = 0
    content_end = num_frames - 1
    content_bounds = [content_start, content_end]  # 可变列表，供过滤函数就地更新

    # 初始化根节点（使用全范围）
    TARGET_MIN_FRAMES = 8
    TARGET_MAX_FRAMES = 11
    
    effective_length = content_end - content_start + 1
    # 初始固定 init_interval 帧（将区间均匀分成 init_interval-1 段）
    INIT_FRAMES = args.init_interval  # init_interval 现在表示初始帧数（默认4）
    if effective_length <= INIT_FRAMES:
        sample_idx = list(range(content_start, content_end + 1))
    else:
        interval = max(1, effective_length // (INIT_FRAMES - 1))
        sample_idx = list(range(content_start, content_end + 1, interval))
        if sample_idx[-1] != content_end:
            sample_idx.append(content_end)
    
    print(f"[{video_id}] Video: {num_frames} frames, {num_frames/fps:.1f}s | Initial frames (before filter): {len(sample_idx)}")
    
    # 对初始帧从首尾逐帧检查相关性，去掉无关的片头片尾帧
    sample_idx = filter_irrelevant_frames(model, goal, sample_idx, video_path, video_id, content_bounds, fps, num_frames)
    content_start, content_end = content_bounds
    
    # 如果过滤后帧太少，在收缩后的区间内重新采样补充
    if len(sample_idx) < INIT_FRAMES and (content_end - content_start) > INIT_FRAMES:
        eff_len = content_end - content_start + 1
        new_interval = max(1, eff_len // (INIT_FRAMES - 1))
        sample_idx = list(range(content_start, content_end + 1, new_interval))
        if sample_idx[-1] != content_end:
            sample_idx.append(content_end)
        print(f"[{video_id}] Re-sampled after filter: {len(sample_idx)} frames in [{content_start}, {content_end}]")
    
    print(f"[{video_id}] Content range: [{content_start}, {content_end}] | Initial frames: {len(sample_idx)}")
        
    video_segments = [VideoSeg(sample_idx[i-1], sample_idx[i], i, None) for i in range(1, len(sample_idx))]
    
    # Histogram cache for visual deduplication (shared across steps)
    hist_cache = {}
    
    # 冻结机制：记录已被探索且新帧被去重的 segment，不再重复拆分
    frozen_segments = set()
    
    # 树搜索主循环
    # 退出条件：帧数达到上限 或 置信度足够高（兜底：总尝试次数防无限循环）
    max_total_attempts = TARGET_MAX_FRAMES + 10  # 安全上限，防止去重/过滤导致无限循环
    effective_step = 0  # 有效步数（只有真正新增帧的轮次才计数）
    last_confidence = 0  # 追踪最后一次置信度评估结果
    for attempt in range(1, max_total_attempts + 1):
        current_frames = len(sample_idx)
        print(f"\n>>>>> [Video: {video_id}] Attempt {attempt} (effective_step={effective_step}) | Current Frames: {current_frames} | Target: {TARGET_MIN_FRAMES}-{TARGET_MAX_FRAMES} <<<<<")
        
        # 已达到目标帧数上限，停止扩展
        if current_frames >= TARGET_MAX_FRAMES:
            print(f"[{video_id}] Reached target max frames ({TARGET_MAX_FRAMES}). Stopping expansion.")
            break
        
        img_paths = extract_and_cache_frames(video_path, sample_idx, video_id)
        
        # Confidence check: only allow early stop after reaching minimum frame count AND min effective steps
        if current_frames >= TARGET_MIN_FRAMES and effective_step > args.min_steps:
            answer, confidence = qa_and_reflect(model, goal, img_paths)
            last_confidence = confidence
            print(f"[{video_id}] Effective step {effective_step} Eval Result -> Confidence: {confidence}")
            
            if confidence >= args.conf_lower:
                print(f"[{video_id}] Confidence met threshold after {effective_step} effective steps with {current_frames} frames. Stopping search.")
                break
        else:
            if current_frames < TARGET_MIN_FRAMES:
                print(f"[{video_id}] Effective step {effective_step} - Forced expansion (only {current_frames} frames, need >= {TARGET_MIN_FRAMES})")
            else:
                print(f"[{video_id}] Effective step {effective_step} - Forced expansion (min_steps={args.min_steps}, skipping confidence check)")
            
        # 节点扩展: 始终使用配置的搜索策略
        if args.search_strategy == "bfs":
            select_fn = bfs_select_segments
        elif args.search_strategy == "gbfs":
            select_fn = gbfs_select_one_segment
        elif args.search_strategy == "dijkstra":
            select_fn = dijkstra_select_one_segment
        else:
            select_fn = a_star_select_one_segment
            
        video_segments, sample_idx, actually_added = select_process(
            model, goal, img_paths, sample_idx, num_frames, video_segments, select_fn,
            video_path=video_path, video_id=video_id, hist_cache=hist_cache, content_bounds=content_bounds,
            frozen_segments=frozen_segments, max_frames=TARGET_MAX_FRAMES
        )
        
        if actually_added:
            effective_step += 1
        else:
            print(f"[{video_id}] Attempt {attempt} did not add new frames (dedup), not counting as effective step.")
    
    # 安全检查：如果迭代结束后帧数仍不够，且置信度也不够高，才强制补充
    if len(sample_idx) < TARGET_MIN_FRAMES and last_confidence < args.conf_lower:
        print(f"[{video_id}] WARNING: Only {len(sample_idx)} frames after all steps (confidence={last_confidence}). Force-splitting largest gaps to reach {TARGET_MIN_FRAMES}...")
        max_force_attempts = TARGET_MIN_FRAMES + 5  # 防止无限循环
        force_attempt = 0
        while len(sample_idx) < TARGET_MIN_FRAMES and force_attempt < max_force_attempts:
            force_attempt += 1
            # 找最大间隔（跳过 frozen 的 segment）
            max_gap = 0
            max_gap_idx = 0
            for i in range(len(sample_idx) - 1):
                seg_key = (sample_idx[i], sample_idx[i+1])
                if seg_key in frozen_segments:
                    continue
                gap = sample_idx[i+1] - sample_idx[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            if max_gap <= 1:
                break  # 无法再分割
            # 用视觉变化分割点代替简单中点
            seg_start = sample_idx[max_gap_idx]
            seg_end = sample_idx[max_gap_idx + 1]
            sp = find_visual_change_split_point(video_path, seg_start, seg_end, video_id)
            
            # 强制补帧也要去重
            sp_frame_path = os.path.join(CACHE_DIR, f"{video_id}_frame_{sp:04d}.jpg")
            if not os.path.exists(sp_frame_path):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sp)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(sp_frame_path, frame)
                cap.release()
            
            existing_frame_paths = [
                os.path.join(CACHE_DIR, f"{video_id}_frame_{idx:04d}.jpg")
                for idx in sample_idx
            ]
            existing_frame_paths = [p for p in existing_frame_paths if os.path.exists(p)]
            
            if os.path.exists(sp_frame_path) and is_frame_redundant(sp_frame_path, existing_frame_paths, hist_cache, threshold=0.985):
                print(f"  [Force-Fill Dedup] Frame {sp} is redundant, freezing segment {seg_start}-{seg_end}")
                frozen_segments.add((seg_start, seg_end))
                continue  # 不插入，尝试下一个最大间隔
            
            sample_idx.insert(max_gap_idx + 1, sp)
            print(f"  Force-added frame {sp} (gap was {max_gap}, using visual change split point)")
        video_segments = [VideoSeg(sample_idx[i-1], sample_idx[i], i, None) for i in range(1, len(sample_idx))]
    elif len(sample_idx) < TARGET_MIN_FRAMES:
        print(f"[{video_id}] Only {len(sample_idx)} frames but confidence={last_confidence} is sufficient. Skipping forced fill.")

    # 保存最终结果图片
    video_out_dir = os.path.join(OUT_ROOT, video_id)
    os.makedirs(video_out_dir, exist_ok=True)
    
    final_img_paths = extract_and_cache_frames(video_path, sample_idx, video_id)
    for i, path in enumerate(final_img_paths):
        target_path = os.path.join(video_out_dir, f"frame_akeys_{i:04d}.png")
        img = cv2.imread(path)
        if img is not None:
            cv2.imwrite(target_path, img)
            
    print(f"[{video_id}] Finished. Saved {len(sample_idx)} keyframes.\n")
    
    # 【新增功能 3】：计算并记录最终选出帧的时间戳和百分比
    frame_records = []
    for idx in sample_idx:
        percent = (idx / max(1, num_frames - 1)) * 100
        total_seconds = idx / fps
        mm = int(total_seconds // 60)
        ss = int(total_seconds % 60)
        ff = int(idx % max(1, int(fps))) # 当前秒内的帧号
        
        frame_records.append({
            "frame_idx": idx,
            "time": f"{mm:02d}:{ss:02d}:{ff:02d}",
            "percent": f"{percent:.2f}%"
        })
        
    return frame_records

def main():
    global VIDEO_DIR, JSON_DIR, OUT_ROOT, CACHE_DIR, RECORD_JSON_PATH
    args = parse_args()
    
    # 从命令行参数初始化路径
    VIDEO_DIR = args.video_dir
    JSON_DIR = args.json_dir
    OUT_ROOT = args.out_root
    CACHE_DIR = args.cache_dir
    RECORD_JSON_PATH = args.record_json_path
    
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    name_list = ['all'] if args.qwen_name_list.strip().lower() == 'all' else [x.strip() for x in args.qwen_name_list.split(',')]
    model = get_api(name_list, model_type=args.model_type, EXTRA_PARAMS={"temperature": args.temperature})
    
    with open(JSON_DIR, 'r') as f:
        data = json.load(f)
        
    video_tasks = {}
    for task_list in data.get("ours", []):
        for step in task_list:
            ep_id = step["ep_id"]
            if ep_id not in video_tasks:
                video_tasks[ep_id] = step["goal"]

    # Convert to sorted list for deterministic sharding
    all_video_items = sorted(video_tasks.items(), key=lambda x: x[0])
    total = len(all_video_items)
    
    # Shard splitting: --shard=1 processes [0%, 10%), --shard=10 processes [90%, 100%]
    if args.shard > 0:
        shard_size = total / 10  # float division for precise boundaries
        start_idx = int(round(shard_size * (args.shard - 1)))
        end_idx = int(round(shard_size * args.shard))
        # Clamp to valid range
        start_idx = max(0, min(start_idx, total))
        end_idx = max(0, min(end_idx, total))
        all_video_items = all_video_items[start_idx:end_idx]
        print(f"[Shard {args.shard}/10] Processing videos {start_idx}-{end_idx-1} (total {len(all_video_items)}/{total})")
    
    print(f"Starting AKeyS extraction for {len(all_video_items)} videos using {args.search_strategy.upper()} strategy...")
    
    all_video_records = {}
    records_lock = threading.Lock()
    skipped = 0
    
    # Filter out already-processed videos first
    pending_items = []
    for video_id, goal in all_video_items:
        video_out_dir = os.path.join(OUT_ROOT, video_id)
        if os.path.isdir(video_out_dir) and os.listdir(video_out_dir):
            print(f"[{video_id}] Output dir already non-empty, skipping.")
            skipped += 1
            continue
        pending_items.append((video_id, goal))
    
    print(f"Skipped {skipped} already-processed videos. {len(pending_items)} videos to process with {args.max_workers} threads.")
    
    def process_one(item):
        video_id, goal = item
        try:
            records = run_one_video(video_id, goal, model, args)
            if records is not None:
                with records_lock:
                    all_video_records[video_id] = records
            return video_id, True
        except Exception as e:
            print(f"\n[{video_id}] ERROR: {e}\n")
            return video_id, False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_one, item): item[0] for item in pending_items}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing videos"):
            video_id = futures[future]
            try:
                vid, success = future.result()
            except Exception as e:
                print(f"\n[{video_id}] FATAL: {e}\n")

    # Save records to shard-specific JSON file to avoid write conflicts
    if args.shard > 0:
        record_path = RECORD_JSON_PATH.replace(".json", f"_shard{args.shard}.json")
    else:
        record_path = RECORD_JSON_PATH
    
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(all_video_records, f, indent=4, ensure_ascii=False)
    print(f"\nDone. Processed {len(all_video_records)} videos, skipped {skipped}. Records saved to {record_path}")

if __name__ == "__main__":
    main()