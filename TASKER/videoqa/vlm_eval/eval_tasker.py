"""
方法3: TASKER 取帧 + VLM 理解 (ours)

核心设计：Coverage-Aware A* (覆盖度感知的自适应选帧)

核心问题:
  A* 容易反复选同一个区域（视觉变化大的区域），导致帧在局部聚集，
  而视频其他区域完全缺失关键信息。

解决方案:
  1. 覆盖度感知选帧: segment 可选性由「问题相关性 × 覆盖度需求」决定
     - 已密集采样的区域，其相邻 segments 优先级显著降低
     - 大 gap（稀疏区域）自动获得更高优先级
  2. 强制覆盖保证: 每步选帧后检查分布，若某段视频的覆盖密度 < 阈值，
     则下一步强制选该区域（不让 VLM 决定）
  3. 帧数硬上限 16: 与 Uniform baseline 完全对齐
  4. 每步 mid-reasoning: 不惜 API 成本，每步都分析缺什么
  5. A* 单选: 保持一次一帧的精细搜索

流程:
  1. 初始化: 均匀采样 4 帧（保证首尾覆盖）
  2. 主循环 (最多 12 步, 4→16 帧):
     a. 计算当前覆盖度分布（每个 segment 的密度）
     b. 若有严重稀疏区域 → 强制选该区域（跳过 VLM 决策）
     c. 否则 → mid-reasoning 分析 + 覆盖度加权的 A* 选帧
     d. 视觉变化分割 + 适度去重
  3. Final QA: 16 帧输入做最终问答
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
    TASKER_INIT_INTERVAL, TASKER_FINAL_STEP, TASKER_SEARCH_STRATEGY,
    TASKER_MAX_FRAMES, MAX_CONCURRENT_REQUESTS,
    RESULTS_BASE_DIR, SAMPLE_FPS
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
#  超参数
# ============================================================
TARGET_MAX_FRAMES = 16       # 最终帧数上限（与 Uniform 对齐）
INIT_FRAMES = 4              # 初始均匀采样帧数
DEDUP_THRESHOLD = 0.985      # 视觉去重阈值
COVERAGE_IMBALANCE_RATIO = 3.0  # 覆盖度失衡倍率：最大gap > 平均gap×此值 → 强制覆盖


# ============================================================
#  视觉去重工具
# ============================================================

def compute_color_histogram(img_path):
    """计算图片的颜色直方图特征（3通道联合直方图）"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    hist = cv2.calcHist([img], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def frame_similarity(hist1, hist2):
    """计算两个颜色直方图之间的相关系数相似度"""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def is_frame_redundant(new_frame_path, existing_frame_paths, hist_cache, threshold=DEDUP_THRESHOLD):
    """检查新帧是否与已选帧过于相似"""
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


# ============================================================
#  视觉变化分割点
# ============================================================

def _frame_hist(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _hist_similarity(hist_a, hist_b):
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def _laplacian_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def find_visual_change_split_point(video_path, segment_start, segment_end):
    """
    在 segment 内找视觉变化最大的分割点:
    1. 粗采样找最大变化区间
    2. 密集采样找过渡后稳定帧
    3. 模糊度检测
    Fallback: 中点
    """
    midpoint = (segment_start + segment_end) // 2
    try:
        seg_length = segment_end - segment_start
        if seg_length <= 2:
            return midpoint
        
        cap = cv2.VideoCapture(video_path)
        
        # 粗采样
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
        
        # 密集采样找稳定帧
        transition_len = best_b - best_a
        if transition_len <= 2:
            candidate = best_b
        else:
            dense_step = max(1, transition_len // 20)
            dense_indices = list(range(best_a, best_b + 1, dense_step))
            if dense_indices[-1] != best_b:
                dense_indices.append(best_b)
            
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
                        break
            candidate = stable_idx
        
        # 模糊度检测
        if candidate not in frames:
            f = _read_frame(cap, candidate)
            if f is not None:
                frames[candidate] = f
        
        if candidate in frames:
            blur_score = _laplacian_variance(frames[candidate])
            if blur_score < 100:
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
                candidate = best_clear_idx
        
        cap.release()
        
        # Fallback：分割点太偏则回退中点
        min_pos = segment_start + int(seg_length * 0.15)
        max_pos = segment_start + int(seg_length * 0.85)
        if candidate < min_pos or candidate > max_pos:
            return midpoint
        
        return candidate
    except Exception:
        return midpoint


# ============================================================
#  VideoSegment 数据结构
# ============================================================

class VideoSegment:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    def __eq__(self, other):
        if isinstance(other, VideoSegment):
            return self.start == other.start and self.end == other.end
        return False
    
    def __hash__(self):
        return hash((self.start, self.end))
    
    def __repr__(self):
        return f"Seg({self.start}-{self.end})"


# ============================================================
#  帧提取与缓存
# ============================================================

def extract_and_cache_frames(video_path, sample_idx, cache_dir):
    """懒加载：动态抽取所需帧并存入缓存目录"""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    img_paths = []
    cap = cv2.VideoCapture(video_path)
    for idx in sample_idx:
        frame_path = os.path.join(cache_dir, f"{video_id}_frame_{idx:06d}.jpg")
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
#  覆盖度分析
# ============================================================

def analyze_coverage(sample_idx, num_frames_total):
    """
    分析当前帧的时间覆盖度分布。
    
    Returns:
        {
            "gaps": [(start, end, gap_size), ...],  # 所有相邻帧间隙
            "max_gap": (start, end, gap_size),      # 最大间隙
            "avg_gap": float,                        # 平均间隙
            "imbalanced": bool,                      # 是否存在严重失衡
            "sparse_segments": [(start, end), ...],  # 过于稀疏的 segments
        }
    """
    if len(sample_idx) < 2:
        return {
            "gaps": [], "max_gap": (0, num_frames_total, num_frames_total),
            "avg_gap": num_frames_total, "imbalanced": True,
            "sparse_segments": [(0, num_frames_total)],
        }
    
    gaps = []
    for i in range(len(sample_idx) - 1):
        gap = sample_idx[i + 1] - sample_idx[i]
        gaps.append((sample_idx[i], sample_idx[i + 1], gap))
    
    gap_sizes = [g[2] for g in gaps]
    avg_gap = np.mean(gap_sizes)
    max_gap_info = max(gaps, key=lambda g: g[2])
    
    # 判断是否失衡：最大 gap > 平均 gap × COVERAGE_IMBALANCE_RATIO
    imbalanced = max_gap_info[2] > avg_gap * COVERAGE_IMBALANCE_RATIO
    
    # 找出所有"过于稀疏"的区域（> 平均 gap × 2）
    sparse_threshold = avg_gap * 2.0
    sparse_segments = [(g[0], g[1]) for g in gaps if g[2] > sparse_threshold]
    
    return {
        "gaps": gaps,
        "max_gap": max_gap_info,
        "avg_gap": avg_gap,
        "imbalanced": imbalanced,
        "sparse_segments": sparse_segments,
    }


# ============================================================
#  安全 JSON 解析
# ============================================================

def parse_json_safe(response):
    if not response:
        return None
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(json_str)
    except:
        return None


# ============================================================
#  Mid-Reasoning 分析
# ============================================================

def mid_reasoning_analysis(img_paths, question, options_text, num_frames):
    """让 VLM 分析当前帧还缺什么信息"""
    prompt = f"""You are shown {len(img_paths)} frames sampled from a video with approximately {num_frames} total frames.

Question: {question}
Options:
{options_text}

Analyze what's MISSING from these frames that prevents you from answering confidently:
1. What key actions, events, or state changes are NOT visible?
2. Between which adjacent frames (by their image number) do you suspect important events are hidden?
3. Is the beginning or end of the relevant activity captured?

Be specific and concise (2-3 sentences). Focus on what's MISSING."""

    response = call_qwen_vl(
        question=prompt,
        image_paths=img_paths,
        system_prompt="You are a video analysis assistant. Identify missing visual information.",
        temperature=0.0,
        max_tokens=200,
    )
    return response or ""


# ============================================================
#  Coverage-Aware A* Segment Selection
#  
#  核心创新：在 prompt 中告诉 VLM 当前覆盖度分布，
#  并明确指出哪些区域已经足够密集（不需要再选），
#  哪些区域是稀疏的（优先选择）。
# ============================================================

def coverage_aware_select_segment(
    question, options_text, img_paths, num_frames,
    segment_des, missing_info_hint, coverage_info
):
    """
    覆盖度感知的 A* segment 选择。
    
    在标准 A* 基础上，明确告诉 VLM:
    - 哪些区域已经密集（避免选择）
    - 哪些区域稀疏（优先考虑）
    - 缺失信息分析结果
    """
    hint_section = ""
    if missing_info_hint:
        hint_section = f"""
Analysis of missing information:
"{missing_info_hint}"
"""
    
    coverage_section = ""
    if coverage_info:
        coverage_section = f"""
IMPORTANT - Current frame coverage status:
{coverage_info}
You MUST factor in coverage balance: prefer segments in SPARSE regions over those in already DENSE regions,
even if the dense region seems more "interesting". A well-distributed frame set answers questions better.
"""
    
    prompt = f"""You are provided with sequential frames sampled from a video.
Each image is labeled with its frame index. The images are shown in chronological order.

Question: {question}
Options:
{options_text}

Candidate segments (gaps between current frames):
{segment_des}
{hint_section}{coverage_section}
Select the ONE segment that is MOST VALUABLE to explore next, considering:
1. QUESTION RELEVANCE: Does this gap likely hide crucial visual info for the question?
2. COVERAGE NEED: Is this segment in a sparsely-covered region? (strongly prefer sparse regions)
3. STATE CHANGE: Do the boundary frames of this segment look very different? (indicates hidden events)

Return JSON: {{"frame_descriptions": [{{"segment_id": "X", "duration": "start - end", "description": "reason"}}]}}"""

    response = call_qwen_vl(
        question=prompt,
        image_paths=img_paths,
        system_prompt="You are an expert video analysis planner. Output JSON only.",
        temperature=0.0,
        max_tokens=256,
    )
    return response


# ============================================================
#  Segment 选择 + 分割 + 去重 (无冻结, 覆盖度感知)
# ============================================================

def select_and_split_segment(
    question, options_text, img_paths, sample_idx, num_frames_total,
    video_segments, video_path, cache_dir, hist_cache,
    missing_info_hint="", forced_segment_idx=None
):
    """
    选择一个 segment 并分割。
    
    Args:
        forced_segment_idx: 如果不为 None，跳过 VLM 决策，强制选此 segment
                           （用于覆盖度强制补充）
    
    Returns:
        (video_segments, sample_idx, actually_added)
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 构建 segment 描述
    frame_to_img_idx = {frame: i + 1 for i, frame in enumerate(sample_idx)}
    
    segment_des_lines = []
    splittable_seg_ids = []
    for i, seg in enumerate(video_segments):
        seg_id = i + 1
        if seg.end - seg.start <= 1:
            continue
        start_img = frame_to_img_idx.get(seg.start, "?")
        end_img = frame_to_img_idx.get(seg.end, "?")
        segment_des_lines.append(
            f"  Segment {seg_id}: frames {seg.start}-{seg.end} "
            f"(Image #{start_img} -> Image #{end_img}, gap={seg.end - seg.start} frames)"
        )
        splittable_seg_ids.append(seg_id)
    
    if not splittable_seg_ids:
        return video_segments, sample_idx, False
    
    # 决定选哪个 segment
    selected_seg_id = None
    
    if forced_segment_idx is not None:
        # 强制选择模式（覆盖度补充）
        selected_seg_id = forced_segment_idx
    else:
        # VLM 决策模式
        segment_des_str = "\n".join(segment_des_lines)
        
        # 构建覆盖度信息文本
        coverage = analyze_coverage(sample_idx, num_frames_total)
        coverage_info = ""
        if coverage["gaps"]:
            gap_sizes = [g[2] for g in coverage["gaps"]]
            avg_gap = coverage["avg_gap"]
            
            dense_regions = []
            sparse_regions = []
            for g in coverage["gaps"]:
                seg_start, seg_end, gap_size = g
                if gap_size < avg_gap * 0.5:
                    dense_regions.append(f"frames {seg_start}-{seg_end} (gap={gap_size}, ALREADY DENSE - low priority)")
                elif gap_size > avg_gap * 1.5:
                    sparse_regions.append(f"frames {seg_start}-{seg_end} (gap={gap_size}, SPARSE - high priority)")
            
            parts = []
            if sparse_regions:
                parts.append("SPARSE regions (HIGH priority, prefer these):\n  " + "\n  ".join(sparse_regions))
            if dense_regions:
                parts.append("DENSE regions (LOW priority, avoid unless critical):\n  " + "\n  ".join(dense_regions))
            coverage_info = "\n".join(parts)
        
        response = coverage_aware_select_segment(
            question, options_text, img_paths, num_frames_total,
            segment_des_str, missing_info_hint, coverage_info
        )
        
        parsed = parse_json_safe(response)
        if parsed and "frame_descriptions" in parsed:
            for desc in parsed["frame_descriptions"]:
                for key in desc:
                    if key.lower() == "segment_id":
                        val = str(desc[key]).strip()
                        nums = re.findall(r'\d+', val)
                        if nums:
                            candidate_id = int(nums[0])
                            if 1 <= candidate_id <= len(video_segments):
                                seg = video_segments[candidate_id - 1]
                                if seg.end - seg.start > 1:
                                    selected_seg_id = candidate_id
                        break
                if selected_seg_id:
                    break
    
    # Fallback: 选最长的可拆分 segment
    if selected_seg_id is None:
        longest_seg_id = None
        longest_len = 0
        for i, seg in enumerate(video_segments):
            seg_len = seg.end - seg.start
            if seg_len > longest_len and seg_len > 1:
                longest_len = seg_len
                longest_seg_id = i + 1
        if longest_seg_id is not None:
            selected_seg_id = longest_seg_id
        else:
            return video_segments, sample_idx, False
    
    # 分割选中的 segment
    target_seg = video_segments[selected_seg_id - 1]
    if target_seg.end - target_seg.start <= 1:
        return video_segments, sample_idx, False
    
    sp = find_visual_change_split_point(video_path, target_seg.start, target_seg.end)
    if not (target_seg.start < sp < target_seg.end):
        sp = (target_seg.start + target_seg.end) // 2
    
    # 提取新帧并检查去重
    new_frame_path = os.path.join(cache_dir, f"{video_id}_frame_{sp:06d}.jpg")
    if not os.path.exists(new_frame_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sp)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(new_frame_path, frame)
        cap.release()
    
    if not os.path.exists(new_frame_path):
        return video_segments, sample_idx, False
    
    # 视觉去重
    existing_frame_paths = [
        os.path.join(cache_dir, f"{video_id}_frame_{idx:06d}.jpg")
        for idx in sample_idx
    ]
    existing_frame_paths = [p for p in existing_frame_paths if os.path.exists(p)]
    
    if is_frame_redundant(new_frame_path, existing_frame_paths, hist_cache, threshold=DEDUP_THRESHOLD):
        # 去重失败：尝试用中点代替
        alt_sp = (target_seg.start + target_seg.end) // 2
        if alt_sp != sp and alt_sp not in sample_idx:
            alt_frame_path = os.path.join(cache_dir, f"{video_id}_frame_{alt_sp:06d}.jpg")
            if not os.path.exists(alt_frame_path):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, alt_sp)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(alt_frame_path, frame)
                cap.release()
            
            if os.path.exists(alt_frame_path) and not is_frame_redundant(
                alt_frame_path, existing_frame_paths, hist_cache, threshold=DEDUP_THRESHOLD
            ):
                sp = alt_sp
            else:
                # 两个都重复，该 segment 确实没有新信息
                # 但不冻结，缩小 segment 等待下次
                new_segments = []
                for i, seg in enumerate(video_segments):
                    if i + 1 == selected_seg_id:
                        # 仍然分割（记录分割点），但不加帧
                        new_segments.append(VideoSegment(seg.start, sp))
                        new_segments.append(VideoSegment(sp, seg.end))
                    else:
                        new_segments.append(seg)
                return new_segments, sample_idx, False
        else:
            # 无法找到非重复替代
            new_segments = []
            for i, seg in enumerate(video_segments):
                if i + 1 == selected_seg_id:
                    new_segments.append(VideoSegment(seg.start, sp))
                    new_segments.append(VideoSegment(sp, seg.end))
                else:
                    new_segments.append(seg)
            return new_segments, sample_idx, False
    
    # 成功添加新帧
    new_sample_idx = sorted(list(set(sample_idx + [sp])))
    
    # 重建 segments
    new_segments = [
        VideoSegment(new_sample_idx[i], new_sample_idx[i + 1])
        for i in range(len(new_sample_idx) - 1)
    ]
    
    return new_segments, new_sample_idx, True


# ============================================================
#  核心流程: TASKER-v4 选帧 (Coverage-Aware)
# ============================================================

def tasker_select_frames(
    video_path: str,
    question: str,
    options: list,
    cache_dir: str,
    search_strategy: str = "a_star",
    max_frames: int = TARGET_MAX_FRAMES,
    init_frames: int = INIT_FRAMES,
    conf_lower: int = 3,
    min_steps: int = 2,
) -> dict:
    """
    TASKER-v4 覆盖度感知选帧。
    
    核心改进: 解决 A* 在局部区域反复聚集的问题。
    
    流程:
    1. 初始化: 均匀 4 帧（首尾+中间均匀）
    2. 主循环（每步 +1 帧，最多到 16 帧）:
       a. 覆盖度审计: 检查帧分布是否严重失衡
       b. 如果失衡 → 强制选最大 gap 区域（不问 VLM）
       c. 如果平衡 → mid-reasoning + coverage-aware A* 选帧
       d. 视觉变化分割 + 去重
    3. 返回 16 帧
    """
    os.makedirs(cache_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if num_frames_total <= 0 or fps <= 0:
        return {
            "selected_frame_paths": [],
            "selected_frame_indices": [],
            "num_frames": 0,
            "num_steps": 0,
            "stop_reason": "video_error",
        }
    
    # 构建 options_text
    option_labels = ['A', 'B', 'C', 'D', 'E']
    options_text = "\n".join([f"({l}) {o}" for l, o in zip(option_labels[:len(options)], options)])
    
    # === 初始化：均匀 init_frames 帧 ===
    if num_frames_total <= max_frames:
        # 视频帧数很少，直接用全部帧
        sample_idx = list(range(num_frames_total))
        final_paths = extract_and_cache_frames(video_path, sample_idx, cache_dir)
        return {
            "selected_frame_paths": final_paths,
            "selected_frame_indices": sample_idx,
            "num_frames": len(final_paths),
            "num_steps": 0,
            "stop_reason": "short_video",
        }
    
    # 均匀采样 init_frames 帧（含首尾）
    sample_idx = np.linspace(0, num_frames_total - 1, init_frames, dtype=int).tolist()
    sample_idx = sorted(list(set(sample_idx)))
    
    # 构建 segments
    video_segments = [
        VideoSegment(sample_idx[i], sample_idx[i + 1])
        for i in range(len(sample_idx) - 1)
    ]
    
    hist_cache = {}
    
    # === 主搜索循环 ===
    max_attempts = (max_frames - len(sample_idx)) + 10  # 足够的尝试次数
    effective_step = 0
    stall_counter = 0
    stop_reason = "max_frames_reached"
    forced_coverage_count = 0  # 强制覆盖次数统计
    
    for attempt in range(1, max_attempts + 1):
        # 已达上限
        if len(sample_idx) >= max_frames:
            stop_reason = "max_frames_reached"
            break
        
        # 连续 5 次没加新帧
        if stall_counter >= 5:
            stop_reason = "stalled"
            break
        
        # 没有可分割的 segment
        splittable = [seg for seg in video_segments if seg.end - seg.start > 1]
        if not splittable:
            stop_reason = "no_splittable_segments"
            break
        
        # === 覆盖度审计 ===
        coverage = analyze_coverage(sample_idx, num_frames_total)
        forced_segment_idx = None
        
        if coverage["imbalanced"] and len(sample_idx) >= init_frames + 2:
            # 严重失衡：强制选最大 gap 所在的 segment
            max_gap_start, max_gap_end, _ = coverage["max_gap"]
            # 找到这个 gap 对应的 segment id
            for i, seg in enumerate(video_segments):
                if seg.start == max_gap_start and seg.end == max_gap_end:
                    if seg.end - seg.start > 1:
                        forced_segment_idx = i + 1
                        forced_coverage_count += 1
                    break
        
        # === 如果不是强制，做 mid-reasoning ===
        missing_info_hint = ""
        if forced_segment_idx is None:
            img_paths = extract_and_cache_frames(video_path, sample_idx, cache_dir)
            if not img_paths:
                stop_reason = "frame_extract_error"
                break
            
            # 限制传给 VLM 的帧数
            if len(img_paths) > 16:
                step_size = len(img_paths) / 16
                display_indices = [int(i * step_size) for i in range(16)]
                display_paths = [img_paths[i] for i in display_indices]
            else:
                display_paths = img_paths
            
            # mid-reasoning（每步都做）
            missing_info_hint = mid_reasoning_analysis(
                display_paths, question, options_text, num_frames_total
            )
        
        # === 选帧 + 分割 ===
        img_paths_for_select = None
        if forced_segment_idx is None:
            img_paths_for_select = display_paths
        else:
            # 强制模式下也需要 img_paths（用于 fallback，但不传给 VLM）
            img_paths_for_select = extract_and_cache_frames(video_path, sample_idx, cache_dir)
            if img_paths_for_select and len(img_paths_for_select) > 16:
                step_size = len(img_paths_for_select) / 16
                display_indices = [int(i * step_size) for i in range(16)]
                img_paths_for_select = [img_paths_for_select[i] for i in display_indices]
        
        video_segments, sample_idx, actually_added = select_and_split_segment(
            question, options_text, img_paths_for_select, sample_idx, num_frames_total,
            video_segments, video_path, cache_dir, hist_cache,
            missing_info_hint=missing_info_hint,
            forced_segment_idx=forced_segment_idx
        )
        
        if actually_added:
            effective_step += 1
            stall_counter = 0
        else:
            stall_counter += 1
    
    # === Force-fill: 帧数不够时补最大 gap ===
    while len(sample_idx) < max_frames:
        if len(sample_idx) < 2:
            break
        max_gap = 0
        max_gap_idx = 0
        for i in range(len(sample_idx) - 1):
            gap = sample_idx[i + 1] - sample_idx[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        if max_gap <= 1:
            break
        
        seg_start = sample_idx[max_gap_idx]
        seg_end = sample_idx[max_gap_idx + 1]
        sp = find_visual_change_split_point(video_path, seg_start, seg_end)
        
        if sp in sample_idx:
            sp = (seg_start + seg_end) // 2
        if sp in sample_idx:
            break
        
        # 去重检查
        sp_frame_path = os.path.join(cache_dir, f"{video_id}_frame_{sp:06d}.jpg")
        if not os.path.exists(sp_frame_path):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sp)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(sp_frame_path, frame)
            cap.release()
        
        existing_paths = [
            os.path.join(cache_dir, f"{video_id}_frame_{idx:06d}.jpg")
            for idx in sample_idx
        ]
        existing_paths = [p for p in existing_paths if os.path.exists(p)]
        
        if os.path.exists(sp_frame_path) and is_frame_redundant(
            sp_frame_path, existing_paths, hist_cache, threshold=DEDUP_THRESHOLD
        ):
            # 用中点
            sp = (seg_start + seg_end) // 2
            if sp in sample_idx:
                break
        
        sample_idx = sorted(list(set(sample_idx + [sp])))
    
    # 截断到 max_frames
    if len(sample_idx) > max_frames:
        # 均匀采样
        indices = np.linspace(0, len(sample_idx) - 1, max_frames, dtype=int).tolist()
        sample_idx = [sample_idx[i] for i in indices]
    
    # 获取最终帧
    final_frame_paths = extract_and_cache_frames(video_path, sample_idx, cache_dir)
    
    return {
        "selected_frame_paths": final_frame_paths,
        "selected_frame_indices": sample_idx,
        "num_frames": len(final_frame_paths),
        "num_steps": effective_step,
        "stop_reason": stop_reason,
        "forced_coverage_count": forced_coverage_count,
    }


# ============================================================
#  评测入口
# ============================================================

def run_single_item(item, cache_dir, search_strategy, max_frames, init_frames, conf_lower, min_steps):
    """处理单个样本（选帧 + 独立 QA 解耦）"""
    video_path = item["video_path"]
    question = item["question"]
    options = item["options"]
    uid = item.get("uid") or item.get("quid")
    
    try:
        # === Stage 1: 选帧 ===
        selection = tasker_select_frames(
            video_path, question, options, cache_dir,
            search_strategy=search_strategy,
            max_frames=max_frames,
            init_frames=init_frames,
            conf_lower=conf_lower,
            min_steps=min_steps,
        )
        
        selected_frames = selection["selected_frame_paths"]
        
        if not selected_frames:
            return {
                "uid": uid, "pred": -1, "num_frames": 0,
                "num_steps": selection["num_steps"],
                "stop_reason": selection["stop_reason"],
                "error": "no_frames_selected",
            }
        
        # === Stage 2: 统一 Final QA（与 Uniform/VideoTree 一致） ===
        prompt = build_vqa_prompt(question, options, len(selected_frames))
        
        response = call_qwen_vl(
            question=prompt,
            image_paths=selected_frames,
            system_prompt=(
                "You are an expert video understanding assistant. "
                "You are shown key frames intelligently selected from a video. "
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
            "stop_reason": selection["stop_reason"],
            "selected_indices": selection["selected_frame_indices"],
            "forced_coverage": selection.get("forced_coverage_count", 0),
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
    print(f"方法: TASKER-v4 Coverage-Aware (max_frames={args.max_frames})")
    print(f"      覆盖度感知 A* 选帧 → 统一 Final QA")
    print(f"数据集: EgoSchema {split_name}")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_egoschema_dataset(subset_only=subset_only)
    print(f"  加载了 {len(dataset)} 个样本")
    
    # 使用 v4 checkpoint 目录
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "tasker_v4_checkpoint", split_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
    
    processed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed = json.load(f)
        print(f"  从断点恢复: 已处理 {len(processed)} 个样本")
    
    remaining = [item for item in dataset if item["uid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    if not remaining:
        print("  所有样本已处理完毕！")
    else:
        effective_workers = args.max_workers  # 8xA800 vLLM 支持高并发 batch
        pbar = tqdm(total=len(remaining), desc="EgoSchema TASKER-v4")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    run_single_item, item, args.cache_dir,
                    args.search_strategy, args.max_frames, args.init_frames,
                    args.conf_lower, args.min_steps
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
    ground_truth = {item["uid"]: item["answer"] for item in dataset}
    
    eval_result = evaluate_egoschema(predictions, ground_truth)
    eval_result["method"] = f"tasker_v4_coverage_aware_{args.max_frames}frames"
    eval_result["split"] = "Sub." if subset_only else "Full"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result["avg_frames"] = float(np.mean([r.get("num_frames", 0) for r in processed.values()]))
    eval_result["avg_steps"] = float(np.mean([r.get("num_steps", 0) for r in processed.values()]))
    eval_result["avg_forced_coverage"] = float(np.mean([r.get("forced_coverage", 0) for r in processed.values()]))
    
    # 按 stop_reason 统计
    stop_stats = {}
    for r in processed.values():
        reason = r.get("stop_reason", "unknown")
        if reason not in stop_stats:
            stop_stats[reason] = {"total": 0, "correct": 0}
        stop_stats[reason]["total"] += 1
        uid = r["uid"]
        if uid in ground_truth and r["pred"] == ground_truth[uid]:
            stop_stats[reason]["correct"] += 1
    eval_result["stop_stats"] = stop_stats
    
    print(f"\n{'='*60}")
    print(f"  EgoSchema {split_name} TASKER-v4 评测结果:")
    print(f"  准确率: {eval_result['accuracy_pct']}")
    print(f"  平均帧数: {eval_result['avg_frames']:.1f}")
    print(f"  平均搜索步数: {eval_result['avg_steps']:.1f}")
    print(f"  平均强制覆盖次数: {eval_result['avg_forced_coverage']:.1f}")
    print(f"  停止原因: {stop_stats}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"tasker_v4_coverage_aware_{args.max_frames}frames", split_tag)
    
    # Full set: 导出提交文件
    if not subset_only:
        export_egoschema_submission(predictions, f"tasker_v4_coverage_aware_{args.max_frames}frames")
    
    return eval_result


def run_nextqa(args):
    """在 NExT-QA 上评测"""
    print("=" * 60)
    print(f"方法: TASKER-v4 Coverage-Aware (max_frames={args.max_frames})")
    print(f"      覆盖度感知 A* 选帧 → 统一 Final QA")
    print(f"数据集: NExT-QA MC (test)")
    print(f"并行数: {args.max_workers}")
    print("=" * 60)
    
    dataset = load_nextqa_dataset()
    print(f"  加载了 {len(dataset)} 个样本")
    
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "tasker_v4_checkpoint", "nextqa")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
    
    processed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed = json.load(f)
        print(f"  从断点恢复: 已处理 {len(processed)} 个样本")
    
    remaining = [item for item in dataset if item["quid"] not in processed]
    print(f"  待处理: {len(remaining)} 个样本")
    
    if not remaining:
        print("  所有样本已处理完毕！")
    else:
        effective_workers = args.max_workers  # 8xA800 vLLM 支持高并发 batch
        pbar = tqdm(total=len(remaining), desc="NExTQA TASKER-v4")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    run_single_item, item, args.cache_dir,
                    args.search_strategy, args.max_frames, args.init_frames,
                    args.conf_lower, args.min_steps
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
    
    pred_list = [{"quid": uid, "pred": r["pred"]} for uid, r in processed.items()]
    eval_result = evaluate_nextqa(pred_list, dataset)
    eval_result["method"] = f"tasker_v4_coverage_aware_{args.max_frames}frames"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result["avg_frames"] = float(np.mean([r.get("num_frames", 0) for r in processed.values()]))
    eval_result["avg_steps"] = float(np.mean([r.get("num_steps", 0) for r in processed.values()]))
    
    stop_stats = {}
    for r in processed.values():
        reason = r.get("stop_reason", "unknown")
        if reason not in stop_stats:
            stop_stats[reason] = 0
        stop_stats[reason] += 1
    eval_result["stop_stats"] = stop_stats
    
    print(f"\n{'='*60}")
    print(f"  NExT-QA TASKER-v4 评测结果:")
    print(f"  Avg: {eval_result['Avg']}")
    print(f"  平均帧数: {eval_result['avg_frames']:.1f}")
    print(f"  平均搜索步数: {eval_result['avg_steps']:.1f}")
    print(f"  停止原因: {stop_stats}")
    if "by_category" in eval_result:
        for cat, info in eval_result["by_category"].items():
            print(f"  {cat}: {info['accuracy_pct']}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"tasker_v4_coverage_aware_{args.max_frames}frames", "nextqa")
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TASKER-v4 Coverage-Aware 评测")
    parser.add_argument("--dataset", type=str, choices=["egoschema", "nextqa", "both"], default="both")
    parser.add_argument("--max_frames", type=int, default=TARGET_MAX_FRAMES,
                        help="最大帧数上限 (与 Uniform 对齐)")
    parser.add_argument("--max_workers", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help="并行处理的视频数")
    parser.add_argument("--init_frames", type=int, default=INIT_FRAMES,
                        help="初始均匀采样帧数")
    parser.add_argument("--search_strategy", type=str, default="a_star",
                        choices=["a_star"],
                        help="搜索策略 (v4 统一使用 coverage-aware A*)")
    parser.add_argument("--conf_lower", type=int, default=3,
                        help="置信度阈值 (暂未使用，保留接口)")
    parser.add_argument("--min_steps", type=int, default=2,
                        help="最少搜索步数")
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
