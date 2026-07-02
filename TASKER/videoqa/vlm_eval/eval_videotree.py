"""
方法2: VideoTree 取帧 + VLM 理解
使用 CLIP (ViT-L/14) 语义特征做聚类（忠实于原版 VideoTree）

流程:
1. 从视频中按 1 FPS 采样所有帧
2. 用 CLIP (openai/clip-vit-large-patch14) 提取语义特征
3. Phase 1: 自适应宽度扩展 (K-Means + VLM 相关性评分)
4. Phase 2: 基于相关性的深度扩展 (层次聚类)
5. 用选出的关键帧让 VLM 回答问题
"""
import os
import sys
import json
import argparse
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    VIDEOTREE_INIT_CLUSTER_NUM, VIDEOTREE_MAX_CLUSTER_NUM,
    VIDEOTREE_ADAPTIVE_RATE, VIDEOTREE_ITER_THRESHOLD,
    VIDEOTREE_NUM_SUBCLUSTERS, VIDEOTREE_NUM_SUBSUBCLUSTERS,
    VIDEOTREE_MAX_FRAMES, MAX_CONCURRENT_REQUESTS, RESULTS_BASE_DIR, SAMPLE_FPS,
    CLIP_MODEL_PATH
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
#  CLIP 模型管理（全局单例，线程安全）
# ============================================================

_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()
_clip_device = None


def get_clip_model():
    """获取全局 CLIP 模型实例（单例，线程安全）"""
    global _clip_model, _clip_processor, _clip_device
    
    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                # 优先使用本地模型，其次尝试在线下载
                if os.path.exists(CLIP_MODEL_PATH):
                    model_path = CLIP_MODEL_PATH
                    print(f"  [CLIP] 加载本地模型: {model_path}")
                else:
                    model_path = "openai/clip-vit-large-patch14"
                    print(f"  [CLIP] 本地模型不存在，尝试在线下载: {model_path}")
                
                # 强制使用 CPU（GPU 被 Qwen VLM 占满）
                _clip_device = "cpu"
                
                # 加载模型
                from transformers import CLIPModel
                _clip_model = CLIPModel.from_pretrained(model_path).to(_clip_device).eval()
                
                # 加载图像预处理器（多种尝试，确保兼容性）
                try:
                    from transformers import CLIPImageProcessor
                    _clip_processor = CLIPImageProcessor.from_pretrained(model_path)
                    print(f"  [CLIP] 使用 CLIPImageProcessor")
                except Exception as e1:
                    print(f"  [CLIP] CLIPImageProcessor 加载失败: {e1}")
                    try:
                        from transformers import AutoImageProcessor
                        _clip_processor = AutoImageProcessor.from_pretrained(model_path)
                        print(f"  [CLIP] 使用 AutoImageProcessor")
                    except Exception as e2:
                        print(f"  [CLIP] AutoImageProcessor 也失败: {e2}")
                        # 最后的兜底：手动构建 processor
                        from torchvision import transforms
                        _clip_processor = transforms.Compose([
                            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]
                            ),
                        ])
                        print(f"  [CLIP] 使用 torchvision 手动 transform (兜底)")
                
                param_count = sum(p.numel() for p in _clip_model.parameters()) / 1e6
                print(f"  [CLIP] 模型加载成功 ({param_count:.0f}M params, device={_clip_device})")
    
    return _clip_model, _clip_processor, _clip_device


# ============================================================
#  帧特征提取（CLIP 语义特征）
# ============================================================

def extract_clip_features(img_paths: list) -> torch.Tensor:
    """
    用 CLIP (ViT-L/14) 提取帧的视觉语义特征。
    返回 (N, 768) 的 tensor (L2 normalized)。
    
    使用 batch 处理提升效率。
    """
    from PIL import Image
    from torchvision import transforms as tv_transforms
    
    model, processor, device = get_clip_model()
    
    # 判断 processor 类型：HF processor vs torchvision transforms
    is_hf_processor = hasattr(processor, '__call__') and hasattr(processor, 'feature_extractor_type' if hasattr(processor, 'feature_extractor_type') else 'image_mean')
    # 更可靠的判断：检查是否是 torchvision.transforms.Compose
    is_torchvision = isinstance(processor, tv_transforms.Compose)
    
    features = []
    batch_size = 32  # CLIP ViT-L/14 比较轻量，可以大 batch
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # 打开失败时用黑色图片占位
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        
        if is_torchvision:
            # torchvision transforms 兜底方案
            pixel_values = torch.stack([processor(img) for img in images]).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)
        else:
            # HF ImageProcessor（标准路径）
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
        
        # 新版 transformers 可能返回 BaseModelOutputWithPooling 而非 tensor
        if not isinstance(image_features, torch.Tensor):
            # 取 pooler_output 或 last_hidden_state 的 CLS token
            if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state[:, 0, :]  # CLS token
            else:
                # 最后兜底：尝试索引
                image_features = image_features[0]
        
        # L2 normalize（与原版一致，聚类用 cosine distance）
        image_features = F.normalize(image_features, dim=1)
        features.append(image_features.cpu())
    
    return torch.cat(features, dim=0)  # (N, 768)


# ============================================================
#  K-Means (cosine distance) — 忠实于原版 VideoTree
# ============================================================

def kmeans_cosine(X, num_clusters, iter_limit=20, tol=1e-4):
    """K-Means with cosine distance (忠实于原版 kmeans_pytorch)"""
    X = X.float()
    num_samples = len(X)
    num_clusters = min(num_clusters, num_samples)
    
    if num_clusters <= 0:
        return torch.zeros(num_samples, dtype=torch.long), X[:1]
    
    # 初始化：随机选中心（与原版 kmeans_pytorch 一致）
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    centers = X[indices].clone()
    
    for _ in range(iter_limit):
        # cosine similarity → 分配
        X_norm = F.normalize(X, dim=1)
        C_norm = F.normalize(centers, dim=1)
        sim = torch.mm(X_norm, C_norm.T)  # (N, K)
        cluster_ids = torch.argmax(sim, dim=1)
        
        # 更新中心
        centers_prev = centers.clone()
        for k in range(num_clusters):
            mask = cluster_ids == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(dim=0)
            else:
                centers[k] = X[torch.randint(num_samples, (1,))]
        
        shift = torch.sqrt(((centers - centers_prev) ** 2).sum(dim=1)).sum()
        if shift ** 2 < tol:
            break
    
    return cluster_ids, centers


# ============================================================
#  Phase 1: 自适应宽度扩展
#  忠实于原版 VideoTree/adaptive_breath_expansion.py
# ============================================================

def adaptive_width_expansion(
    frame_features: torch.Tensor,
    frame_paths: list,
    question: str,
    options: list,
    init_cluster_num: int = 8,
    max_cluster_num: int = 32,
    adaptive_rate: int = 2,
    iter_threshold: int = 4,
):
    """
    自适应宽度扩展（忠实于原版）：
    1. K-Means cosine 聚类
    2. 每簇选距中心最近的代表帧
    3. VLM 评分（相关性 1/2/3）
    4. 高相关帧不够 → 翻倍 k，直到满足或达上限
    
    Returns:
        cluster_ids (list), relevance_scores (list), tree_nodes (list)
    """
    cluster_num = init_cluster_num
    option_text = " | ".join([f"({chr(65+i)}) {o}" for i, o in enumerate(options)])
    goal = f"Question: {question}\nOptions: {option_text}"
    
    while True:
        cluster_ids, centers = kmeans_cosine(frame_features, cluster_num)
        
        # 找每簇最近中心的帧（欧氏距离，与原版 find_closest_points_per_cluster 一致）
        tree_nodes = []
        for k in range(cluster_num):
            mask = (cluster_ids == k)
            if mask.sum() == 0:
                continue
            indices_in_cluster = torch.where(mask)[0]
            points = frame_features[indices_in_cluster]
            dists = torch.norm(points - centers[k], dim=1)
            closest_idx = indices_in_cluster[torch.argmin(dists)].item()
            tree_nodes.append(closest_idx)
        
        tree_nodes.sort()
        
        # VLM 相关性评分
        node_paths = [frame_paths[i] for i in tree_nodes]
        relevance_scores = vllm_relevance_scoring(node_paths, goal)
        
        high_count = sum(1 for s in relevance_scores if s == 3)
        
        if high_count >= iter_threshold or cluster_num >= max_cluster_num:
            break
        
        cluster_num = min(cluster_num * adaptive_rate, max_cluster_num)
    
    return cluster_ids.tolist(), relevance_scores, tree_nodes


def vllm_relevance_scoring(frame_paths: list, goal: str) -> list:
    """让 VLM 对帧打相关性分数 (1/2/3)"""
    N = len(frame_paths)
    
    prompt = (
        f"You are shown {N} key frames sampled from a video.\n\n"
        f"Task: {goal}\n\n"
        f"Rate each frame's relevance to answering the question above.\n"
        f"Score each frame: 1 = low relevance, 2 = medium, 3 = high relevance.\n"
        f"Respond ONLY in format: [{', '.join(['score'] * min(N, 5))}...]\n"
        f"Example for 4 frames: [2, 1, 3, 2]"
    )
    
    response = call_qwen_vl(
        question=prompt,
        image_paths=frame_paths,
        system_prompt="You are a video analysis assistant. Rate frame relevance scores.",
        temperature=0.0,
        max_tokens=256,
    )
    
    if response:
        match = re.search(r'\[([0-9,\s]+)\]', response)
        if match:
            try:
                scores = list(map(int, match.group(1).split(',')))
                if len(scores) == N:
                    return [max(1, min(3, s)) for s in scores]
            except:
                pass
    
    # fallback: 全部给 2 分
    return [2] * N


# ============================================================
#  Phase 2: 基于相关性的深度扩展
#  完全忠实于原版 VideoTree/depth_expansion.py
# ============================================================

def cosine_similarity_dist(points, centroid):
    """
    计算 cosine distances (1 - cosine_similarity)
    与原版 depth_expansion.cosine_similarity 完全一致
    """
    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()


def hierarchical_clustering_with_external_primary(
    video_features: torch.Tensor,
    cluster_ids: list,
    relevance_scores: list,
    num_subclusters: int = 4,
    num_subsubclusters: int = 4,
) -> dict:
    """
    与原版 VideoTree/depth_expansion.py 完全一致。
    
    根据 relevance_scores 决定聚类深度:
    - score=1: clusters[cluster_id] = [primary_indices]  (list, 只选1个代表帧)
    - score=2: clusters[cluster_id] = {subcluster_id: [indices]}  (2级)
    - score=3: clusters[cluster_id] = {subcluster_id: {subsubcluster_id: [indices]}}  (3级)
    """
    clusters = {i: {} for i in range(0, max(cluster_ids) + 1)}

    for cluster_id in set(cluster_ids):
        primary_indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]

        if cluster_id < len(relevance_scores):
            score = relevance_scores[cluster_id]
        else:
            score = 3

        if len(primary_indices) < 2:
            clusters[cluster_id] = primary_indices
            continue

        sub_features = video_features[primary_indices].numpy()

        if score == 1:
            clusters[cluster_id] = primary_indices
            continue

        linked_sub = linkage(sub_features, method='ward')
        sub_cluster_labels = fcluster(linked_sub, num_subclusters, criterion='maxclust')
        sub_cluster_labels = sub_cluster_labels - 1

        if score == 2:
            clusters[cluster_id] = {
                i: [primary_indices[j] for j in np.where(sub_cluster_labels == i)[0]]
                for i in range(0, num_subclusters)
            }
            continue

        # score == 3
        for subcluster_id in range(0, num_subclusters):
            sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
            if len(sub_indices) < 2:
                continue

            subsub_features = sub_features[sub_indices]
            linked_subsub = linkage(subsub_features, method='ward')
            subsub_cluster_labels = fcluster(linked_subsub, num_subsubclusters, criterion='maxclust')
            subsub_cluster_labels = subsub_cluster_labels - 1

            clusters[cluster_id][subcluster_id] = {}
            for subsubcluster_id in range(0, num_subsubclusters):
                final_indices = sub_indices[np.where(subsub_cluster_labels == subsubcluster_id)[0]]
                original_indices = [primary_indices[i] for i in final_indices]
                clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices

    return clusters


def find_closest_points_in_temporal_order_subsub(
    x: torch.Tensor,
    clusters: dict,
    relevance_scores: list,
) -> list:
    """
    与原版 VideoTree/depth_expansion.py 完全一致。
    
    使用 cosine similarity 为每个 cluster 选代表帧：
    - score=1 (list): 1 个代表帧
    - score=2 (dict of lists): 1 个全局代表 + 每个 subcluster 1 个代表
    - score=3 (dict of dicts): 1 个全局代表 + 每个 sub-subcluster 1 个代表
    """
    closest_points_indices = []

    for cluster_id, cluster_data in clusters.items():
        if cluster_id < len(relevance_scores):
            relevance = relevance_scores[cluster_id]
        else:
            relevance = 3

        if isinstance(cluster_data, list):  # Primary cluster directly (score=1 or < 2 frames)
            cluster_data_arr = np.array(cluster_data)
            if cluster_data_arr.size == 0:
                continue
            points_in_cluster = x[torch.tensor(cluster_data_arr, dtype=torch.long)]
            cluster_centroid = points_in_cluster.mean(dim=0)
            distances = cosine_similarity_dist(points_in_cluster, cluster_centroid)
            if distances.numel() > 0:
                closest_idx = torch.argmin(distances).item()
                closest_points_indices.append(int(cluster_data_arr[closest_idx]))

        elif isinstance(cluster_data, dict):  # Handle subclusters and sub-subclusters
            if relevance == 1:
                # Only take 1 representative frame for the primary cluster
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if len(sub_data) > 0:
                                primary_indices.append(np.array(sub_data))
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        primary_indices.append(np.array(subcluster_data))

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity_dist(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))
                continue

            elif relevance == 2 or relevance == 3:
                # 1. First: Include primary cluster representative
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if len(sub_data) > 0:
                                primary_indices.append(sub_data)
                    elif isinstance(subcluster_data, list) and len(subcluster_data) > 0:
                        primary_indices.append(np.array(subcluster_data))

                if primary_indices:
                    primary_indices = np.concatenate(primary_indices)
                    primary_points = x[torch.tensor(primary_indices, dtype=torch.long)]
                    primary_centroid = primary_points.mean(dim=0)
                    primary_distances = cosine_similarity_dist(primary_points, primary_centroid)
                    if primary_distances.numel() > 0:
                        closest_primary_idx = torch.argmin(primary_distances).item()
                        closest_points_indices.append(int(primary_indices[closest_primary_idx]))

                # 2. Then: Per-subcluster or per-sub-subcluster representatives
                for subcluster_id, subclusters in cluster_data.items():
                    if isinstance(subclusters, dict):  # Sub-subclusters (score=3)
                        for subsubcluster_id, indices in subclusters.items():
                            if len(indices) == 0:
                                continue
                            indices_tensor = torch.tensor(indices, dtype=torch.long)
                            points_in_subsubcluster = x[indices_tensor]
                            subsubcluster_centroid = points_in_subsubcluster.mean(dim=0)
                            distances = cosine_similarity_dist(points_in_subsubcluster, subsubcluster_centroid)
                            if distances.numel() > 0:
                                closest_idx_in_subsubcluster = torch.argmin(distances).item()
                                closest_global_idx = indices[closest_idx_in_subsubcluster]
                                closest_points_indices.append(int(closest_global_idx))

                    elif isinstance(subclusters, list):  # Subclusters (score=2)
                        subclusters_arr = np.array(subclusters)
                        if subclusters_arr.size == 0:
                            continue
                        points_in_subcluster = x[torch.tensor(subclusters_arr, dtype=torch.long)]
                        subcluster_centroid = points_in_subcluster.mean(dim=0)
                        distances = cosine_similarity_dist(points_in_subcluster, subcluster_centroid)
                        if distances.numel() > 0:
                            closest_idx = torch.argmin(distances).item()
                            closest_points_indices.append(int(subclusters_arr[closest_idx]))

    # 去重并按时间排序（与原版一致）
    closest_points_indices = sorted(list(set(closest_points_indices)))
    return closest_points_indices


def depth_expansion(
    frame_features: torch.Tensor,
    cluster_ids: list,
    relevance_scores: list,
    num_subclusters: int = 4,
    num_subsubclusters: int = 4,
) -> list:
    """
    深度扩展完整流程，忠实于原版 VideoTree/depth_expansion.py
    """
    clusters = hierarchical_clustering_with_external_primary(
        frame_features, cluster_ids, relevance_scores,
        num_subclusters=num_subclusters,
        num_subsubclusters=num_subsubclusters,
    )
    
    selected_indices = find_closest_points_in_temporal_order_subsub(
        frame_features, clusters, relevance_scores
    )
    
    return selected_indices


# ============================================================
#  VideoTree 完整流程
# ============================================================

def videotree_select_frames(video_path: str, question: str, options: list, 
                            cache_dir: str, max_frames: int = 16) -> list:
    """
    VideoTree 完整帧选择流程（使用 CLIP 语义特征）
    
    Returns:
        选出的帧路径列表
    """
    # 1. 按 1 FPS 采样
    all_frame_paths, all_frame_indices = extract_frames_at_fps(video_path, fps=SAMPLE_FPS, cache_dir=cache_dir)
    
    if len(all_frame_paths) < 2:
        return all_frame_paths
    
    # 2. 提取 CLIP 语义特征（替代旧版的颜色直方图）
    features = extract_clip_features(all_frame_paths)
    
    # 3. Phase 1: 自适应宽度扩展
    cluster_ids, relevance_scores, tree_nodes = adaptive_width_expansion(
        features, all_frame_paths, question, options,
        init_cluster_num=VIDEOTREE_INIT_CLUSTER_NUM,
        max_cluster_num=VIDEOTREE_MAX_CLUSTER_NUM,
        adaptive_rate=VIDEOTREE_ADAPTIVE_RATE,
        iter_threshold=VIDEOTREE_ITER_THRESHOLD,
    )
    
    # 4. Phase 2: 深度扩展
    selected_indices = depth_expansion(
        features, cluster_ids, relevance_scores,
        num_subclusters=VIDEOTREE_NUM_SUBCLUSTERS,
        num_subsubclusters=VIDEOTREE_NUM_SUBSUBCLUSTERS,
    )
    
    # 5. 限制最大帧数（如果超过，均匀采样）
    if len(selected_indices) > max_frames:
        step = len(selected_indices) / max_frames
        selected_indices = [selected_indices[int(i * step)] for i in range(max_frames)]
    
    # 确保至少有 max_frames 帧（如果 depth_expansion 选得太少，补充 tree_nodes）
    if len(selected_indices) < max_frames:
        # 补充 tree_nodes 中未被选中的帧
        existing = set(selected_indices)
        for node in tree_nodes:
            if node not in existing:
                selected_indices.append(node)
                existing.add(node)
                if len(selected_indices) >= max_frames:
                    break
        selected_indices.sort()
    
    return [all_frame_paths[i] for i in selected_indices if i < len(all_frame_paths)]


_first_error_printed = False
_first_error_lock = threading.Lock()

def run_single_item(item, cache_dir, max_frames):
    """处理单个样本"""
    global _first_error_printed
    video_path = item["video_path"]
    question = item["question"]
    options = item["options"]
    uid = item.get("uid") or item.get("quid")
    
    try:
        # VideoTree 帧选择
        selected_frames = videotree_select_frames(
            video_path, question, options, cache_dir, max_frames
        )
        
        if not selected_frames:
            return {"uid": uid, "pred": -1, "response": None, "num_frames": 0, "error": "无法选帧"}
        
        # 构建 prompt 并调用 VLM
        prompt = build_vqa_prompt(question, options, len(selected_frames))
        
        system_prompt = (
            "You are an expert video understanding assistant. "
            "You are shown key frames selected by an adaptive tree-based algorithm "
            "that focuses on the most relevant parts of the video. "
            "Analyze these frames carefully to answer the question."
        )
        
        response = call_qwen_vl(
            question=prompt,
            image_paths=selected_frames,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=64,
        )
        
        pred = parse_answer(response)
        
        return {
            "uid": uid,
            "pred": pred,
            "response": response,
            "num_frames": len(selected_frames),
        }
    
    except Exception as e:
        # 打印第一个错误（帮助快速诊断）
        with _first_error_lock:
            if not _first_error_printed:
                _first_error_printed = True
                import traceback
                print(f"\n{'!'*60}")
                print(f"  ⚠️  VideoTree-CLIP 首个错误 (uid={uid}):")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                print(f"{'!'*60}\n")
        return {"uid": uid, "pred": -1, "response": None, "num_frames": 0, "error": str(e)}


def run_egoschema(max_frames: int, max_workers: int, cache_dir: str, subset_only: bool = True):
    """在 EgoSchema 上评测"""
    split_name = "subset (500)" if subset_only else "full (5031)"
    print("=" * 60)
    print(f"方法: VideoTree + CLIP (ViT-L/14), max_frames={max_frames}")
    print(f"数据集: EgoSchema {split_name}")
    print(f"并行数: {max_workers}")
    print("=" * 60)
    
    dataset = load_egoschema_dataset(subset_only=subset_only)
    print(f"  加载了 {len(dataset)} 个样本")
    
    # 预加载 CLIP 模型（避免多线程同时初始化）
    print("  预加载 CLIP 模型...")
    get_clip_model()
    
    # Resume
    split_tag = "egoschema_subset" if subset_only else "egoschema_full"
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "videotree_clip_checkpoint", split_tag)
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
        # CLIP 特征提取是 GPU 密集的，并行度要适中
        effective_workers = min(max_workers, 8)
        
        pbar = tqdm(total=len(remaining), desc="EgoSchema VideoTree+CLIP")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(run_single_item, item, cache_dir, max_frames): item
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
    eval_result["method"] = f"videotree_clip_{max_frames}frames"
    eval_result["split"] = "Sub." if subset_only else "Full"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result["avg_frames"] = float(np.mean([r.get("num_frames", 0) for r in processed.values()]))
    
    print(f"\n{'='*60}")
    print(f"  EgoSchema {split_name} VideoTree+CLIP 评测结果:")
    print(f"  准确率 (Table 1 '{eval_result['split']}'): {eval_result['accuracy_pct']}")
    print(f"  平均帧数: {eval_result['avg_frames']:.1f}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"videotree_clip_{max_frames}frames", split_tag)
    
    # Full set: 导出提交文件
    if not subset_only:
        export_egoschema_submission(predictions, f"videotree_clip_{max_frames}frames")
    
    return eval_result


def run_nextqa(max_frames: int, max_workers: int, cache_dir: str):
    """在 NExT-QA 上评测"""
    print("=" * 60)
    print(f"方法: VideoTree + CLIP (ViT-L/14), max_frames={max_frames}")
    print(f"数据集: NExT-QA MC (test)")
    print(f"并行数: {max_workers}")
    print("=" * 60)
    
    dataset = load_nextqa_dataset()
    print(f"  加载了 {len(dataset)} 个样本")
    
    # 预加载 CLIP 模型
    print("  预加载 CLIP 模型...")
    get_clip_model()
    
    checkpoint_dir = os.path.join(RESULTS_BASE_DIR, "videotree_clip_checkpoint", "nextqa")
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
        effective_workers = min(max_workers, 8)
        pbar = tqdm(total=len(remaining), desc="NExTQA VideoTree+CLIP")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(run_single_item, item, cache_dir, max_frames): item
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
    eval_result["method"] = f"videotree_clip_{max_frames}frames"
    eval_result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result["avg_frames"] = float(np.mean([r.get("num_frames", 0) for r in processed.values()]))
    
    print(f"\n{'='*60}")
    print(f"  NExT-QA VideoTree+CLIP 评测结果:")
    print(f"  Avg (Table 1): {eval_result['Avg']}")
    print(f"  平均帧数: {eval_result['avg_frames']:.1f}")
    if "by_category" in eval_result:
        for cat, info in eval_result["by_category"].items():
            print(f"  {cat}: {info['accuracy_pct']}")
    print(f"{'='*60}\n")
    
    save_results(eval_result, f"videotree_clip_{max_frames}frames", "nextqa")
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VideoTree + CLIP 评测")
    parser.add_argument("--dataset", type=str, choices=["egoschema", "nextqa", "both"], default="both")
    parser.add_argument("--max_frames", type=int, default=VIDEOTREE_MAX_FRAMES)
    parser.add_argument("--max_workers", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--cache_dir", type=str, default="/tmp/benchmark_frames")
    parser.add_argument("--egoschema_split", type=str, choices=["subset", "full", "both"], default="subset",
                        help="EgoSchema 评测范围: subset(500), full(5031), 或 both")
    args = parser.parse_args()
    
    if args.dataset in ["egoschema", "both"]:
        splits = []
        if args.egoschema_split in ["subset", "both"]:
            splits.append(True)
        if args.egoschema_split in ["full", "both"]:
            splits.append(False)
        
        for subset_only in splits:
            run_egoschema(args.max_frames, args.max_workers, args.cache_dir, subset_only=subset_only)
    
    if args.dataset in ["nextqa", "both"]:
        run_nextqa(args.max_frames, args.max_workers, args.cache_dir)
