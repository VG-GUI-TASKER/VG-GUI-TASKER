import os
import ast
import csv
import json
import logging
import datetime
import numpy as np
import re
import math
from collections import defaultdict
from core import action_matching

# ==========================================
# 1. 基础配置与工具函数
# ==========================================

# 评分权重
W_TYPE = 0.3  # 类型正确的得分权重 (30%)
W_PARAM = 0.7 # 参数正确的得分权重 (70%)

# 阈值设置
CLICK_DIST_THRESHOLD = 0.14   # 点击距离阈值 (屏幕对角线比例)
SCROLL_DIST_THRESHOLD = 0.14  # 滑动距离阈值
TEXT_SIMILARITY_THRESHOLD = 0.8 # 文本被视为"正确"的最低相似度

def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_text_similarity(s1, s2):
    """计算归一化的文本相似度 (0.0 ~ 1.0)"""
    s1 = str(s1).strip().lower()
    s2 = str(s2).strip().lower()
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)

def get_scroll_direction_label(start, end):
    """根据滑动向量判断方向标签"""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dy) > abs(dx): # 垂直
        return "scroll down" if dy < 0 else "scroll up"
    else: # 水平
        return "scroll left" if dx > 0 else "scroll right"

def parse_model_response(response):
    """解析模型输出的 Function Call 字符串"""
    if not response: return "NULL", {}
    response = response.strip()
    
    # CLICK(x, y)
    match = re.search(r"CLICK\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)", response, re.IGNORECASE)
    if match: return "CLICK", {"point": [float(match.group(1)), float(match.group(2))]}
    
    # SCROLL(x1, y1, x2, y2)
    match = re.search(r"SCROLL\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)", response, re.IGNORECASE)
    if match:
        return "SCROLL", {
            "start": [float(match.group(1)), float(match.group(2))],
            "end": [float(match.group(3)), float(match.group(4))]
        }
        
    # TYPE("text")
    match = re.search(r"TYPE\s*\(\s*([\"'])(.*?)\1\s*\)", response, re.IGNORECASE | re.DOTALL)
    if match: return "TYPE", {"text": match.group(2)}
    
    # PRESS("key")
    match = re.search(r"PRESS\s*\(\s*([\"'])(.*?)\1\s*\)", response, re.IGNORECASE)
    if match: return "PRESS", {"key": match.group(2).upper()}
    
    # ZOOM / FINISH (HARDWARE 已移除)
    if "ZOOM" in response.upper(): return "ZOOM", {}
    if "FINISH" in response.upper(): return "FINISH", {}
    
    return "UNKNOWN", {}

def process_ground_truth(action_dict):
    """将原始数据集的 action_dict 转换为标准化的 GT 格式"""
    action_type = action_dict.get("action_type_id")
    text_label = action_dict.get("action_type_text", "").lower()
    
    gt = {"type": None, "params": {}}
    
    if action_type == 4:
        touch = action_dict.get("touch", [0.0, 0.0])
        lift = action_dict.get("lift", [0.0, 0.0])
        if "scroll" in text_label or touch != lift:
            gt["type"] = "SCROLL"
            gt["params"] = {"start": touch, "end": lift, "direction": text_label}
        else:
            gt["type"] = "CLICK"
            gt["params"] = {"point": touch, "annot_position": action_dict.get("annot_position", [])}
            
    elif action_type == 3:
        gt["type"] = "TYPE"
        gt["params"] = {"text": action_dict.get("type_text", "")}
        
    elif action_type == 5: gt["type"] = "PRESS"; gt["params"] = {"key": "BACK"}
    elif action_type == 6: gt["type"] = "PRESS"; gt["params"] = {"key": "HOME"}
    elif action_type == 7: gt["type"] = "PRESS"; gt["params"] = {"key": "ENTER"}
    elif action_type == 12: gt["type"] = "ZOOM"
    # ID 14 (HARDWARE) 已移除，将落入下面的 else 变成 UNKNOWN
    else: gt["type"] = "UNKNOWN"
        
    return gt

# ==========================================
# 2. 核心评测逻辑
# ==========================================

def evaluate(j, response, action_common):
    """单步评测函数"""
    result = {
        "score": 0.0, "type_correct": False, "param_score": 0.0,
        "gt_type": "None", "pred_type": "None", "error": ""
    }

    try:
        pred_type, pred_params = parse_model_response(response)
        result["pred_type"] = pred_type
    except Exception as e:
        result["error"] = f"Parse Error: {e}"
        return result

    best_step_score = 0.0
    best_step_info = None

    for action_dict in action_common:
        gt = process_ground_truth(action_dict)
        current_type_correct = False
        current_param_score = 0.0
        
        if pred_type == gt["type"]:
            current_type_correct = True
            
            if pred_type == "CLICK":
                pred_pt = pred_params.get("point")
                gt_pt = gt["params"].get("point")
                # A. 检查红框
                annot_pos = gt["params"].get("annot_position", [])
                hit_bbox = False
                if annot_pos:
                    py, px = pred_pt[1], pred_pt[0] 
                    for k in range(0, len(annot_pos), 4):
                        if k+3 >= len(annot_pos): break
                        ay, ax, ah, aw = annot_pos[k:k+4]
                        if (ax <= px <= ax + aw) and (ay <= py <= ay + ah):
                            hit_bbox = True; break
                # B. 检查距离
                dist = math.dist(pred_pt, gt_pt)
                if hit_bbox or (dist <= CLICK_DIST_THRESHOLD): current_param_score = 1.0
            
            elif pred_type == "SCROLL":
                pred_start = pred_params.get("start")
                pred_end = pred_params.get("end")
                pred_dir = get_scroll_direction_label(pred_start, pred_end)
                gt_dir = gt["params"].get("direction")
                
                if pred_dir == gt_dir:
                    current_param_score = 0.5 
                    gt_start = gt["params"].get("start")
                    gt_end = gt["params"].get("end")
                    if math.dist(pred_start, gt_start) <= SCROLL_DIST_THRESHOLD and \
                       math.dist(pred_end, gt_end) <= SCROLL_DIST_THRESHOLD:
                        current_param_score = 1.0 
            
            elif pred_type == "TYPE":
                pred_text = pred_params.get("text", "")
                gt_text = gt["params"].get("text", "")
                sim = calculate_text_similarity(pred_text, gt_text)
                if sim >= TEXT_SIMILARITY_THRESHOLD: current_param_score = sim
                else: current_param_score = 0.0

            elif pred_type == "PRESS":
                if pred_params.get("key") == gt["params"].get("key"): current_param_score = 1.0
            
            # 移除了 HARDWARE
            elif pred_type in ["ZOOM", "FINISH"]:
                current_param_score = 1.0
                
        step_score = (W_TYPE + current_param_score * W_PARAM) if current_type_correct else 0.0
            
        if step_score >= best_step_score:
            best_step_score = step_score
            best_step_info = {
                "score": step_score, "type_correct": current_type_correct,
                "param_score": current_param_score, "gt_type": gt["type"], "pred_type": pred_type
            }
    
    if best_step_info: result.update(best_step_info)
    else:
        if action_common:
            gt0 = process_ground_truth(action_common[0])
            result["gt_type"] = gt0["type"]
            
    return result

# ==========================================
# 3. 统计与汇总
# ==========================================

def calculate_metrics(results):
    metrics = {"Total_Score": 0.0, "Total_Count": 0, "Type_Acc": 0.0}
    class_stats = defaultdict(lambda: {"count": 0, "type_correct": 0, "score": 0.0})
    
    total_score = 0.0
    total_type_correct = 0
    count = 0
    
    for episode in results.values():
        for step_res in episode:
            count += 1
            total_score += step_res["score"]
            if step_res["type_correct"]: total_type_correct += 1
            
            gt_type = step_res["gt_type"]
            class_stats[gt_type]["count"] += 1
            class_stats[gt_type]["score"] += step_res["score"]
            if step_res["type_correct"]: class_stats[gt_type]["type_correct"] += 1
                
    if count > 0:
        metrics["Total_Score"] = total_score / count
        metrics["Type_Acc"] = total_type_correct / count
    metrics["Total_Count"] = count
    
    logging.info("\n" + "="*65)
    logging.info(f"{'Metric':<20} | {'Value':<10}")
    logging.info("-" * 35)
    logging.info(f"{'Total Steps':<20} | {count:<10}")
    logging.info(f"{'Overall Score':<20} | {metrics['Total_Score']:<10.4f}")
    logging.info(f"{'Overall Type Acc':<20} | {metrics['Type_Acc']:<10.4f}")
    logging.info("="*65 + "\n")

    logging.info("Detailed Breakdown by Action Type:")
    logging.info("-" * 65)
    logging.info(f"{'Type':<12} | {'Num':<6} | {'Score (Acc)':<15} | {'Type Acc':<12}")
    logging.info("-" * 65)
    
    # [MOD] 移除了 HARDWARE
    ALL_TYPES = ["CLICK", "SCROLL", "TYPE", "PRESS", "ZOOM", "FINISH"]
    for action_type in ALL_TYPES:
        stat = class_stats.get(action_type, {"count": 0, "type_correct": 0, "score": 0.0})
        cnt = stat["count"]
        if cnt > 0:
            avg_score = stat["score"] / cnt
            type_acc = stat["type_correct"] / cnt
        else:
            avg_score = 0.0
            type_acc = 0.0
            
        metrics[f"{action_type}_Count"] = cnt
        metrics[f"{action_type}_Score"] = avg_score
        metrics[f"{action_type}_TypeAcc"] = type_acc
        
        logging.info(f"{action_type:<12} | {cnt:<6} | {avg_score:<15.4f} | {type_acc:<12.4f}")
    
    logging.info("-" * 65 + "\n")

    return metrics

def summarize_and_save_results(args, predictions, start_timestamp, extra_info=None):
    with open(args.prediction_file_path, 'w') as fp:
        json.dump(predictions, fp, indent=2)
    print(f"Predictions saved at {args.prediction_file_path}")

    results = {task: {} for task in predictions.keys()}
    for task, episodes in predictions.items():
        for ep_id, steps in episodes.items():
            if ep_id not in results[task]: results[task][ep_id] = []
            for j, step in enumerate(steps):
                output = evaluate(j, step['response'], step['action_common'])
                results[task][ep_id].append(output)

    eval_dict = {}
    for task in results.keys():
        logging.info("==="*20)
        logging.info(f"Task: {task}")
        eval_dict[task] = calculate_metrics(results[task])

    # [MOD] 移除了 HARDWARE 专项报告模块

    if len(eval_dict) > 0:
        avg_score = sum([x["Total_Score"] for x in eval_dict.values()]) / len(eval_dict)
    else:
        avg_score = 0.0
    logging.info(f"[Overall Avg Score]: {avg_score:.4f}")

    # =========================================================
    # 额外统计：评测元信息、Episode 完成度、图片数量
    # =========================================================
    END_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # 兼容新旧时间戳格式
    for fmt in ('%Y%m%d%H%M%S', '%Y_%m_%d-%H_%M_%S'):
        try:
            t_start = datetime.datetime.strptime(start_timestamp, fmt)
            break
        except ValueError:
            t_start = None
    t_end = datetime.datetime.strptime(END_TIMESTAMP, '%Y%m%d%H%M%S')
    time_elapsed = (t_end - t_start).total_seconds() if t_start else 0

    logging.info("\n" + "=" * 65)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 65)
    logging.info(f"  Eval Start Time  : {start_timestamp}")
    logging.info(f"  Eval End Time    : {END_TIMESTAMP}")
    logging.info(f"  Time Elapsed     : {str(datetime.timedelta(seconds=int(time_elapsed)))}")
    if extra_info:
        logging.info(f"  Ref Mode         : {extra_info.get('ref_mode', 'N/A')}")
        logging.info(f"  Ref Images Dir   : {extra_info.get('ref_imgs_dir', 'N/A')}")
        logging.info(f"  No Cut           : {extra_info.get('no_cut', 'N/A')}")

    # Episode 完成度 & 图片数量统计
    all_completion_rates = []
    all_image_counts = []

    for task, episodes in predictions.items():
        for ep_id, steps in episodes.items():
            ep_results = results.get(task, {}).get(ep_id, [])
            total_steps = len(ep_results)
            correct_steps = sum(1 for r in ep_results if r.get("score", 0) > 0)
            if total_steps > 0:
                completion = correct_steps / total_steps
            else:
                completion = 0.0
            all_completion_rates.append(completion)

            # 统计该 episode 每步发了几张图
            for step_pred in steps:
                num_imgs = step_pred.get("num_images", 1)
                all_image_counts.append(num_imgs)

    if all_completion_rates:
        avg_completion = sum(all_completion_rates) / len(all_completion_rates)
    else:
        avg_completion = 0.0

    if all_image_counts:
        avg_images = sum(all_image_counts) / len(all_image_counts)
    else:
        avg_images = 0.0

    logging.info(f"  Avg Episode Completion : {avg_completion:.4f} ({len(all_completion_rates)} episodes)")
    logging.info(f"  Avg Images Per Step    : {avg_images:.2f} ({len(all_image_counts)} steps)")
    logging.info("=" * 65)

    with open(args.csv_path, 'w') as fp:
        all_keys = set()
        for m in eval_dict.values(): all_keys.update(m.keys())
        fieldnames = ["Task"] + sorted(list(all_keys)) + list(vars(args).keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for task, metrics in eval_dict.items():
            row = {"Task": task, **metrics, **vars(args)}
            writer.writerow(row)
    print(f"Metrics saved at {args.csv_path}")

# ==========================================
# 4. 历史动作格式化工具
# ==========================================
def format_history_action(step_data):
    """
    将历史动作数据转换为 Function Call 格式字符串。
    用于在 Prompt 中展示 History。
    """
    action_type = step_data.get("action_type_id")
    text_label = step_data.get("action_type_text", "").lower()
    
    if action_type == 4:
        touch = step_data.get("touch", [0.0, 0.0])
        lift = step_data.get("lift", [0.0, 0.0])
        is_scroll = "scroll" in text_label or touch != lift
        if is_scroll:
            return f"SCROLL({touch[0]:.3f}, {touch[1]:.3f}, {lift[0]:.3f}, {lift[1]:.3f})"
        else:
            return f"CLICK({touch[0]:.3f}, {touch[1]:.3f})"

    elif action_type == 3:
        text = step_data.get("type_text", "")
        safe_text = text.replace('"', '\\"')
        return f'TYPE("{safe_text}")'
        
    elif action_type == 5: return 'PRESS("BACK")'
    elif action_type == 6: return 'PRESS("HOME")'
    elif action_type == 7: return 'PRESS("ENTER")'
    elif action_type == 12: return "ZOOM()"
    # ID 14 (HARDWARE) 已移除，不生成对应的历史记录（返回None，会被上层逻辑忽略）
    elif action_type == 10: return "FINISH()"
    elif action_type == 11: return "FINISH()"
    
    return None