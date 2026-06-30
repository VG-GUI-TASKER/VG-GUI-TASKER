#!/usr/bin/env python3
"""
VG-GUI-Bench Leaderboard 结果汇总脚本（论文格式）

指标说明：
- Acc: Overall accuracy score (W_TYPE=0.3, W_PARAM=0.7)
- Type Acc: Action type match accuracy
- CLICK/SCROLL/TYPE/PRESS/ZOOM/FINISH: Per-action-type scores
- Comp: Episode Completion rate (avg proportion of correct steps per episode)
- Eff ↓: Efficiency (avg number of input frames per step, lower is better)
- PIR: Performance Improvement Rate = (Acc_video - Acc_no_video) / Acc_no_video

用法：
    python aggregate_leaderboard.py [--log_root ./logs] [--output_dir ./leaderboard]
"""

import os
import re
import csv
import json
import argparse
from glob import glob
from collections import defaultdict
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================

MODEL_DISPLAY_NAMES = {
    "qwen3vl": "Qwen3-VL-235B-A22B",
    "gemini_flash": "Gemini-3.1-Flash",
    "gemini_pro": "Gemini-3.1-Pro",
    "claude_sonnet": "Claude-Sonnet-4.6",
    "seed2": "Seed-2.0-Pro",
    "kimi": "Kimi-K2.5",
    "gpt5": "GPT-5",
    "gpt5_mini": "GPT-5-Mini",
}

MODEL_ORDER = ["qwen3vl", "gemini_flash", "gemini_pro", "claude_sonnet", "seed2", "kimi", "gpt5", "gpt5_mini"]


# ============================================================================
# 从 log 文件中提取指标
# ============================================================================

def extract_metrics_from_log(log_path):
    """从 run.log 中提取所有指标"""
    metrics = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Overall Avg Score (= Acc)
    m = re.search(r'\[Overall Avg Score\]:\s*([\d.]+)', content)
    if m:
        metrics["Acc"] = float(m.group(1))
    
    # Type Acc - 从 "Overall Type Acc" 行
    m = re.search(r'Overall Type Acc\s*\|\s*([\d.]+)', content)
    if m:
        metrics["Type_Acc"] = float(m.group(1))
    
    # Per-action scores - 从详细表格
    for action in ["CLICK", "SCROLL", "TYPE", "PRESS", "ZOOM", "FINISH"]:
        pattern = rf'{action}\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)'
        m = re.search(pattern, content)
        if m:
            metrics[f"{action}_Count"] = int(m.group(1))
            metrics[f"{action}_Score"] = float(m.group(2))
            metrics[f"{action}_TypeAcc"] = float(m.group(3))
    
    # Avg Episode Completion (= Comp)
    m = re.search(r'Avg Episode Completion\s*:\s*([\d.]+)', content)
    if m:
        metrics["Comp"] = float(m.group(1))
    
    # Avg Images Per Step (= Eff)
    m = re.search(r'Avg Images Per Step\s*:\s*([\d.]+)', content)
    if m:
        metrics["Eff"] = float(m.group(1))
    
    # Total Steps
    m = re.search(r'Total Steps\s*\|\s*(\d+)', content)
    if m:
        metrics["Total_Steps"] = int(m.group(1))
    
    return metrics


def find_log_file(eval_dir):
    """找到评测目录中的 run.log"""
    log_path = os.path.join(eval_dir, "run.log")
    if os.path.exists(log_path):
        return log_path
    return None


def collect_all_results(log_root):
    """收集所有评测结果"""
    results = defaultdict(dict)
    
    for model_key in MODEL_ORDER:
        for mode in ["single", "uniform10"]:
            eval_name = f"{model_key}_{mode}"
            eval_dir = os.path.join(log_root, eval_name)
            
            if not os.path.isdir(eval_dir):
                continue
            
            log_path = find_log_file(eval_dir)
            if log_path is None:
                print(f"  [WARN] 未找到 run.log: {eval_dir}")
                continue
            
            metrics = extract_metrics_from_log(log_path)
            if metrics:
                results[model_key][mode] = metrics
                print(f"  [OK] {eval_name}: Acc={metrics.get('Acc', 0):.4f}, Comp={metrics.get('Comp', 0):.4f}")
            else:
                print(f"  [WARN] 无法解析: {log_path}")
    
    return results


# ============================================================================
# 计算 PIR
# ============================================================================

def compute_pir(acc_video, acc_no_video):
    """PIR = (Acc_video - Acc_no_video) / Acc_no_video"""
    if acc_no_video is None or acc_no_video == 0:
        return None
    if acc_video is None:
        return None
    return (acc_video - acc_no_video) / acc_no_video


# ============================================================================
# 生成论文格式表格
# ============================================================================

def generate_paper_table(results):
    """生成类似论文 Table 的 Markdown 表格"""
    lines = []
    lines.append("# VG-GUI-Bench Leaderboard")
    lines.append("")
    lines.append(f"> Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # ========== Mode A (No Video) ==========
    lines.append("## Mode A: No Video Input (single)")
    lines.append("")
    lines.append("| Method | Acc.(%) | Type Acc.(%) | CLICK | SCROLL | TYPE | PRESS | ZOOM | FINISH | Comp.(%) | Eff.↓ |")
    lines.append("|--------|:-------:|:------------:|:-----:|:------:|:----:|:-----:|:----:|:------:|:--------:|:-----:|")
    
    # 按 Acc 排序
    sorted_models = sorted(
        MODEL_ORDER,
        key=lambda m: results.get(m, {}).get("single", {}).get("Acc", 0),
        reverse=True
    )
    
    for model_key in sorted_models:
        m = results.get(model_key, {}).get("single", {})
        if not m:
            continue
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        acc = f"{m.get('Acc', 0)*100:.2f}"
        type_acc = f"{m.get('Type_Acc', 0)*100:.2f}"
        click = f"{m.get('CLICK_Score', 0)*100:.2f}"
        scroll = f"{m.get('SCROLL_Score', 0)*100:.2f}"
        type_s = f"{m.get('TYPE_Score', 0)*100:.2f}"
        press = f"{m.get('PRESS_Score', 0)*100:.2f}"
        zoom = f"{m.get('ZOOM_Score', 0)*100:.2f}"
        finish = f"{m.get('FINISH_Score', 0)*100:.2f}"
        comp = f"{m.get('Comp', 0)*100:.2f}"
        eff = f"{m.get('Eff', 0):.0f}"
        
        lines.append(f"| {name} | {acc} | {type_acc} | {click} | {scroll} | {type_s} | {press} | {zoom} | {finish} | {comp} | {eff} |")
    
    lines.append("")
    
    # ========== Mode B (10 Frames) ==========
    lines.append("## Mode B: With Video Input (uniform10)")
    lines.append("")
    lines.append("| Method | Acc.(%) | Type Acc.(%) | CLICK | SCROLL | TYPE | PRESS | ZOOM | FINISH | Comp.(%) | Eff.↓ | PIR |")
    lines.append("|--------|:-------:|:------------:|:-----:|:------:|:----:|:-----:|:----:|:------:|:--------:|:-----:|:---:|")
    
    sorted_models_b = sorted(
        MODEL_ORDER,
        key=lambda m: results.get(m, {}).get("uniform10", {}).get("Acc", 0),
        reverse=True
    )
    
    for model_key in sorted_models_b:
        m = results.get(model_key, {}).get("uniform10", {})
        m_single = results.get(model_key, {}).get("single", {})
        if not m:
            continue
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        acc = f"{m.get('Acc', 0)*100:.2f}"
        type_acc = f"{m.get('Type_Acc', 0)*100:.2f}"
        click = f"{m.get('CLICK_Score', 0)*100:.2f}"
        scroll = f"{m.get('SCROLL_Score', 0)*100:.2f}"
        type_s = f"{m.get('TYPE_Score', 0)*100:.2f}"
        press = f"{m.get('PRESS_Score', 0)*100:.2f}"
        zoom = f"{m.get('ZOOM_Score', 0)*100:.2f}"
        finish = f"{m.get('FINISH_Score', 0)*100:.2f}"
        comp = f"{m.get('Comp', 0)*100:.2f}"
        eff = f"{m.get('Eff', 0):.2f}"
        
        pir = compute_pir(m.get('Acc'), m_single.get('Acc'))
        pir_str = f"{pir:.3f}" if pir is not None else "—"
        
        lines.append(f"| {name} | {acc} | {type_acc} | {click} | {scroll} | {type_s} | {press} | {zoom} | {finish} | {comp} | {eff} | {pir_str} |")
    
    lines.append("")
    
    # ========== 合并表（论文风格） ==========
    lines.append("## Combined Table (Paper Format)")
    lines.append("")
    lines.append("| Method | Mode | Acc.(%) | Type Acc.(%) | CLICK | SCROLL | TYPE | PRESS | ZOOM | FINISH | Comp.(%) | Eff.↓ | PIR |")
    lines.append("|--------|------|:-------:|:------------:|:-----:|:------:|:----:|:-----:|:----:|:------:|:--------:|:-----:|:---:|")
    
    for model_key in sorted_models_b:
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        
        # Mode A row
        m_a = results.get(model_key, {}).get("single", {})
        if m_a:
            lines.append(
                f"| {name} | No Video | "
                f"{m_a.get('Acc',0)*100:.2f} | {m_a.get('Type_Acc',0)*100:.2f} | "
                f"{m_a.get('CLICK_Score',0)*100:.2f} | {m_a.get('SCROLL_Score',0)*100:.2f} | "
                f"{m_a.get('TYPE_Score',0)*100:.2f} | {m_a.get('PRESS_Score',0)*100:.2f} | "
                f"{m_a.get('ZOOM_Score',0)*100:.2f} | {m_a.get('FINISH_Score',0)*100:.2f} | "
                f"{m_a.get('Comp',0)*100:.2f} | {m_a.get('Eff',0):.0f} | — |"
            )
        
        # Mode B row
        m_b = results.get(model_key, {}).get("uniform10", {})
        if m_b:
            pir = compute_pir(m_b.get('Acc'), m_a.get('Acc'))
            pir_str = f"{pir:.3f}" if pir is not None else "—"
            lines.append(
                f"| | Uniform10 | "
                f"{m_b.get('Acc',0)*100:.2f} | {m_b.get('Type_Acc',0)*100:.2f} | "
                f"{m_b.get('CLICK_Score',0)*100:.2f} | {m_b.get('SCROLL_Score',0)*100:.2f} | "
                f"{m_b.get('TYPE_Score',0)*100:.2f} | {m_b.get('PRESS_Score',0)*100:.2f} | "
                f"{m_b.get('ZOOM_Score',0)*100:.2f} | {m_b.get('FINISH_Score',0)*100:.2f} | "
                f"{m_b.get('Comp',0)*100:.2f} | {m_b.get('Eff',0):.2f} | {pir_str} |"
            )
    
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- **Acc.**: Overall accuracy = Σ 𝟙(type_match) × (0.3 + 0.7 × 𝟙(param_match)) / N")
    lines.append("- **Type Acc.**: Proportion of steps where predicted action type matches ground truth")
    lines.append("- **Comp.**: Episode Completion = avg(correct_steps / total_steps) per episode")
    lines.append("- **Eff.↓**: Average number of input frames per prediction step (lower = more efficient)")
    lines.append("- **PIR**: Performance Improvement Rate = (Acc_video − Acc_no_video) / Acc_no_video")
    lines.append("")
    
    return "\n".join(lines)


def generate_csv(results, output_path):
    """生成完整 CSV"""
    fieldnames = [
        "Model", "Mode", "Acc", "Type_Acc",
        "CLICK_Score", "SCROLL_Score", "TYPE_Score", "PRESS_Score", "ZOOM_Score", "FINISH_Score",
        "Comp", "Eff", "PIR", "Total_Steps"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_key in MODEL_ORDER:
            name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            m_single = results.get(model_key, {}).get("single", {})
            m_uniform = results.get(model_key, {}).get("uniform10", {})
            
            # Mode A
            if m_single:
                writer.writerow({
                    "Model": name, "Mode": "No Video",
                    "Acc": f"{m_single.get('Acc', 0):.4f}",
                    "Type_Acc": f"{m_single.get('Type_Acc', 0):.4f}",
                    "CLICK_Score": f"{m_single.get('CLICK_Score', 0):.4f}",
                    "SCROLL_Score": f"{m_single.get('SCROLL_Score', 0):.4f}",
                    "TYPE_Score": f"{m_single.get('TYPE_Score', 0):.4f}",
                    "PRESS_Score": f"{m_single.get('PRESS_Score', 0):.4f}",
                    "ZOOM_Score": f"{m_single.get('ZOOM_Score', 0):.4f}",
                    "FINISH_Score": f"{m_single.get('FINISH_Score', 0):.4f}",
                    "Comp": f"{m_single.get('Comp', 0):.4f}",
                    "Eff": f"{m_single.get('Eff', 0):.2f}",
                    "PIR": "",
                    "Total_Steps": m_single.get('Total_Steps', ''),
                })
            
            # Mode B
            if m_uniform:
                pir = compute_pir(m_uniform.get('Acc'), m_single.get('Acc'))
                writer.writerow({
                    "Model": name, "Mode": "Uniform10",
                    "Acc": f"{m_uniform.get('Acc', 0):.4f}",
                    "Type_Acc": f"{m_uniform.get('Type_Acc', 0):.4f}",
                    "CLICK_Score": f"{m_uniform.get('CLICK_Score', 0):.4f}",
                    "SCROLL_Score": f"{m_uniform.get('SCROLL_Score', 0):.4f}",
                    "TYPE_Score": f"{m_uniform.get('TYPE_Score', 0):.4f}",
                    "PRESS_Score": f"{m_uniform.get('PRESS_Score', 0):.4f}",
                    "ZOOM_Score": f"{m_uniform.get('ZOOM_Score', 0):.4f}",
                    "FINISH_Score": f"{m_uniform.get('FINISH_Score', 0):.4f}",
                    "Comp": f"{m_uniform.get('Comp', 0):.4f}",
                    "Eff": f"{m_uniform.get('Eff', 0):.2f}",
                    "PIR": f"{pir:.4f}" if pir is not None else "",
                    "Total_Steps": m_uniform.get('Total_Steps', ''),
                })
    
    print(f"CSV saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VG-GUI-Bench Leaderboard 汇总（论文格式）")
    parser.add_argument("--log_root", type=str, default="./logs", help="评测日志根目录")
    parser.add_argument("--output_dir", type=str, default="./leaderboard", help="输出目录")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VG-GUI-Bench Leaderboard 汇总")
    print("=" * 60)
    
    # 收集
    results = collect_all_results(args.log_root)
    if not results:
        print("\n[ERROR] 未找到评测结果！")
        return
    
    # 输出
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Markdown
    md_content = generate_paper_table(results)
    md_path = os.path.join(args.output_dir, "leaderboard.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"\nMarkdown saved: {md_path}")
    
    # CSV
    csv_path = os.path.join(args.output_dir, "leaderboard.csv")
    generate_csv(results, csv_path)
    
    # JSON
    json_path = os.path.join(args.output_dir, "leaderboard.json")
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "models": MODEL_DISPLAY_NAMES,
        "results": dict(results),
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")
    
    # 终端预览
    print("\n")
    print(md_content)


if __name__ == "__main__":
    main()
