import os
import random
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import json
from tqdm import tqdm
import logging
import argparse
import numpy as np
import datetime
import time
from glob import glob
import concurrent.futures
import threading

from .eval import summarize_and_save_results, format_history_action
from .prompt import prompt_cheat, prompt_keyframe, prompt_public, prompt_single, prompt_uniform
from api.vllm_tool import get_api
from api.closed_model_adapter import build_closed_model, CLOSED_MODEL_CONFIGS
from api.aiping_adapter import build_aiping_model, AIPING_MODEL_CONFIGS

# ========== 所有模式 & 对应默认参考图子目录 ==========
# 值为 images/ 下的子目录名，None 表示不需要参考图
MODE_DIR_MAP = {
    'single':     None,
    'origin':     'origin',
    'gt':         'annotation',
    'annotation': 'annotation',
    'tasker':     'tasker',
    'bfs':        'bfs',
    'gbfs':       'gbfs',
    'dijkstra':   'dijkstra',
    'videoagent': 'videoagent',
    'videotree':  'videotree',
    'uniform5':   'uniform_5',
    'uniform10':  'uniform_10',
}

ALL_MODES = list(MODE_DIR_MAP.keys())


def get_seeclick_response(model, prompt, img_path, ref_imgs=None, system_prompt=None):
    start_time = time.time()
    img_list = []
    if ref_imgs and isinstance(ref_imgs, list):
        img_list.extend(ref_imgs)
    img_list.append(img_path)

    response = model(
        img_path_or_list=img_list,
        question=prompt,
        system_prompt=system_prompt,
        image_first=True
    )
    end_time = time.time()
    logging.debug(f"Inference time: {end_time - start_time:.2f} seconds")
    return response


# ========== 参考图获取 ==========

def get_gt_image(img_filename_base, annot_root_dir):
    """gt 模式：只返回当前 step 对应的唯一一张带红框标注图"""
    if not annot_root_dir or not os.path.exists(annot_root_dir):
        return []
    target_annot_path = os.path.join(annot_root_dir, img_filename_base + '_annot.png')
    if os.path.exists(target_annot_path):
        return [target_annot_path]
    return []


def get_folder_images(video_id, root_dir):
    """通用：获取对应视频文件夹下的所有图片（按文件名排序）"""
    if not root_dir or not os.path.exists(root_dir):
        return []
    video_folder = os.path.join(root_dir, video_id)
    if not os.path.exists(video_folder):
        return []
    ref_imgs = glob(os.path.join(video_folder, "*.jpg")) + glob(os.path.join(video_folder, "*.png"))
    ref_imgs.sort()
    return ref_imgs


def uniform_sample(imgs, max_count=10):
    """当图片超过 max_count 时，均匀采样到 max_count 张"""
    if len(imgs) <= max_count:
        return imgs
    n = len(imgs)
    indices = [round(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
    return [imgs[i] for i in indices]


def get_ref_images(ref_mode, video_id, img_filename_base, ref_imgs_dir):
    """根据 ref_mode 分发到不同的参考图获取策略"""
    if ref_mode == 'single':
        return []
    elif ref_mode == 'gt':
        return get_gt_image(img_filename_base, ref_imgs_dir)
    elif ref_mode == 'annotation':
        imgs = get_folder_images(video_id, ref_imgs_dir)
        return uniform_sample(imgs, max_count=10)
    elif ref_mode == 'origin':
        return get_folder_images(video_id, ref_imgs_dir)
    else:
        # tasker / bfs / gbfs / dijkstra / videoagent / videotree / uniform5 / uniform10
        return get_folder_images(video_id, ref_imgs_dir)


def resolve_ref_imgs_dir(ref_mode, dataset_root, no_cut):
    """根据模式和 cut/nocut 自动推导参考图目录"""
    subdir = MODE_DIR_MAP.get(ref_mode)
    if subdir is None:
        return None  # single 模式不需要参考图
    if no_cut:
        subdir = subdir + '_no_cut'
    return os.path.join(dataset_root, 'images', subdir)


def resolve_imgs_dir(dataset_root, no_cut):
    """自动推导 target screen 图片目录"""
    if no_cut:
        return os.path.join(dataset_root, 'images', 'origin_no_cut')
    else:
        return os.path.join(dataset_root, 'images', 'origin')


def parse_args():
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--qwen_name_list', type=str, default='all')
    parser.add_argument('--model_type', type=str, default='qwen35')
    parser.add_argument('--max_try', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.6)

    # data args
    parser.add_argument('--imgs_dir', type=str, default=None,
                        help='Directory for target screens. Auto-resolved if not set.')
    parser.add_argument('--test_json_path', type=str, default='../data/aitw_data_test.json')

    # 参考图参数
    parser.add_argument('--ref_mode', type=str, required=True, choices=ALL_MODES,
                        help=f'Reference image mode: {"/".join(ALL_MODES)}')
    parser.add_argument('--ref_imgs_dir', type=str, default=None,
                        help='Directory for reference context images. Auto-resolved if not set.')
    parser.add_argument('--dataset_root', type=str,
                        default='/data/home/stevefan/projects/lql/VG-GUI-Bench/MONDAY',
                        help='Dataset root dir.')
    parser.add_argument('--no_cut', action='store_true',
                        help='Use no-cut (uncropped) image directories.')

    # eval config
    parser.add_argument('--multianswer_history_mode', type=str, default='first')
    parser.add_argument('--num_history', type=int, default=4)
    parser.add_argument('--log_root', type=str, default='./logs/')
    parser.add_argument('--eval_name', type=str, default='qwen3vl')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--quick_test', action='store_true',
                        help='Only run 1/50 of episodes for quick testing')
    parser.add_argument('--num_threads', type=int, default=16,
                        help='Number of threads for concurrent API calls')
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.num_history > 0

    # ---- 自动推导路径 ----
    if args.imgs_dir is None:
        args.imgs_dir = resolve_imgs_dir(args.dataset_root, args.no_cut)

    if args.ref_mode == 'single':
        args.ref_imgs_dir = None
    elif args.ref_imgs_dir is None:
        args.ref_imgs_dir = resolve_ref_imgs_dir(args.ref_mode, args.dataset_root, args.no_cut)

    # ---- 构建文件命名：mode_nocut_YYYYMMDDHHmmss ----
    TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    cut_tag = 'nocut' if args.no_cut else 'cut'
    file_prefix = f"{args.ref_mode}_{cut_tag}_{TIMESTAMP}"

    os.makedirs(args.log_root, exist_ok=True)
    args.log_file_path = os.path.join(args.log_root, f'{file_prefix}.log')
    args.prediction_file_path = os.path.join(args.log_root, f'{file_prefix}_prediction.json')
    args.csv_path = os.path.join(args.log_root, f'{file_prefix}_evaluation.csv')

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(args.log_file_path), logging.StreamHandler()])
    logging.info(f"Start evaluation: {file_prefix}")
    logging.info(f"Reference image mode: {args.ref_mode}, no_cut: {args.no_cut}")
    logging.info(f"Resolved imgs_dir: {args.imgs_dir}")
    logging.info(f"Resolved ref_imgs_dir: {args.ref_imgs_dir}")

    arg_dict = vars(args)
    for key, value in sorted(arg_dict.items()):
        logging.info(f"  {key} : {value}")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Init Model
    if args.use_api:
        if args.model_type in AIPING_MODEL_CONFIGS:
            # Aiping.cn 模型：GPT-5 / GPT-5 Mini（OpenAI 兼容格式）
            logging.info(f"使用 Aiping 适配器: {args.model_type}")
            model = build_aiping_model(
                model_type=args.model_type,
                max_try=args.max_try,
                timeout=300,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.model_type in CLOSED_MODEL_CONFIGS:
            # 闭源模型：使用 ClosedModelAdapter（NACI 格式）
            logging.info(f"使用闭源模型适配器: {args.model_type}")
            model = build_closed_model(
                model_type=args.model_type,
                max_try=args.max_try,
                timeout=300,
            )
        else:
            # 开源/自部署模型：使用 vLLM
            name_list = (['all'] if args.qwen_name_list.strip().lower() == 'all'
                         else [x.strip() for x in args.qwen_name_list.split(',')])
            model = get_api(name_list, model_type=args.model_type, max_try=args.max_try,
                            EXTRA_PARAMS={"max_tokens": args.max_tokens, "temperature": args.temperature},
                            split=['gui_video', 'ocr'])
    else:
        raise NotImplementedError

    imgs_dir = args.imgs_dir
    test_data = json.load(open(args.test_json_path, 'r'))

    # =========================================================
    # 根据模式选择 System Prompt
    # =========================================================
    if args.ref_mode == 'gt':
        system_prompt = prompt_cheat
    elif args.ref_mode == 'single':
        system_prompt = prompt_single
    # elif args.ref_mode in ('uniform5', 'uniform10'):
    #     system_prompt = prompt_uniform
    else:
        system_prompt = prompt_keyframe

    # =========================================================
    # User Prompt Template
    # =========================================================
    if args.ref_mode == 'single':
        prompt_user_template = """Task Goal: {}

Previous Actions:
{}

Based on the Task Goal, Previous Actions, and the current Target Screen, what is the exact Next Action? Output ONLY the action format:"""

    elif args.ref_mode == 'gt':
        prompt_user_template = """Instruction: {}

Previous Actions:
{}

Output the Next Action (Extract from the first image):"""

    else:
        prompt_user_template = """Task Goal: {}

Previous Actions:
{}

Based on the provided Reference Frames and the Previous Actions, what is the exact Next Action to perform on the VERY LAST IMAGE (Target Screen)? Output ONLY the action format:"""

    predictions = {task: {} for task in test_data.keys()}
    skip_ep_id = [
        "-rU3IvxM60o", "0X4zAts5Ztg", "17NVq7KGIu0", "1xbO6Qxor3s", "2MkK7p346O4",
        "7hIkuu6oRmM", "82BdLwolMOo", "ALRHSq_ccBc", "EYGhzCYR9tk", "FTMdSF-LVo8",
        "H39s9U-uxpk", "NXw8_0hzl90", "QsSrJdQYsLA", "R4-NitC_XEo", "RgKJy2rY_RQ",
        "Rh1UxulONjI", "T8Aw2S-xgPk", "UBpHRbv0R_A", "Y06w2Y4vcIU", "ZODKMvN7w18",
        "_O7smMRqmtg", "bsOjQvIIFEc", "exNKc-7sshQ", "gjUwSo3A7ss", "i3aWAsn1L8s",
        "ijc-5IwBprE", "kndMclUY41k", "oI1I78orUxQ", "o_KA2Dw2Drc", "qXFJrqoubzc",
        "tJxyaixwRFc", "uKKBMICOoMY", "uYcKkj141MI", "yQUVmKkhnpQ", "z9X8z2WZ1Gg",
    ]

    # =========================================================
    # 阶段1：收集所有需要调用的任务
    # =========================================================
    all_tasks = []

    for task, episodes in test_data.items():
        total_eps = len(episodes)
        if args.quick_test:
            run_count = max(1, total_eps // 50)
            episodes = episodes[:run_count]
            logging.info(f"[QUICK TEST] Running {run_count}/{total_eps} episodes")
        for j, episode in enumerate(episodes):
            if not episode:
                continue
            ep_id = episode[0]['ep_id']
            if ep_id in skip_ep_id:
                continue

            current_history_accumulated = []

            for step_idx, step in enumerate(episode):
                img_filename_base = step["img_filename"]
                img_filename = img_filename_base + '.png'
                img_path = os.path.join(imgs_dir, img_filename)

                # GT Processing for History
                action_list = step.get('action_list', [step])
                valid_actions_str = []
                for ad in action_list:
                    s = format_history_action(ad)
                    if s:
                        valid_actions_str.append(s)

                current_gt_str = valid_actions_str[0] if valid_actions_str else "Wait()"

                if not os.path.exists(img_path):
                    current_history_accumulated.append(current_gt_str)
                    continue

                # 获取 Video ID 和参考图
                video_id = img_filename_base.split('/')[0]
                ref_img_paths = get_ref_images(args.ref_mode, video_id,
                                               img_filename_base, args.ref_imgs_dir)

                # 构建 Prompt History
                history_window = (current_history_accumulated[-args.num_history:]
                                  if args.num_history > 0 else [])
                if history_window:
                    prev_text_formatted = ""
                    for idx, h_str in enumerate(history_window):
                        prev_text_formatted += f"Step {idx}: {h_str}\n"
                else:
                    prev_text_formatted = "None"

                prompt_user = prompt_user_template.format(step["goal"], prev_text_formatted)

                all_tasks.append({
                    "task": task,
                    "ep_id": ep_id,
                    "step_idx": step_idx,
                    "prompt_user": prompt_user,
                    "img_path": img_path,
                    "ref_img_paths": ref_img_paths,
                    "system_prompt": system_prompt,
                    "img_filename": img_filename,
                    "action_list": action_list,
                })

                current_history_accumulated.append(current_gt_str)

    logging.info(f"Total API calls to make: {len(all_tasks)}, using {args.num_threads} threads")

    # =========================================================
    # 阶段2：多线程并发调用
    # =========================================================
    results_lock = threading.Lock()
    progress_bar = tqdm(total=len(all_tasks), desc="API calls")

    first_error_logged = threading.Event()

    def process_one_task(task_info):
        try:
            response = get_seeclick_response(
                model, task_info["prompt_user"], task_info["img_path"],
                task_info["ref_img_paths"], task_info["system_prompt"]
            )
            if response is None or response == "":
                msg = f"Empty response on {task_info['img_filename']}"
                logging.warning(msg)
                if not first_error_logged.is_set():
                    first_error_logged.set()
                    logging.error(f"[FIRST EMPTY] {msg}")
        except Exception as error:
            import traceback
            tb = traceback.format_exc()
            logging.error(f"Error on {task_info['img_filename']}: {error}\n{tb}")
            if not first_error_logged.is_set():
                first_error_logged.set()
                raise  # 让第一个错误直接抛出，终止执行便于调试
            response = None

        prediction = {
            "filename": task_info["img_filename"].split("/")[-1],
            "action_common": task_info["action_list"],
            "response": response,
            "num_images": len(task_info["ref_img_paths"]) + 1
        }

        with results_lock:
            task_name = task_info["task"]
            ep_id = task_info["ep_id"]
            if ep_id not in predictions[task_name]:
                predictions[task_name][ep_id] = []
            predictions[task_name][ep_id].append(prediction)
            progress_bar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_one_task, t) for t in all_tasks]
        concurrent.futures.wait(futures)

    progress_bar.close()

    extra_info = {
        "ref_mode": args.ref_mode,
        "ref_imgs_dir": args.ref_imgs_dir,
        "no_cut": args.no_cut,
    }
    summarize_and_save_results(args, predictions, TIMESTAMP, extra_info=extra_info)


if __name__ == "__main__":
    main()
