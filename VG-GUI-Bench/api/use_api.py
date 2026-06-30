import sys
# sys.path.append('/apdcephfs_gy2/share_303242896/harveyshen/code/1101_xgen/vllm_scripts')
# sys.path.append('/apdcephfs_gy2/share_303242896/harveyshen/share_code/request_vllm')
from api.vllm_tool import get_api
from PIL import Image
import io
import base64
import concurrent.futures
# 所有 Qwen3VL-Thinking 的服务
api = get_api(['all'], model_type='qwen35', max_try=5, split=['gui_video', 'ocr'])

# NOTE: 可以设置 image_first 参数，默认为 False，即文本在前，图片在后


def process_image_gpt(image_path):
    image = Image.open(image_path)
    image = image.resize((image.width // 2, image.height // 2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def batch_api_call(tasks, num_threads=16):
    """
    并发调用 api，加速批量请求。

    Args:
        tasks: 任务列表，每个元素是传给 api() 的参数字典，例如:
            [
                {"img_path": "./api/test_images/test1.jpg", "question": "描述图片"},
                {"img_path": "./api/test_images/test2.jpg", "question": "图中有几个人？"},
            ]
        num_threads: 并发线程数，建议设为服务端 IP 数的 1~2 倍

    Returns:
        与 tasks 顺序一致的结果列表，失败的任务对应值为 None
    """
    results = [None] * len(tasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {executor.submit(api, **task): i for i, task in enumerate(tasks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"任务 {idx} 出错: {e}")
                results[idx] = None

    return results


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 串行调用（单条）
    # ans1 = api(img_path_or_list='test_images/test1.jpg',
    #         question="请描述图片中的内容，并说明图片中有哪些物体？",
    #         system_prompt="请用英语回答")
    # print("ans1:", ans1)

    # 并发调用（批量）
    tasks = [
        {"img_path_or_list": "./api/test_images/test1.jpg", "question": "请描述图片中的内容，并说明图片中有哪些物体？Please answer in English."},
        {"img_path_or_list": "./api/test_images/test1.jpg", "question": "图片中有几个人？"},
        # 继续添加更多任务...
    ]
    results = batch_api_call(tasks, num_threads=16)
    for i, res in enumerate(results):
        print(f"任务 {i}: {res}")

