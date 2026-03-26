"""
vllm_tool_v2 - 新版工具模块，从 JSON 配置文件读取 IP 数据。

配置文件：vllm_ips_config.json
"""

from api.gpt_new import VLLMModel, VLLMEmbeddingModel
import os
import json
import random
import concurrent.futures
from glob import glob
from tqdm import tqdm
import time
import threading
from polaris.api.consumer import create_consumer_by_config
from polaris.pkg.model.service import GetInstancesRequest

# ==================== 配置文件路径 ====================
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "vllm_ips_config.json")

# ==================== 配置缓存 ====================
_config_cache = None
_config_mtime = 0
_config_lock = threading.Lock()


def _load_config(force=False):
    """
    加载 JSON 配置文件，带 mtime 缓存机制。
    只有文件修改时间变化时才重新解析。
    """
    global _config_cache, _config_mtime

    try:
        current_mtime = os.path.getmtime(CONFIG_FILE_PATH)
    except OSError:
        raise FileNotFoundError(
            f"配置文件不存在: {CONFIG_FILE_PATH}\n"
            f"请先运行 migrate_ips_to_json.py 生成配置文件，或通过 vllm_manager.py 管理配置。"
        )

    if not force and _config_cache is not None and current_mtime == _config_mtime:
        return _config_cache

    with _config_lock:
        # Double-check after acquiring lock
        if not force and _config_cache is not None and current_mtime == _config_mtime:
            return _config_cache

        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
        _config_mtime = current_mtime

    return _config_cache


# ==================== Polaris Consumer ====================
g_consumer_api = None


def get_consumer_api():
    global g_consumer_api
    if not g_consumer_api:
        g_consumer_api = create_consumer_by_config("")
    return g_consumer_api


# ==================== IP List 构建 ====================


def get_ip_list_map(ip_list, gpu_nums, begin_port=8000):
    """保持与原版兼容的函数"""
    ip_list_map = {}
    for single_ip_list in ip_list:
        for single_ip in single_ip_list['ip']:
            for i in range(gpu_nums):
                ip_list_map[f'{single_ip}:{begin_port + i}'] = {
                    'model_path': single_ip_list['model_path'],
                    'api_key': single_ip_list.get('api_key', ''),
                }
    return ip_list_map


# ==================== 核心类 get_api ====================


class get_api:
    def __init__(self, name_list, EXTRA_PARAMS=None, model_type='', max_try=3,
                 refresh_interval=120, service_type="generate", split=None, **kwargs):
        self.name_list = name_list
        if EXTRA_PARAMS is None:
            EXTRA_PARAMS = {}
        if 'timeout' not in EXTRA_PARAMS:
            EXTRA_PARAMS['timeout'] = 120.0

        self.EXTRA_PARAMS = EXTRA_PARAMS
        self.model_type = model_type
        self.max_try = max_try
        self.service_type = service_type
        self.split = split if split is not None else ['default']
        self.kwargs = kwargs

        self.refresh_interval = refresh_interval
        self.last_refresh_time = time.monotonic()
        self.refresh_lock = threading.Lock()

        self.refresh_api_instance()

    def _get_ip_list_map_with_custom_params(self, ip_list, default_gpu_nums, default_begin_port):
        """
        为每个 IP 配置单独处理 begin_port 和 gpu_nums 参数
        如果 ip_list 中的元素包含 begin_port 或 gpu_nums，则使用该值，否则使用默认值
        同时根据 split 参数过滤 IP 配置
        """
        ip_list_map = {}
        for single_ip_list in ip_list:
            # 检查 split 是否匹配，默认为 'default'
            ip_split = single_ip_list.get('split', 'default')
            if ip_split not in self.split:
                continue

            if single_ip_list.get('type', 'self_vllm') == 'polaris':
                try:
                    for _ in range(3):
                        namespace = "Production"
                        service = single_ip_list['polaris_name']
                        request = GetInstancesRequest(namespace=namespace, service=service)
                        response = get_consumer_api().get_all_instances(request)
                        if response.is_service_data_expired():
                            print(
                                "service {namespace}/{service} data is expired".format(
                                    namespace=namespace, service=service
                                )
                            )
                        else:
                            ip_list = [instance.get_host() for instance in response]
                            port_list = [instance.get_port() for instance in response]
                            if len(ip_list) == 0:
                                time.sleep(1)  # 不知道为什么，一站式请求有时候会返回空list，只能重试一下了
                                continue

                            for single_ip, single_port in zip(ip_list, port_list):
                                ip_list_map[f'{single_ip}:{single_port}'] = {
                                    'model_path': single_ip_list['model_path'],
                                    'api_key': single_ip_list.get('api_key', ''),
                                }
                        
                        break
                except:
                    continue
            else:
                current_begin_port = single_ip_list.get('begin_port', default_begin_port)
                current_gpu_nums = single_ip_list.get('gpu_nums', default_gpu_nums)

                for single_ip in single_ip_list['ip']:
                    for i in range(current_gpu_nums):
                        ip_list_map[f'{single_ip}:{current_begin_port + i}'] = {
                            'model_path': single_ip_list['model_path'],
                            'api_key': single_ip_list.get('api_key', ''),
                        }
        if ip_list_map:
            return ip_list_map
        else:
            return None

    def refresh_api_instance(self):
        """
        从 JSON 配置文件读取配置，替代原来的 importlib.reload 机制。
        支持 v2 list 格式和 v1 dict 格式的 model_types 和 entries。
        """
        try:
            config = _load_config(force=True)
        except FileNotFoundError as e:
            print(f"[vllm_tool_v2] 配置文件加载失败: {e}")
            return

        raw_model_types = config.get("model_types", {})

        # 兼容 v2 list 格式和 v1 dict 格式
        if isinstance(raw_model_types, list):
            # v2 格式: model_types 是 list，每项有 "name" 字段
            mt_config = None
            available_names = []
            for mt in raw_model_types:
                available_names.append(mt.get("name", ""))
                if mt.get("name") == self.model_type:
                    mt_config = mt
                    break
            if mt_config is None:
                raise ValueError(
                    f'Invalid model type: {self.model_type}\n'
                    f'Available model types: {available_names}'
                )
        else:
            # v1 格式: model_types 是 dict
            if self.model_type not in raw_model_types:
                raise ValueError(
                    f'Invalid model type: {self.model_type}\n'
                    f'Available model types: {list(raw_model_types.keys())}'
                )
            mt_config = raw_model_types[self.model_type]

        raw_entries = mt_config.get("entries", {})
        default_begin_port = mt_config.get("default_begin_port")
        default_gpu_nums = mt_config.get("default_gpu_nums")

        # 兼容 v2 list 格式和 v1 dict 格式的 entries
        if isinstance(raw_entries, list):
            # v2 格式: entries 是 list，每项有 "key" 字段
            name_to_ip = {}
            for entry in raw_entries:
                k = entry.get("key", "")
                name_to_ip[k] = {ek: ev for ek, ev in entry.items() if ek != "key"}
        else:
            # v1 格式: entries 是 dict
            name_to_ip = raw_entries

        # 构建 ip_list
        ip_list = []
        if 'all' in self.name_list:
            for name in name_to_ip:
                ip_list.append(name_to_ip[name])
        else:
            for name in self.name_list:
                if name not in name_to_ip:
                    raise KeyError(
                        f"Entry '{name}' not found in model_type '{self.model_type}'\n"
                        f"Available entries: {list(name_to_ip.keys())}"
                    )
                ip_list.append(name_to_ip[name])

        # 为每个 IP 配置单独处理 begin_port 和 gpu_nums
        new_ip_list_map = self._get_ip_list_map_with_custom_params(
            ip_list, default_gpu_nums, default_begin_port
        )

        if new_ip_list_map is None:
            print(f"[vllm_tool] WARNING: No valid IPs found for model_type='{self.model_type}', "
                  f"name_list={self.name_list}, split={self.split}. "
                  f"Check vllm_ips_config.json entries.")
            return

        # 如果已有实例，更新 IP 列表而非重新创建，以保留统计数据
        if hasattr(self, 'api_instance') and self.api_instance is not None:
            self.api_instance.update_ip_list(new_ip_list_map)
        else:
            if self.service_type == 'embedding':
                self.api_instance = VLLMEmbeddingModel(
                    IP_LIST_MAP=new_ip_list_map,
                    EXTRA_PARAMS=self.EXTRA_PARAMS,
                    max_try=self.max_try,
                    **self.kwargs
                )
            else:
                self.api_instance = VLLMModel(
                    IP_LIST_MAP=new_ip_list_map,
                    EXTRA_PARAMS=self.EXTRA_PARAMS,
                    max_try=self.max_try,
                    **self.kwargs
                )

    def _thread_safe_refresh(self):
        """线程安全的刷新检查与执行"""
        with self.refresh_lock:
            current_time = time.monotonic()
            if current_time - self.last_refresh_time >= self.refresh_interval:
                self.refresh_api_instance()
                self.last_refresh_time = current_time

    def openapi_call(self, *args, **kwds):
        if random.random() > 0.99 and time.monotonic() - self.last_refresh_time >= self.refresh_interval:
            self._thread_safe_refresh()
        return self.api_instance.openapi_call(*args, **kwds)

    def __call__(self, *args, **kwds):
        if random.random() > 0.99 and time.monotonic() - self.last_refresh_time >= self.refresh_interval:
            self._thread_safe_refresh()
        return self.api_instance(*args, **kwds)


# ==================== 通用工具函数（与原版完全相同） ====================


def multi_thread_request(
    process_func, output_file_path, all_data, num_threads, max_pending_tasks=1000,
    desc='', writing_process_func=None, show_success_ratio=0, filter_func=None,
):
    max_pending_tasks = max(max_pending_tasks, 2 * num_threads)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'a') as adder, concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = set()
        processed_count = 0
        success_count = 0

        from collections import deque
        recent_results = deque(maxlen=show_success_ratio) if show_success_ratio > 0 else None

        try:
            total_length = len(all_data)
            data_iter = all_data
        except:
            total_length = float("inf")
            try:
                data_iter = all_data()
            except:
                data_iter = all_data

        with tqdm(total=total_length, desc=desc) as pbar:
            for item in data_iter:
                if filter_func:
                    filter_item = filter_func(item)
                    if filter_item is None:
                        continue
                    item = filter_item

                while len(futures) >= max_pending_tasks:
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        result = future.result()
                        is_success = bool(result)
                        if is_success:
                            if writing_process_func:
                                writing_process_func(adder, result)
                            else:
                                adder.write(json.dumps(result, ensure_ascii=False) + '\n')
                            success_count += 1

                        processed_count += 1
                        if recent_results is not None:
                            recent_results.append(is_success)
                            recent_success = sum(recent_results)
                            recent_total = len(recent_results)
                            pbar.set_postfix(recent_success_ratio=f'{recent_success}/{recent_total}={recent_success/recent_total*100:.2f}%')
                        pbar.update(1)

                    adder.flush()

                future = executor.submit(process_func, item)
                futures.add(future)

            while futures:
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    result = future.result()
                    is_success = bool(result)
                    if is_success:
                        if writing_process_func:
                            writing_process_func(adder, result)
                        else:
                            adder.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1

                    processed_count += 1
                    if recent_results is not None:
                        recent_results.append(is_success)
                        recent_success = sum(recent_results)
                        recent_total = len(recent_results)
                        pbar.set_postfix(recent_success_ratio=f'{recent_success}/{recent_total}={recent_success/recent_total*100:.2f}%')
                    pbar.update(1)
                adder.flush()


def get_unprocessed_data(
    input_file_path=None, output_file_path=None, input_file_pattern=None,
    input_key='data_id', output_key='data_id', filter_func=None,
    do_shuffle=True, shuffle_seed=None
):
    def get_unique_key(data, key):
        if isinstance(key, list):
            unique_key = ''
            for single_key in key:
                unique_key += get_unique_key(data, single_key)
            return str(unique_key)
        elif isinstance(key, dict):
            unique_key = ''
            for single_key in key:
                unique_key += get_unique_key(data[single_key], key[single_key])
            return str(unique_key)
        else:
            return str(data[key])

    def _data_generator(processed_idx_set, show_progress=True):
        skip_nums = 0
        if input_file_pattern:
            print(f'loading input data: {input_file_pattern}...')
            all_file_paths = glob(input_file_pattern, recursive=True)
            all_file_paths.sort()
            print(f'found {len(all_file_paths)} files to process: \n{all_file_paths}\n\n')

            for jsonl_file in all_file_paths:
                with open(jsonl_file, 'r') as reader:
                    if show_progress:
                        pbar = tqdm(reader, desc=f'loading {os.path.basename(jsonl_file)}')
                    else:
                        pbar = reader

                    for line in pbar:
                        item = json.loads(line.strip())
                        if filter_func:
                            filter_item = filter_func(item)
                            if filter_item is None:
                                continue
                            item = filter_item

                        unique_key = get_unique_key(item, input_key)
                        if unique_key not in processed_idx_set:
                            yield item
                            processed_idx_set.add(unique_key)
                        else:
                            skip_nums += 1
        else:
            print(f'loading input data: {input_file_path}...')
            with open(input_file_path, 'r') as reader:
                if show_progress:
                    pbar = tqdm(reader)
                else:
                    pbar = reader

                for line in pbar:
                    item = json.loads(line.strip())
                    if filter_func:
                        filter_item = filter_func(item)
                        if filter_item is None:
                            continue
                        item = filter_item

                    unique_key = get_unique_key(item, input_key)
                    if unique_key not in processed_idx_set:
                        yield item
                        processed_idx_set.add(unique_key)
                    else:
                        skip_nums += 1

    processed_idx_set = set()
    if os.path.exists(output_file_path):
        print(f'loading processed data :{output_file_path}...')
        with open(output_file_path, 'r') as reader:
            for line in tqdm(reader):
                data = json.loads(line.strip())
                processed_idx_set.add(get_unique_key(data, output_key))

    if not do_shuffle:
        return _data_generator(processed_idx_set, show_progress=False)
    else:
        final_data = list(_data_generator(processed_idx_set, show_progress=True))
        if shuffle_seed:
            random.seed(shuffle_seed)
        random.shuffle(final_data)
        return final_data
