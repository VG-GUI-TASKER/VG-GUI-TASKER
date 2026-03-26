import json
import requests
import imghdr
import copy
import base64
import random
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
import time
import threading
import datetime
import hmac
import hashlib
import uuid
from PIL import Image

# patch 格式相关的代码
from PIL import Image, ImageFile
import io
import base64
import json
import argparse
import os
from tqdm import tqdm
from collections import OrderedDict
from threading import Lock, RLock
from typing import Optional, Tuple, Dict, Any
import random

# 添加 HEIC 支持
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed, HEIC support disabled")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class PatchFileManager:
    """
    Patch文件管理器，使用LRU缓存策略管理打开的文件句柄。
    支持并发读取，通过细粒度锁保证线程安全。
    """
    
    def __init__(self, max_open_files: int = 10):
        """
        初始化Patch文件管理器
        
        Args:
            max_open_files: 最多同时打开的patch文件数量
        """
        self.max_open_files = max_open_files
        # 使用OrderedDict实现LRU，最近使用的在末尾
        self._file_handles: OrderedDict[str, io.BufferedReader] = OrderedDict()
        # 每个文件一个锁，用于并发读取同一文件
        self._file_locks: Dict[str, Lock] = {}
        # 全局锁，用于管理文件句柄的打开/关闭
        self._global_lock = RLock()
    
    def _get_or_open_file(self, patch_path: str) -> Tuple[io.BufferedReader, Lock]:
        """
        获取或打开一个patch文件
        
        Args:
            patch_path: patch文件路径
            
        Returns:
            (文件句柄, 该文件的锁)
        """
        with self._global_lock:
            # 如果文件已打开，移动到末尾（标记为最近使用）
            if patch_path in self._file_handles:
                self._file_handles.move_to_end(patch_path)
                return self._file_handles[patch_path], self._file_locks[patch_path]
            
            # 如果达到最大打开数，关闭最久未使用的文件
            while len(self._file_handles) >= self.max_open_files:
                oldest_path, oldest_handle = self._file_handles.popitem(last=False)
                try:
                    oldest_handle.close()
                except Exception:
                    pass
                # 保留锁，因为可能有线程正在等待
                # 锁会在下次打开同一文件时复用
            
            # 打开新文件
            handle = open(patch_path, 'rb')
            self._file_handles[patch_path] = handle
            
            # 为新文件创建锁（如果不存在）
            if patch_path not in self._file_locks:
                self._file_locks[patch_path] = Lock()
            
            return handle, self._file_locks[patch_path]
    
    def read_bytes(self, patch_path: str, start_num: int, size: int) -> bytes:
        """
        从patch文件中读取指定位置的字节数据
        
        Args:
            patch_path: patch文件路径
            start_num: 起始位置
            size: 读取大小
            
        Returns:
            读取的字节数据
        """
        handle, file_lock = self._get_or_open_file(patch_path)
        
        # 使用文件锁保证seek和read的原子性
        with file_lock:
            # 需要重新获取handle，因为可能在等待锁的过程中被关闭
            with self._global_lock:
                if patch_path not in self._file_handles:
                    # 文件被关闭了，重新打开
                    handle, file_lock = self._get_or_open_file(patch_path)
                else:
                    handle = self._file_handles[patch_path]
                    self._file_handles.move_to_end(patch_path)
            
            handle.seek(start_num)
            data = handle.read(size)
        
        return data
    
    def read_image(self, patch_path: str, start_num: int, size: int) -> Image.Image:
        img_bytes = self.read_bytes(patch_path, start_num, size)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        return img_pil, img_bytes

    
    def close_all(self):
        """关闭所有打开的文件"""
        with self._global_lock:
            for handle in self._file_handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._file_handles.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False
    
    def __del__(self):
        self.close_all()


# 全局默认的PatchFileManager实例
_default_patch_manager: Optional[PatchFileManager] = None
_default_manager_lock = Lock()


def get_default_patch_manager(max_open_files: int = 200) -> PatchFileManager:
    """获取或创建默认的PatchFileManager实例"""
    global _default_patch_manager
    with _default_manager_lock:
        if _default_patch_manager is None:
            _default_patch_manager = PatchFileManager(max_open_files)
        return _default_patch_manager


# patch 格式相关的代码结束

class ThreadSafeCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()  # 创建锁对象
    
    def increment(self):
        """线程安全的自加操作"""
        with self.lock:  # 上下文管理器自动获取和释放锁
            self.count += 1
            return self.count  # 返回自增后的值
    
    def get_count(self):
        """获取当前计数（同样需要线程安全）"""
        with self.lock:
            return self.count


class ConcurrencyTracker:
    """追踪每个服务的并发数"""
    def __init__(self):
        self.concurrency = {}  # ip -> 当前并发数
        self.lock = threading.Lock()
    
    def increment(self, ip):
        with self.lock:
            self.concurrency[ip] = self.concurrency.get(ip, 0) + 1
    
    def decrement(self, ip):
        with self.lock:
            self.concurrency[ip] = max(0, self.concurrency.get(ip, 0) - 1)
    
    def get_stats(self):
        with self.lock:
            return dict(self.concurrency)


def get_vllm_server_metrics(ip, timeout=2):
    """
    从 vLLM 服务的 /metrics 端点获取服务端并发信息
    Args:
        ip: 服务地址 (host:port)
        timeout: 请求超时时间
    Returns:
        dict: {'running': int, 'waiting': int} 或 None（获取失败时）
    """
    import re
    try:
        resp = requests.get(f"http://{ip}/metrics", timeout=timeout)
        if resp.status_code != 200:
            return None
        
        text = resp.text
        
        # 解析 running 请求数
        running_match = re.search(r'vllm:num_requests_running\{[^}]*\}\s+([\d.]+)', text)
        waiting_match = re.search(r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)', text)
        
        running = int(float(running_match.group(1))) if running_match else 0
        waiting = int(float(waiting_match.group(1))) if waiting_match else 0
        
        return {'running': running, 'waiting': waiting}
    except Exception as e:
        return None


class AdaptiveLoadBalancer:
    """基于响应时间的自适应负载均衡器，支持熔断和自动恢复"""
    def __init__(self, ips, alpha=0.3, min_weight=0.1, max_weight=10.0,
                 circuit_breaker_threshold=10, probe_interval=30, probe_ratio=0.05, timeout_threshold=600.0):
        """
        Args:
            ips: IP列表
            alpha: 指数移动平均的平滑系数 (0-1)，越大对最新响应时间越敏感
            min_weight: 最小权重
            max_weight: 最大权重
            circuit_breaker_threshold: 连续失败次数阈值，超过后熔断该服务
            probe_interval: 熔断后探测间隔（秒），每隔多久发送一次探测请求
            probe_ratio: 探测请求比例，对熔断服务发送请求的概率
        """
        self.lock = threading.Lock()
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.probe_interval = probe_interval
        self.probe_ratio = probe_ratio
        self.timeout_threshold = timeout_threshold
        
        # 每个IP的统计信息
        self.stats = {}
        for ip in ips:
            self.stats[ip] = {
                'ema_response_time': self.timeout_threshold,  # 指数移动平均响应时间
                'weight': 1.0,              # 当前权重
                'success_count': 0,         # 成功次数
                'error_count': 0,           # 错误次数
                'last_response_time': 0,    # 最近一次响应时间
                'consecutive_errors': 0,    # 连续失败次数
                'is_circuit_open': False,   # 熔断状态（True=熔断/排除）
                'last_probe_time': 0,       # 上次探测时间
            }
        
        self._update_weights()
    
    def record_response(self, ip, response_time, success=True):
        """记录一次请求的响应时间"""
        with self.lock:
            if ip not in self.stats:
                return
            
            stat = self.stats[ip]
            
            def add_fail_time():
                if stat['ema_response_time'] > self.timeout_threshold:
                    stat['ema_response_time'] = self.timeout_threshold
                    stat['error_count'] += 1
                    stat['consecutive_errors'] += 1

                # 检查是否需要熔断
                if stat['consecutive_errors'] >= self.circuit_breaker_threshold and not stat['is_circuit_open']:
                    stat['is_circuit_open'] = True
                    stat['last_probe_time'] = time.time()
                    print(f"[熔断触发] {ip} 连续失败 {stat['consecutive_errors']} 次，已被熔断排除")

            if success:
                stat['success_count'] += 1
                stat['last_response_time'] = response_time
                stat['consecutive_errors'] = 0  # 重置连续错误计数
                
                # 如果之前是熔断状态，成功后恢复
                if stat['is_circuit_open']:
                    stat['is_circuit_open'] = False
                    print(f"[熔断恢复] {ip} 服务已恢复正常")
                
                # 使用指数移动平均更新响应时间
                stat['ema_response_time'] = (
                    self.alpha * response_time + 
                    (1 - self.alpha) * stat['ema_response_time']
                )

                add_fail_time()
            else:
                # 错误时增加惩罚，相当于响应时间翻倍
                stat['ema_response_time'] = stat['ema_response_time'] * 1.1
                
                add_fail_time()
            
            self._update_weights()
    
    def _update_weights(self):
        """根据响应时间更新权重，响应时间越短权重越高，熔断的服务权重为0"""
        if not self.stats:
            return
        
        # 只计算非熔断服务的平均响应时间
        active_stats = {ip: s for ip, s in self.stats.items() if not s['is_circuit_open']}
        
        if not active_stats:
            # 所有服务都熔断了，保持原有权重
            return
        
        response_times = [s['ema_response_time'] for s in active_stats.values()]
        avg_time = sum(response_times) / len(response_times)
        
        for ip, stat in self.stats.items():
            if stat['is_circuit_open']:
                # 熔断的服务权重设为0
                stat['weight'] = 0
            else:
                # 权重与响应时间成反比
                if stat['ema_response_time'] > 0:
                    raw_weight = avg_time / stat['ema_response_time']
                else:
                    raw_weight = 1.0
                stat['weight'] = max(self.min_weight, min(self.max_weight, raw_weight))
    
    def _should_probe(self, ip):
        """判断是否应该对熔断的服务发送探测请求"""
        stat = self.stats[ip]
        if not stat['is_circuit_open']:
            return False
        
        current_time = time.time()
        # 检查是否超过探测间隔
        if current_time - stat['last_probe_time'] >= self.probe_interval:
            # 按概率决定是否探测
            if random.random() < self.probe_ratio:
                stat['last_probe_time'] = current_time
                return True
        return False
    
    def select_ip(self, ip_list):
        """根据权重选择IP，权重越高被选中概率越大，支持探测熔断服务"""
        with self.lock:
            # 首先检查是否需要探测某个熔断的服务
            for i, ip in enumerate(ip_list):
                if self._should_probe(ip):
                    print(f"[探测请求] 向熔断服务 {ip} 发送探测请求")
                    return i, ip
            
            # 获取活跃服务列表（非熔断）
            active_ips = [(i, ip) for i, ip in enumerate(ip_list) if not self.stats[ip]['is_circuit_open']]
            
            if not active_ips:
                # 所有服务都熔断了，随机选择一个尝试
                print("[警告] 所有服务都已熔断，随机选择一个尝试")
                idx = random.randint(0, len(ip_list) - 1)
                return idx, ip_list[idx]
            
            # 从活跃服务中按权重选择
            weights = [self.stats[ip]['weight'] for _, ip in active_ips]
            total_weight = sum(weights)
            
            if total_weight == 0:
                # 权重全为0，随机选择
                chosen = random.choice(active_ips)
                return chosen
            
            # 加权随机选择
            r = random.random() * total_weight
            cumulative = 0
            for idx, ip in active_ips:
                cumulative += self.stats[ip]['weight']
                if r <= cumulative:
                    return idx, ip
            
            # 兜底返回最后一个活跃服务
            return active_ips[-1]
    
    def get_active_count(self):
        """获取活跃（非熔断）服务数量"""
        with self.lock:
            return sum(1 for s in self.stats.values() if not s['is_circuit_open'])
    
    def get_circuit_open_ips(self):
        """获取所有熔断的IP列表"""
        with self.lock:
            return [ip for ip, s in self.stats.items() if s['is_circuit_open']]
    
    def force_recover(self, ip):
        """强制恢复某个熔断的服务"""
        with self.lock:
            if ip in self.stats:
                self.stats[ip]['is_circuit_open'] = False
                self.stats[ip]['consecutive_errors'] = 0
                self._update_weights()
                print(f"[强制恢复] {ip} 已被手动恢复")
    
    def get_stats(self):
        """获取所有IP的统计信息"""
        with self.lock:
            return {
                ip: {
                    'ema_response_time': f"{stat['ema_response_time']:.3f}s",
                    'weight': f"{stat['weight']:.3f}",
                    'success': stat['success_count'],
                    'error': stat['error_count'],
                    'last_rt': f"{stat['last_response_time']:.3f}s",
                    'consecutive_errors': stat['consecutive_errors'],
                    'circuit_open': stat['is_circuit_open'],
                }
                for ip, stat in self.stats.items()
            }


def encode_image(image_path, img_tnx=None):
    try:
        image_format = 'jpg'
        if img_tnx is not None:
            file_content = img_tnx.get(image_path.encode('utf-8'))
            imghdr.what(None, h=file_content)
        elif isinstance(image_path, dict):
            patch_file_manager = get_default_patch_manager()
            img_pil, img_bytes = patch_file_manager.read_image(image_path['patch'], image_path['start_num'], image_path['size'])
            # 将PIL Image转换为bytes
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG')
            file_content = buffer.getvalue()
            image_format = 'jpeg'
        elif image_path.startswith('http'):
            file_content = requests.get(image_path).content
            image_format = imghdr.what(None, h=file_content)
        else:
            with open(image_path, "rb") as image_file:
                file_content = image_file.read()
            with Image.open(image_path) as img:
                image_format = img.format

        base64_data = base64.b64encode(file_content).decode('utf-8')
        # 如果长度不是4的倍数，添加必要的填充
        padding = 4 - (len(base64_data) % 4) if len(base64_data) % 4 != 0 else 0
        base64_data += "=" * padding

        # image_format = imghdr.what(file_content)
        return f"data:image/{image_format.lower()};base64,{base64_data}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


class TaijiGPTAPI:
    def __init__(self, model_name, api_key, model_marker, *args, max_try=3, **kwargs):
        self.url= "http://trpc-utools-prod.turbotke.production.polaris:8009/"
        self.model_name = model_name
        self.json_data = {
            "bid": "open_api_test",
            "server": "open_api",
            "services": [],
            "request_id": "1234",
            "session_id": "12345",  
            "api_key": api_key, # change here
            "model_marker": model_marker,
            "system": "", # 模型人设
            "params": {},
            "timeout": 300, # 超时时间,单位秒
            "model_name": model_name,
            "use_openai_format": 1
        }
        self.max_try = max_try
        if len(args) % 2 == 0:  # 确保是成对的
            for i in range(0, len(args), 2):
                self.json_data[args[i]] = args[i+1]
        else:
            raise ValueError("*args must contain key-value pairs")
        
        # 处理**kwargs - 直接更新字典
        self.json_data.update(kwargs)


    def __call__(self, img_path, question, system_prompt=None):
        if img_path is not None and len(img_path) > 0:
            img_txt_prompt = encode_image(img_path)
            if img_txt_prompt is None:
                return None
        
        messages = [{
            "role": "user", 
            "content": [
                {"type": "text", "value": question},
                {"type": "image_url", "value": img_txt_prompt}
            ]
        }]
   
        data_json = copy.deepcopy(self.json_data)
        data_json["messages"] = messages
        if system_prompt is not None:
            data_json["system"] = system_prompt

        response = None
        num_request = 0
        while response is None:
            original_resp = requests.post(url=self.url, json=data_json)
            try:
                response = original_resp.json()['answer'][0]['value']
            except Exception as e:
                print("ERROR")
                print(json.loads(original_resp.text))
                print(img_path)
                response = None
                if 'InvalidParameter.UnsupportedImageFormat' in original_resp.text \
                    or 'Image dimensions are too small' in original_resp.text \
                    or 'Maximum allowed: 36000000 pixels' in original_resp.text:
                    num_request = self.max_try + 1
            num_request += 1
            if num_request > self.max_try:
                print(f"Exceed maximum request num.")
                break 
        return response


def get_simple_auth(source, SecretId, SecretKey):
    dateTime = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, dateTime


class DistillationAPI:
    def __init__(self, model_name, api_key, model_marker, *args, max_try=3, user=None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.model_marker = model_marker
        self.user = user
        self.base_url = "http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval"
        self.timeout = 300

        self.model_name = model_name
        self.json_data = {
            "request_id": "1234",
            "model_marker": model_marker,
            "system": "", # 模型人设
            "params": {},
            "timeout": self.timeout, # 超时时间,单位秒
        }
        self.max_try = max_try
        if len(args) % 2 == 0:  # 确保是成对的
            for i in range(0, len(args), 2):
                self.json_data[args[i]] = args[i+1]
        else:
            raise ValueError("*args must contain key-value pairs")
        
        # 处理**kwargs - 直接更新字典
        self.json_data.update(kwargs)
        
    def get_header(self):
        API_VERSION = "v2.03"
        source = 'xxxxxx'  # 签名水印值，可填写任意值
        sign, dateTime = get_simple_auth(source, self.user, self.api_key)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def __call__(self, img_path, question, system_prompt=None, txn=None):
        if img_path is not None and len(img_path) > 0:
            img_txt_prompt = encode_image(img_path)
            if img_txt_prompt is None:
                return None
        
        messages = [{
            "role": "user", 
            "content": [
                {"type": "text", "value": question},
                {"type": "image_url", "value": img_txt_prompt}
            ]
        }]
        data_json = copy.deepcopy(self.json_data)
        data_json["messages"] = messages
        data_json["request_id"] = str(uuid.uuid4())
        if system_prompt is not None:
            data_json["system"] = system_prompt
   
        response = None
        num_request = 0
        while response is None:
            headers = dict(self.get_header())
            original_resp = requests.post(url=self.base_url, headers=headers, json=data_json, timeout=self.timeout)
            try:
                response_data = original_resp.json()
                if len(response_data['answer']) == 2:
                    think = response_data['answer'][0]['value']
                    answer = response_data['answer'][1]['value']
                    response = {
                        "think": think,
                        "answer": answer
                    }
                else:
                    response = response_data['answer'][0]['value']
            except Exception as e:
                print("ERROR")
                print(json.loads(original_resp.text))
                print(img_path)
                response = None
                if 'InvalidParameter.UnsupportedImageFormat' in original_resp.text \
                    or 'Image dimensions are too small' in original_resp.text \
                    or 'Maximum allowed: 36000000 pixels' in original_resp.text:
                    num_request = self.max_try + 1
            num_request += 1
            if num_request > self.max_try:
                print(f"Exceed maximum request num.")
                break 
        return response
    
    
class VLLMModel:
    def __init__(self, IP_LIST_MAP, api_key='hyocr_20260105', EXTRA_PARAMS={}, max_try=3, print_interval=60, 
                 adaptive_lb=True, lb_alpha=0.3, circuit_breaker_threshold=10,
                 probe_interval=30, probe_ratio=0.02):
        """
        Args:
            IP_LIST_MAP: IP到模型路径的映射
            EXTRA_PARAMS: 额外参数
            max_try: 最大重试次数
            print_interval: 打印并发统计的间隔时间（秒），设为0或None禁用
            adaptive_lb: 是否启用自适应负载均衡（基于响应时间）
            lb_alpha: 负载均衡器的平滑系数，越大对最新响应时间越敏感
            circuit_breaker_threshold: 连续失败次数阈值，超过后熔断该服务
            probe_interval: 熔断后探测间隔（秒），每隔多久发送一次探测请求
            probe_ratio: 探测请求比例，对熔断服务发送请求的概率
            timeout_threshold: 超时阈值（秒），响应超过这个时间则视为失败，会被EXTRA_PARAMS中的timeout覆盖
        """
        self.max_try = max_try
        self.ip_model_pairs = [(k, v) for k, v in IP_LIST_MAP.items()]
        self.api_key = api_key
        self.ip_list = list(IP_LIST_MAP.keys())
        self.request_count = ThreadSafeCounter()
        self.concurrency_tracker = ConcurrencyTracker()
        self.adaptive_lb = adaptive_lb
        if 'timeout' in EXTRA_PARAMS:
            self.timeout_threshold = EXTRA_PARAMS['timeout']
        else:
            self.timeout_threshold = 120.0
        
        # 初始化自适应负载均衡器
        if adaptive_lb:
            self.load_balancer = AdaptiveLoadBalancer(
                self.ip_list, 
                alpha=lb_alpha,
                circuit_breaker_threshold=circuit_breaker_threshold,
                probe_interval=probe_interval,
                probe_ratio=probe_ratio,
                timeout_threshold=self.timeout_threshold
            )
        else:
            self.load_balancer = None
        
        self.clients = {}
        for ip, model_info in IP_LIST_MAP.items():
            client = OpenAI(
                base_url=f"http://{ip}/v1",
                api_key=model_info['api_key'],
                timeout=self.timeout_threshold,
            )
            self.clients[ip] = (client, model_info['model_path'])
        self.EXTRA_PARAMS = EXTRA_PARAMS
        
        # 启动并发统计打印线程
        self._stop_monitor = threading.Event()
        self._monitor_thread = None
        if print_interval and print_interval > 0:
            self._start_monitor(print_interval)
    
    def _start_monitor(self, interval):
        """启动并发监控线程"""
        def monitor_loop():
            while not self._stop_monitor.is_set():
                self._stop_monitor.wait(interval)
                stats = self.concurrency_tracker.get_stats()
                total = sum(stats.values())
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] 并发统计 - 总并发: {total}")
                for ip, count in stats.items():
                    print(f"  {ip}: 并发={count}")
                
                # 打印负载均衡统计
                if self.adaptive_lb and self.load_balancer:
                    lb_stats = self.load_balancer.get_stats()
                    active_count = self.load_balancer.get_active_count()
                    circuit_open_ips = self.load_balancer.get_circuit_open_ips()
                    
                    # 收集所有统计信息后一次性输出
                    output_lines = []
                    output_lines.append(f"[{timestamp}] 负载均衡统计 (活跃服务: {active_count}/{len(lb_stats)}):")
                    if circuit_open_ips:
                        output_lines.append(f"  [熔断服务]: {', '.join(circuit_open_ips)}")
                    for ip, stat in lb_stats.items():
                        status = "[熔断]" if stat['circuit_open'] else "[正常]"
                        # 获取服务端真实并发信息
                        server_metrics = get_vllm_server_metrics(ip)
                        if server_metrics:
                            server_info = f"服务端[运行={server_metrics['running']}, 等待={server_metrics['waiting']}]"
                        else:
                            server_info = "服务端[获取失败]"
                        output_lines.append(f"  {ip} {status}: {server_info}, 本地[权重={stat['weight']}, EMA响应时间={stat['ema_response_time']}, "
                              f"成功={stat['success']}, 错误={stat['error']}, 连续错误={stat['consecutive_errors']}]")
                    
                    # 一次性输出所有统计信息
                    print('\n'.join(output_lines))
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitor(self):
        """停止并发监控线程"""
        if self._monitor_thread:
            self._stop_monitor.set()
            self._monitor_thread.join(timeout=2)

    def update_ip_list(self, new_ip_list_map):
        """
        更新 IP 列表和 client，保留已有 IP 的统计数据
        Args:
            new_ip_list_map: 新的 IP 到模型路径的映射
        """
        new_ip_list = list(new_ip_list_map.keys())
        
        # 更新 clients，复用已有的 client
        new_clients = {}
        for ip, model_info in new_ip_list_map.items():
            if ip in self.clients:
                # 复用已有 client
                new_clients[ip] = self.clients[ip]
            else:
                # 创建新 client
                client = OpenAI(
                    base_url=f"http://{ip}/v1",
                    api_key=model_info['api_key'],
                    timeout=self.timeout_threshold,
                )
                new_clients[ip] = (client, model_info['model_path'])
        
        # 更新负载均衡器的统计信息
        if self.adaptive_lb and self.load_balancer:
            with self.load_balancer.lock:
                old_stats = self.load_balancer.stats
                new_stats = {}
                for ip in new_ip_list:
                    if ip in old_stats:
                        # 保留已有 IP 的统计
                        new_stats[ip] = old_stats[ip]
                    else:
                        # 新 IP 使用默认统计
                        new_stats[ip] = {
                            'ema_response_time': self.timeout_threshold,
                            'weight': 1.0,
                            'success_count': 0,
                            'error_count': 0,
                            'last_response_time': 0,
                            'consecutive_errors': 0,
                            'is_circuit_open': False,
                            'last_probe_time': 0,
                        }
                self.load_balancer.stats = new_stats
                self.load_balancer._update_weights()
        
        # 原子性更新
        self.ip_model_pairs = [(k, v) for k, v in new_ip_list_map.items()]
        self.ip_list = new_ip_list
        self.clients = new_clients
        
        print(f"[IP更新] 已更新 IP 列表，当前共 {len(new_ip_list)} 个服务")

    def _select_client(self):
        """选择客户端，支持自适应负载均衡"""
        if self.adaptive_lb and self.load_balancer:
            idx, ip = self.load_balancer.select_ip(self.ip_list)
        else:
            # 轮询选择
            idx = self.request_count.increment() % len(self.ip_list)
            ip = self.ip_list[idx]
        
        client, model_name = self.clients[ip]
        return client, model_name, ip

    def openapi_call(self, messages, img_tnx=None):
        response = None
        num_request = 0

        encoded_messages = []
        for conv in messages:
            new_conv = {
                'role': conv['role'],
                'content': []
            }
            for content in conv['content']:
                if content['type'] == 'image_path':
                    image_url = encode_image(content['image_path'], img_tnx)
                    if image_url is None:
                        return None
                    new_conv['content'].append({'type': 'image_url', 'image_url': {'url': image_url}})
                else:
                    new_conv['content'].append(content)
            encoded_messages.append(new_conv)

        while response is None and num_request < self.max_try:
            ip = None
            start_time = time.time()
            try:
                client, model_name, ip = self._select_client()
                self.concurrency_tracker.increment(ip)

                completion = client.chat.completions.create(
                    model=model_name,  
                    messages=encoded_messages,
                    **self.EXTRA_PARAMS
                )
                
                response = completion.choices[0].message.content
                
                # 记录成功的响应时间
                if self.adaptive_lb and self.load_balancer:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=True)
                
            except Exception as e:
                print(f"Error in API call {ip}: {str(e)}")
                # 记录失败
                if self.adaptive_lb and self.load_balancer and ip:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=False)
                num_request += 1
                if num_request >= self.max_try:
                    return ""
                time.sleep(1)
            finally:
                if ip:
                    self.concurrency_tracker.decrement(ip)
                
        return response


    def __call__(self, img_path_or_list=None, question='', img_tnx=None, image_first=False, system_prompt=None) -> str:
        if question == '':
            raise ValueError("Question cannot be empty")
        
        """调用vllm API，支持多张图片；支持 system prompt"""
        # 兼容传入 None，str，或 list 的 img_path_or_list
        if img_path_or_list is None or len(img_path_or_list) == 0:
            image_urls = []
        elif isinstance(img_path_or_list, (list, tuple)):
            # 多张图片，批量encode
            image_urls = []
            for single_img in img_path_or_list:
                encoded = encode_image(single_img, img_tnx)
                if encoded is None:
                    return None
                image_urls.append(encoded)
        elif isinstance(img_path_or_list, str):
            # 单张图片(str)
            encoded = encode_image(img_path_or_list, img_tnx)
            if encoded is None:
                return None
            image_urls = [encoded]
        else:
            raise TypeError("img_path_or_list error")

        response = None
        num_request = 0
        while response is None and num_request < self.max_try:
            ip = None
            start_time = time.time()
            try:
                client, model_name, ip = self._select_client()
                self.concurrency_tracker.increment(ip)
                
                # 构造消息格式，支持多张图片和 system prompt
                messages = []
                if system_prompt is not None and system_prompt != "":
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    })
                user_msg = {
                    "role": "user",
                    "content": []
                }
                if not image_urls:  # 没有图片
                    user_msg["content"].append({"type": "text", "text": question})
                else:
                    if image_first:
                        # 先图片，后文本
                        for img_url in image_urls:
                            user_msg['content'].append({"type": "image_url", "image_url": {"url": img_url}})
                        user_msg['content'].append({"type": "text", "text": question})
                    else:
                        # 先文本，后图片
                        user_msg['content'].append({"type": "text", "text": question})
                        for img_url in image_urls:
                            user_msg['content'].append({"type": "image_url", "image_url": {"url": img_url}})
                messages.append(user_msg)
                # print(self.EXTRA_PARAMS)
                completion = client.chat.completions.create(
                    model=model_name,  
                    messages=messages,
                    **self.EXTRA_PARAMS
                )
                
                response = completion.choices[0].message.content
                
                # 记录成功的响应时间
                if self.adaptive_lb and self.load_balancer:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=True)
                
            except Exception as e:
                print(f"Error in API call {ip}: {str(e)}")
                # 记录失败
                if self.adaptive_lb and self.load_balancer and ip:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=False)
                num_request += 1
                if num_request >= self.max_try:
                    return ""
                time.sleep(1)
            finally:
                if ip:
                    self.concurrency_tracker.decrement(ip)
                
        return response


class VLLMEmbeddingModel(VLLMModel):
    """
    继承自VLLMModel的Embedding模型类，复用负载均衡、熔断等功能
    专门用于处理embedding请求
    """
    
    def __init__(self, IP_LIST_MAP, api_key='hyocr_20260105', EXTRA_PARAMS={}, max_try=3, print_interval=60, 
                 adaptive_lb=True, lb_alpha=0.3, circuit_breaker_threshold=10,
                 probe_interval=30, probe_ratio=0.02):
        """
        初始化embedding模型，参数与VLLMModel相同
        """
        super().__init__(IP_LIST_MAP, api_key, EXTRA_PARAMS, max_try, print_interval,
                        adaptive_lb, lb_alpha, circuit_breaker_threshold, probe_interval, probe_ratio)
    
    # 暂时只支持 Qwen3VL-Embedding
    def __call__(self, img_path=None, question=None, img_tnx=None) -> str:
        if not img_path and not question:
            raise ValueError("img_path and question cannot be both empty")

        default_instruction = "Represent the user's input."

        image_url = None
        if img_path is not None and len(img_path) > 0:
            image_url = encode_image(img_path, img_tnx)
            if image_url is None:
                return None
        
        response = None
        num_request = 0
        while response is None and num_request < self.max_try:
            ip = None
            start_time = time.time()
            try:
                client, model_name, ip = self._select_client()
                self.concurrency_tracker.increment(ip)

                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": default_instruction},
                        ],
                    },
                ]
                user_content = []
                if image_url is not None:
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
                user_content.append({"type": "text", "text": question if question else ""})
                messages.append({
                    "role": "user",
                    "content": user_content,
                })
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                })

                # if random.random() > 0.99:
                #     # 打印messages，但不要打印image_url
                #     message_to_print = copy.deepcopy(messages)
                #     for message in message_to_print:
                #         for content in message["content"]:
                #             if content["type"] == "image_url":
                #                 content["image_url"]["url"] = img_path
                #     print(message_to_print)

                response = client.post(
                    "/embeddings",
                    cast_to=CreateEmbeddingResponse,
                    body={
                        "messages": messages,
                        "model": model_name,
                        "encoding_format": "float",
                        "add_special_tokens": True,
                    },
                )
                
                response = response.data[0].embedding
                
                # 记录成功的响应时间
                if self.adaptive_lb and self.load_balancer:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=True)
                
            except Exception as e:
                response = None
                print(f"Error in API call {ip}: {str(e)}")
                # 记录失败
                if self.adaptive_lb and self.load_balancer and ip:
                    elapsed = time.time() - start_time
                    self.load_balancer.record_response(ip, elapsed, success=False)
                num_request += 1
                if num_request >= self.max_try:
                    return None
                time.sleep(1)
            finally:
                if ip:
                    self.concurrency_tracker.decrement(ip)
                
        return response
