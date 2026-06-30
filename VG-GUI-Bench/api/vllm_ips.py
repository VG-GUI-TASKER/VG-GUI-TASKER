import os

# ========== 版本校验 ==========
# 当前配置版本号（每次大版本变动时更新）
__version__ = "1.0.0"

def _check_version():
    """
    检查环境变量中的版本是否符合要求。
    """
    required_version = os.environ.get("VLLM_IPS_VERSION")
    
    if required_version is None or required_version != __version__:
        raise ImportError(
            f"vllm_ips.py 版本不匹配！\n"
            f"  请联系 @harveyshen 获取最新版本代码"
        )

# 模块导入时自动执行版本检查
_check_version()

# ========== IP 配置 ==========

qwen35_name_to_ip = {
    'ocr_polaris': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-397B-A17B-FP8-eth_AIDE',
        'model_path': 'GUI-Qwen3.5-397B-A17B-FP8-eth',
        'api_key': 'no_key',
        'split': 'ocr',
    },
    'gui_video_polaris': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-397B-A17B-FP8-H20_AIDE',
        'model_path': 'GUI-Qwen3.5-397B-A17B-FP8-H20',
        'api_key': 'no_key',
        'split': 'gui_video',
    },
    'gui_polaris_128_batch_size': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-397B-A17B-FP8-H20_harveyshen_128concurrency_AIDE',
        'model_path': 'GUI-Qwen3.5-397B-A17B-FP8-H20_harveyshen_128concurrency',
        'api_key': 'no_key',
        'split': 'gui',
    },
    'gui_polaris_128_batch_size_v2': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-397B-A17B-FP8-H20_harveyshen_128concurrency_v2_AIDE',
        'model_path': 'GUI-Qwen3.5-397B-A17B-FP8-H20_harveyshen_128concurrency_v2',
        'api_key': 'no_key',
        'split': 'default',
    },
    'gui_polaris_share': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-397B-A17B-FP8-H20_share_AIDE',
        'model_path': 'GUI-Qwen3.5-397B-A17B-FP8-H20_share',
        'api_key': 'no_key',
        'split': 'default',
    },
    'test_polaris': {
        'type': 'polaris',
        'polaris_name': 'qwen35_a17b_gy2_AIDE',
        'model_path': 'qwen35_a17b_gy2',
        'api_key': 'no_key',
        'split': 'test',
    },
    'test_polaris_2': {
        'type': 'polaris',
        'polaris_name': 'qwen35_a17b_gy3_AIDE',
        'model_path': 'qwen35_a17b_gy3',
        'api_key': 'no_key',
        'split': 'test',
    },
}


qwen35_base_name_to_ip = {
    'ocr_polaris': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-35B-A3B-Base-eth_AIDE',
        'model_path': 'GUI-Qwen3.5-35B-A3B-Base-eth',
        'api_key': 'no_key',
        'split': 'ocr',
    },
}

qwen35_A3B_name_to_ip = {
    'ocr_polaris': {
        'type': 'polaris',
        'polaris_name': 'GUI-Qwen3.5-35B-A3B-eth_AIDE',
        'model_path': 'GUI-Qwen3.5-35B-A3B-eth',
        'api_key': 'no_key',
        'split': 'ocr',
    },
}

mimo_name_to_ip = {
    'ocr_64': {
        'ip': [
            '29.127.80.169', '29.232.228.85', '29.191.209.106', '29.119.80.246',
            '29.191.208.250', '29.232.224.155', '29.232.228.24', '29.127.69.241'
        ],
        'model_path': 'mimo2508rl'
    }
}


fara_name_to_ip = {
    'gy_8': {
        'ip': [
            '29.119.99.198'
        ],
        'model_path': 'fara',
        'split': 'default',
    }
}


gelab_zero_name_to_ip = {
    'gy_16-2': {
        'ip': [
            '29.119.99.144'
        ],
        'model_path': 'GELab-Zero-4B-preview',
        'split': 'default',
    }
}

kimi_name_to_ip = {
    'gui_8x8': {
        'ip': [
            '29.127.51.199', '29.119.98.153', '29.232.243.132', '29.232.224.211'
        ],
        'model_path': 'kimi',
        'api_key': 'no_key',
        'split': 'default',
    },
    'gui_8x8_2': {
        'ip': [
            '29.127.80.17', '29.119.80.59', '29.232.242.172', '29.119.97.44'
        ],
        'model_path': 'kimi',
        'api_key': 'no_key',
        'split': 'default',
    },
    'gui_8x8_3': {
        'ip': [
            '29.127.80.153', '29.232.241.73', '29.232.240.60', '29.232.243.240'
        ],
        'model_path': 'kimi',
        'api_key': 'no_key',
        'split': 'default',
    },
}


glm_name_to_ip = {
    'gui_sh_4x16': {
        'ip': [
            '11.238.34.100', '11.238.11.217', '11.238.28.75', '11.238.41.238', 
            '29.121.98.13', '11.238.15.182', '11.238.39.247', '11.238.38.101', 
            '11.238.18.99', '11.238.19.12', '11.238.21.194', '11.238.20.176', 
            '11.238.19.173', '11.238.19.134', '29.121.80.29', '29.121.80.30'
        ],
        'model_path': 'glm-4.6v',
        'api_key': 'hyocr_20260129',
        'split': 'sh',
    },
    'gui_sh_4x16_2': {
        'ip': [
            '11.238.34.223', '29.121.92.37', '11.238.35.82', '29.121.92.38', 
            '11.238.10.239', '11.238.10.45', '11.238.34.226', '11.238.16.110', 
            '11.238.16.20', '11.238.35.163', '11.238.33.231', '11.238.33.45', 
            '11.238.25.246', '11.238.24.31', '11.238.34.72', '11.238.39.13'
        ],
        'model_path': 'glm-4.6v',
        'api_key': 'hyocr_20260129',
        'split': 'sh',
    },
    'gui_sh_4x16_3': {
        'ip': [
            '11.238.39.231', '11.238.38.178', '11.238.11.182', '11.238.20.250', 
            '11.238.21.204', '11.238.24.96', '11.238.24.65', '11.238.19.62', 
            '11.238.19.26', '11.238.38.224', '11.238.39.31', '29.121.78.21', 
            '29.121.78.20', '11.238.34.254', '11.238.35.235', '11.238.39.192'
        ],
        'model_path': 'glm-4.6v',
        'api_key': 'hyocr_20260129',
        'split': 'sh',
    },
    'gui_sh_4x16_4': {
        'ip': [
            '11.238.31.187', '11.238.39.131', '11.238.38.22', '11.238.10.232', 
            '11.238.11.48', '11.238.24.61', '11.238.24.13', '11.238.39.78', 
            '11.238.39.123', '11.238.37.40', '11.238.36.242', '11.238.38.79', 
            '11.238.39.125', '11.238.23.132', '11.238.22.142', '11.238.30.92'
        ],
        'model_path': 'glm-4.6v',
        'api_key': 'hyocr_20260129',
        'split': 'sh',
    },
}


qwen3_5_name_to_ip = {
    'qwen35_8x2': {
        'ip': [
            '29.127.68.12', 
        ],
        'model_path': 'Qwen3.5-397B-A17B',
        'api_key': 'no_key',
        'split': 'default',
    },
}


qwen3_5_name_to_ip_2 = {
    'qwen35_8x2': {
        'ip': [
            '29.127.68.126', 
        ],
        'model_path': 'Qwen3.5-397B-A17B',
        'api_key': 'no_key',
        'split': 'default',
    },
}

qwen3vl_name_to_ip = {
    'ocr_qwen_64': {
        'ip': [
            '29.119.80.140', '29.127.82.137', '29.232.225.79', '29.191.192.230', 
            '29.232.229.72', '29.127.69.104', '29.119.80.22', '29.191.210.233', 
        ],
        'model_path': 'Qwen3-VL-235B-A22B-Instruct',
        'api_key': 'hyocr_20260105',
        'split': 'default',
    },
    'ocr_qwen_64_2': {
        'ip': [
            '29.191.192.244', '29.191.209.56', '29.119.83.229', '29.127.65.87', 
            '29.127.36.14', '29.119.80.178', '29.127.69.242', '29.191.193.247', 
        ],
        'model_path': 'Qwen3-VL-235B-A22B-Instruct',
        'api_key': 'hyocr_20260105',
        'split': 'default',
    },
    'gui_qwen_64_from_rl_train': {
        'ip': [
            '29.191.192.221', '29.191.192.235', '29.127.69.105', '29.232.229.237', 
            '29.127.48.135', '29.232.240.219', '29.232.229.171', '29.127.83.11', 
        ],
        'model_path': 'Qwen3-VL-235B-A22B-Instruct',
        'api_key': 'hyocr_20260105',
        'split': 'default',
    },
    'gui_qwen_64_from_rl_train_a800': {
        'ip': [
            '30.207.98.200', '30.207.97.117', '30.207.99.135', '30.207.96.230', 
            '30.207.97.36', '30.207.98.224', '30.207.99.142', '30.207.98.34', 
        ],
        'model_path': 'Qwen3-VL-235B-A22B-Instruct',
        'api_key': 'hyocr_20260105',
        'split': 'default',
    },

    'gui_qwen_64_from_sh': {
        'ip': [
            '30.207.98.83', '30.207.97.244', '30.207.99.78', '30.207.99.74', 
            '30.207.99.227', '30.207.97.27', '30.207.99.251', '30.207.97.156', 
        ],
        'model_path': 'Qwen3-VL-235B-A22B-Instruct',
        'api_key': 'hyocr_20260105',
        'split': 'default',
    },
}

mai_ui_name_to_ip = {
    # 'qwen_1x16': {
    #     'ip': [
    #         '29.174.238.140', '29.86.146.168', '29.246.126.117', '29.238.6.98', 
    #         '29.251.212.224', '29.175.110.128', '29.175.110.125', '29.175.108.34', 
    #         '29.175.108.36', '29.175.108.31', '29.175.108.30', '29.246.112.187', 
    #         '29.246.112.189', '29.246.112.185', '29.246.112.78', '29.174.238.137',
    #     ],
    #     'model_path': 'MAI-UI-8B',
    #     'api_key': 'hyocr_20260112',
    #     'gpu_nums': 1,
    #     'split': 'default',
    # },
}

qwen3vl_4b_name_to_ip = {
    'gui_sh_qwen_4b_1x32': {
        'ip': [
            '11.238.26.66', '11.238.39.144', '29.121.46.11', '11.238.11.183', 
            '11.238.11.194', '11.238.10.109', '11.238.10.223', '11.238.24.241', 
            '11.238.25.148', '11.238.24.92', '11.238.25.41', '11.238.24.22', 
            '11.238.25.20', '11.238.24.169', '11.238.24.162', '11.238.21.192', 
            '11.238.21.67', '11.238.27.230', '11.238.20.241', '11.238.21.214', 
            '11.238.27.64', '11.238.27.40', '11.238.10.140', '11.238.20.13', 
            '11.238.27.129', '11.238.27.214', '11.238.20.120', '11.238.11.177', 
            '11.238.21.142', '11.238.26.131', '11.238.20.239', '11.238.27.195'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
    'gui_sh_qwen_4b_1x16': {
        'ip': [
            '11.238.29.31', '11.238.11.207', '11.238.11.112', '11.238.11.239', 
            '11.238.11.175', '11.238.11.21', '11.238.10.23', '29.121.90.33', 
            '29.121.90.40', '29.121.90.35', '29.121.90.34', '29.121.90.38', 
            '29.121.90.36', '29.121.90.39', '29.121.90.37', '11.238.29.143'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
    'gui_sh_qwen_4b_1x16_2': {
        'ip': [
            '11.238.18.50', '11.238.39.138', '11.238.10.141', '11.238.27.87', 
            '11.238.27.82', '11.238.14.155', '11.238.14.65', '11.238.25.64', 
            '11.238.25.205', '11.238.24.108', '11.238.25.174', '11.238.25.226', 
            '11.238.24.139', '11.238.25.87', '11.238.24.147', '11.238.19.83'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
    'gui_sh_qwen_4b_1x16_3': {
        'ip': [
            '11.238.10.131', '11.238.23.167', '11.238.22.160', '11.238.23.15', 
            '11.238.22.84', '11.238.23.93', '11.238.18.75', '11.238.19.114', 
            '11.238.19.90', '11.238.19.115', '11.238.18.65', '11.238.19.32', 
            '11.238.10.225', '11.238.10.150', '11.238.10.210', '11.238.10.91'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
    'gui_sh_qwen_4b_1x16_4': {
        'ip': [
            '11.238.27.212', '11.238.37.53', '11.238.37.185', '11.238.10.159', 
            '11.238.10.134', '11.238.10.27', '11.238.23.223', '11.238.22.42', 
            '11.238.22.30', '11.238.26.242', '11.238.27.110', '11.238.27.115', 
            '11.238.27.96', '11.238.26.74', '11.238.26.241', '11.238.27.207'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
    'gui_sh_qwen_4b_1x16_5': {
        'ip': [
            '11.238.27.227', '11.238.24.14', '11.238.25.71', '11.238.25.69', 
            '11.238.24.247', '11.238.25.201', '11.238.24.44', '11.238.25.235', 
            '11.238.25.145', '11.238.27.81', '11.238.27.42', '11.238.26.123', 
            '11.238.26.115', '11.238.26.43', '11.238.26.216', '11.238.27.165'
        ],
        'model_path': 'Qwen3-VL-4B-Instruct',
        'api_key': 'hyocr_20260112',
        'gpu_nums': 1,
        'split': 'sh',
    },
}

qwen3vl_8b_name_to_ip = {
    # 'gy_qwen_24': {
    #     'ip': [
    #         '29.232.242.254', '29.191.209.140',
    #     ],
    #     'model_path': 'Qwen3-VL-8B-Instruct',
    #     'api_key': 'hyocr_20260108',
    # },
}

qwen3vl_thinking_name_to_ip = {
    # 'gui_train': {
    #     'ip': [
    #         '29.119.98.212', '29.127.36.186', '29.127.81.252', '29.127.32.212', 
    #     ],
    #     'model_path': 'Qwen3-VL-235B-A22B-Thinking'
    # },
}

ui_venus_ground_to_ip = {
    # 'gui_venus_64': {
    #     'ip': [
    #         '29.232.229.30', '29.127.48.9', '29.119.84.249', '29.119.96.143', 
    #         '29.127.80.53', '29.232.228.246', '29.127.65.86', '29.119.83.220', 
    #     ],
    #     'model_path': 'ui_venus_ground',
    #     'split': 'default',
    # },
}

ui_venus_1_5_name_to_ip = {
    'qwen_2x32': {
        'ip': [
            '29.120.100.129', '29.238.4.175', '29.238.88.137', '29.238.88.21', 
            '29.86.139.77', '29.86.138.241', '29.228.150.15', '29.228.150.18', 
            '29.238.246.36', '29.228.134.152', '29.246.74.206', '29.246.74.202', 
            '29.246.74.204', '29.246.74.205', '29.86.139.131', '29.86.138.25', 
            '29.228.42.126', '29.228.42.131', '29.175.60.126', '29.120.40.141', 
            '29.228.208.87', '29.175.60.186', '29.246.10.125', '29.246.178.223', 
            '29.174.238.73', '29.238.50.197', '29.246.46.172', '29.246.46.93', 
            '29.228.150.150', '29.228.150.14', '29.246.178.172', '29.238.246.34', 
        ],
        'model_path': 'UI-Venus-1.5-30B-A3B',
        'api_key': 'hyocr_20260112',
        'split': 'default',
    },
    'qwen_2x32_2': {
        'ip': [
            '29.238.22.175', '29.228.42.133', '29.246.178.224', '29.174.238.170', 
            '29.174.238.74', '29.174.238.77', '29.251.212.208', '29.251.212.212', 
            '29.251.212.213', '29.228.158.179', '29.251.212.210', '29.238.16.75', 
            '29.238.16.66', '29.238.16.71', '29.238.16.73', '29.228.208.142', 
            '29.228.208.144', '29.228.208.145', '29.228.208.146', '29.238.234.193', 
            '29.238.234.121', '29.238.234.190', '29.238.234.194', '29.228.166.24', 
            '29.228.166.22', '29.228.166.25', '29.228.166.27', '29.228.158.176', 
            '29.246.46.98', '29.228.158.178', '29.228.158.127', '29.238.22.176', 
        ],
        'model_path': 'UI-Venus-1.5-30B-A3B',
        'api_key': 'hyocr_20260112',
        'split': 'default',
    },

    'qwen_2x8_from_sh': {
        'ip': [
            '11.238.40.28', '11.238.26.19', '11.238.27.54', '11.238.27.190', 
            '11.238.27.160', '11.238.41.140', '11.238.41.64', '11.238.41.63', 
        ],
        'model_path': 'UI-Venus-1.5-30B-A3B',
        'api_key': 'hyocr_20260112',
        'split': 'default',
    },
}

qwen3vl_embedding_2b_name_to_ip = {
    # 'embedding_1': {
    #     'ip': [
    #         '29.246.0.53', '29.246.136.72', '29.238.54.150', '29.228.162.92', 
    #         '29.86.133.113', '29.175.94.4', '29.175.36.167', '29.174.174.120'
    #     ],
    #     'model_path': 'qwen3-vl-embedding-2b',
    #     'api_key': 'hyocr_20260105',
    #     'split': 'default',
    # },
}

qwen3_text_name_to_ip = {
}

qwen3_text_thinking_name_to_ip = {
    # 'gengluo_tj_64': {
    #     'ip': [
    #         '',
    #     ],
    #     'model_path': 'Qwen3-235B-A22B-Thinking-2507'
    # },
}

dots_name_to_ip = {
    # 'gl_32': {
    #     'ip': [
    #         '', 
    #     ],
    #     'model_path': 'model'
    # },
}


if __name__ == '__main__':
    name_to_ip_list = qwen3vl_name_to_ip

    with open('qwen3vl.yaml', 'w') as writer:

        def print_and_write(text):
            print(text)
            writer.write(text + '\n')

        print_and_write('qwen3vl_name_to_ip:')
        for name in name_to_ip_list:
            indent = 2
            print_and_write(f'{" " * indent}{name}:')
            indent += 2
            print_and_write(f'{" " * indent}ips:')
            indent += 2
            for single_ip in name_to_ip_list[name]['ip']:
                print_and_write(f'{" " * indent}- \'{single_ip}:22003\'')
            print_and_write(f'{" " * indent}model_path: Qwen3-VL-235B-A22B-Instruct')
            print_and_write(f'{" " * indent}api_key: hyocr_20260105')
            print_and_write(f'{" " * indent}gpu_nums: 1')
    
    name_to_ip_list = mai_ui_name_to_ip
    with open('mai_ui.yaml', 'w') as writer:
        def print_and_write(text):
            print(text)
            writer.write(text + '\n')
        
        print_and_write('mai_ui_name_to_ip:')
        for name in name_to_ip_list:
            indent = 2
            print_and_write(f'{" " * indent}{name}:')
            indent += 2
            print_and_write(f'{" " * indent}ips:')
            indent += 2
            for single_ip in name_to_ip_list[name]['ip']:
                print_and_write(f'{" " * indent}- \'{single_ip}:19982\'')
            print_and_write(f'{" " * indent}model_path: MAI-UI-8B')
            print_and_write(f'{" " * indent}api_key: hyocr_20260112')
            print_and_write(f'{" " * indent}gpu_nums: 1')
