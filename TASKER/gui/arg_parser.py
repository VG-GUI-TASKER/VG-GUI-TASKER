import argparse

def parse_args():
    parser = argparse.ArgumentParser("TreeVideoAgent")

    # ================= Model endpoint (OpenAI-compatible) =================
    # Any argument left as None falls back to the OPENAI_MODEL / OPENAI_API_KEY /
    # OPENAI_BASE_URL environment variables.
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key. Defaults to the OPENAI_API_KEY env var.')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.')
    parser.add_argument('--conf_lower', type=int, default=3, help='Confidence threshold (1-3) to stop searching')
    
    # ================= Original repository arguments (kept for compatibility) =================
    # data
    parser.add_argument("--dataset", default='egoschema_subset', type=str)
    parser.add_argument("--cap_path", default='data/egoschema/lavila_subset.json', type=str) 
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 

    # LLM 
    parser.add_argument("--model_name", default="gpt-4o-2024-11-20", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)   
    parser.add_argument("--cache_path", default="cache/cache_gpt4o.pkl", type=str)
    parser.add_argument("--use_cache", action='store_false', help="Whether to use llm cache")

    # output / logger
    parser.add_argument("--output_base_path", default="results/egoschema_subset/", type=str)  
    parser.add_argument("--logger_base_path", default="results/egoschema_subset/", type=str)

    # iteration
    parser.add_argument("--final_step", default=5, type=int)  
    parser.add_argument("--init_interval", default=4, type=int, help='initial number of frames (uniformly sampled)')  
    
    # agent ensemble during serach process 
    parser.add_argument("--s_conf_lower", default=3, type=int)
    parser.add_argument("--r_conf_lower", default=3, type=int)
    parser.add_argument("--ans_mode", default="vote_conf_and", choices=["s", "r", "sr", "rs", "vote_conf_and", "vote_conf_or"], type=str)
    
    # post process
    parser.add_argument("--post_s_conf_lower", default=1, type=int)
    parser.add_argument("--post_r_conf_lower", default=2, type=int)
    parser.add_argument("--post_ans_mode", default="vote", choices=["s", "r", "sr", "rs", "vote", "vote_conf_and", "vote_conf_or"], type=str)
    parser.add_argument("--post_resume_samples", action='store_false')

    # search process
    parser.add_argument("--search_strategy", default="a_star", choices=["bfs", "gbfs", "dijkstra", "a_star"], type=str)
    parser.add_argument("--beam_size", default=5, type=int)  
    parser.add_argument("--for_seg_not_interested", default="retain", choices=["prune", "retain", "merge"], type=str)
    
    # parallel
    parser.add_argument("--max_workers", default=3, type=int)

    # specific video id processing
    parser.add_argument("--process_num", default=500, type=int)
    parser.add_argument("--specific_id", default=None, type=str)
    parser.add_argument("--specific_id_path", default=None, type=str)
    parser.add_argument("--avoid_id", default=None, type=str)
    parser.add_argument("--reprocess_log", default=None, type=str)

    # in-context learning examples
    parser.add_argument("--example_summary_path", default="data/egoschema/example_summary.txt", type=str)
    parser.add_argument("--example_qa_by_summary_path", default="data/egoschema/example_qa_by_summary.txt", type=str)

    # minimum expansion steps before allowing early stop
    parser.add_argument("--min_steps", default=3, type=int,
                        help='Minimum number of expansion steps before confidence check can stop search.')

    # data sharding for parallel execution (1-10, 0=all)
    parser.add_argument("--shard", default=0, type=int,
                        help='Shard index 1-10 for parallel execution. 0 means process all data.')

    # ================= Path configuration (previously hard-coded in main.py) =================
    parser.add_argument("--video_dir", type=str, required=True,
                        help='Directory containing input video files (e.g. ./MONDAY/ytb_video_test)')
    parser.add_argument("--json_dir", type=str, required=True,
                        help='Path to the task JSON file (e.g. ./MONDAY/ours_data_test.json)')
    parser.add_argument("--out_root", type=str, required=True,
                        help='Output directory for extracted keyframe images')
    parser.add_argument("--cache_dir", type=str, default="/tmp/tasker_frame_cache",
                        help='Temp cache directory for extracted frames')
    parser.add_argument("--record_json_path", type=str, required=True,
                        help='Path to save final selected frame records JSON')

    return parser.parse_args()