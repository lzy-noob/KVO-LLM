import os
import sys
# ---------
# pls use your own path
# ---------
sys.path.append('/home/lizy/Desktop/workspace/LLM2024/KVO-LLM')
os.chdir('/home/lizy/Desktop/workspace/LLM2024/KVO-LLM')
from main_llm import *

# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

config_param = {
    # ---------
    # GPU wait and occupy func, the source file is in llm_eval/multigpu
    # ---------
    "gpu_wait":             1,
    # ---------
    # basic param
    # - benchmark
    # - custom_flag: result file write tag
    # - model
    # - tasks: longbench tasks
    # - longbench_e
    # ---------
    "benchmark":            "longbench",
    "custom_flag":          "quant_only",
    "model":                "llama2-7b-chat-4k",
    # "model":                "vicuna-v1.5-7b-16k", 
    # "model":                "longchat-v1.5-7b-32k",  
    "tasks":                "triviaqa",
    "longbench_e":          False,
    # ---------
    # KV cache compress cfg
    # - kv_compress_en
    # - kv_compress_method
    # - kv_start_budget: kv budget in paper
    # - kv_recent_budget: recent token that will not be compressed
    # - window_diff_select: using attention map to select base token enable
    # ---------
    "kv_compress_en":       1,
    "kv_compress_method":   "kvo",
    "kv_start_budget":      384,
    "kv_recent_budget":     64,
    "window_diff_select":   1, 
    # ---------
    # quant_cfg
    # - bits
    # - group_size: if =32, 32 elements will share the same quantization parameters
    # - sym: symmetric quant enable
    # - clip_ratio
    # - chunk_size: if =8, 8 tokens will form a group 
    # ---------
    "kbits":                2,
    "kgroup_size":          32,
    "ksym":                 0,
    "kclip_ratio":          1,
    "kchunk_size":          8,   
    "vbits":                2,
    "vgroup_size":          32,
    "vsym":                 0,
    "vclip_ratio":          1,
    "vchunk_size":          8,
    # ---------
    # prune_cfg
    # - prune_en
    # - prune_budget: prune ratio
    # ---------
    "kprune_en":            1,
    "kprune_budget":        0.25,
    "vprune_en":            1,
    "vprune_budget":        0.5,
}

if __name__ == '__main__':

    main_func(config_param)
