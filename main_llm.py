import os
import pdb
import json
import argparse
import time
from prettytable import PrettyTable

# from huggingface_hub import login
# login()

# * custom packages
import utils
from config import *
from llm_eval.utils import build_model_and_enc, build_opt
from llm_eval.multigpu import pool_request
from longbench.pred import predict as longbench_pred
from longbench.eval import evaluate as longbench_eval

def longbench(result_path: str, task_list: list, config_param:dict):

    pred_args = {
        'model_name'    : config_param['model'],
        'datasets'      : task_list,
        'e_enable'      : config_param['longbench_e'],
        'kvcache'       : {
            "enable":               config_param['kv_compress_en'],
            "method":               config_param['kv_compress_method'],
            "start_budget":         config_param['kv_start_budget'],
            "recent_budget":        config_param['kv_recent_budget'],
            "window_diff_select":   config_param['window_diff_select'],
            # quant
            # -key
            "key_quant": {
                "quant_grain":      "key",
                "n_bits":           config_param['kbits'],
                "symmetric":        config_param['ksym'],
                "group_size":       config_param['kgroup_size'],
                "clip_ratio":       config_param['kclip_ratio'],
                "chunk_size":       config_param['kchunk_size'],
                "prune_en":         config_param['kprune_en'],
                "prune_budget":     config_param['kprune_budget'],
            },
            # -value
            "value_quant":{
                "quant_grain":      "value",
                "n_bits":           config_param['vbits'],
                "symmetric":        config_param['vsym'],
                "group_size":       config_param['vgroup_size'],
                "clip_ratio":       config_param['vclip_ratio'],
                "chunk_size":       config_param['vchunk_size'],
                "prune_en":         config_param['vprune_en'],
                "prune_budget":     config_param['vprune_budget'],
            },

        },
        'longbench_path': LONGBENCH_PATH,
        'result_path'   : result_path,
    }
    # * prediction and evaluation
    longbench_pred(**pred_args)
    return longbench_eval(
        model_name=config_param['model'],
        e_enable=config_param['longbench_e'],
        result_path=result_path
    )


def main_func(config_param):

    # * print config
    for arg in (config_param):
        print(format(arg, '<20'), format(str(config_param[arg]), '<'), flush=True)
    # * GPU utilize
    if config_param['gpu_wait']:
        pool_request(120, 10000)
        print("You Got It!")
    # * check the models, benchmarks, kvcache methods
    # benchmarks
    if config_param['benchmark'] not in TASKS.keys():
        NotImplementedError(f"{config_param['benchmark']} has not been supported on this platform's version yet!!")
    # tasks
    task_list = []
    if config_param['tasks'] in TASKS[config_param['benchmark']].keys():
        task_list = TASKS[config_param['benchmark']][config_param['tasks']]
    else:
        task_list = config_param['tasks'].split(",")
        for task_ in task_list:
            if task_ not in TASKS[config_param['benchmark']]["full_set"]:
                ValueError(f"{config_param['tasks']} is a illegal task in {config_param['benchmark']}!!!")

    # * log the evaluation prefix
    prefix_tb = PrettyTable()
    prefix_tb.border = True
    # header and main entries
    prefix_tb.title = "Evaluation Info"
    prefix_tb.field_names = [' ', "Entry"]
    prefix_tb.add_row(["Model", config_param['model']])
    prefix_tb.add_row(["Benchmark", config_param['benchmark']])
    if config_param['tasks'] in TASKS[config_param['benchmark']].keys():
        prefix_tb.add_row(["Tasks", ','.join(TASKS[config_param['benchmark']][config_param['tasks']])])
    else:
        prefix_tb.add_row(["Tasks", config_param['tasks']])
    if config_param['kv_compress_en']:
        prefix_tb.add_row(["KV Method", config_param['kv_compress_method']])
        if config_param['kv_start_budget'] is not None:
            prefix_tb.add_row(["KV Start Budget", config_param['kv_start_budget']])
        if config_param['kv_recent_budget'] is not None:
            prefix_tb.add_row(["KV Recent Budget", config_param['kv_recent_budget']])
    print(prefix_tb)

    # * define the result path
    sec = time.time()
    sec_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(sec))
    if config_param['custom_flag'] is not None:
        result_root_path = f"results/{config_param['benchmark']}/{config_param['custom_flag']}"
    else:
        result_root_path = f"results/{config_param['benchmark']}/"

    if config_param['kv_compress_en']:
        file_name = [
            config_param['kv_compress_method'],
            f"{config_param['kv_start_budget']}" if config_param['kv_start_budget'] is not None else "NoSetStart",
            f"{config_param['kv_recent_budget']}" if config_param['kv_recent_budget'] is not None else "NoSetRecent",
        ] 
        result_path = os.path.join(result_root_path, config_param['model'], sec_str,"_".join(file_name))
    else:
        result_path = os.path.join(result_root_path, config_param['model'], sec_str)

    # * evaluation starts
    top_eval = {
        'longbench': longbench
    }
    results = top_eval[config_param['benchmark']](
        result_path=result_path,
        task_list=task_list,
        config_param=config_param
    )

    # * logging into the terminal
    utils.print_results(
        eval_res={f"{config_param['benchmark']}": results},
        model=config_param['model'],
        custom_flag= config_param['custom_flag']
    )

    # print config
    import sys
    log_pth     = os.path.join(result_path,'config.txt')
    sys.stdout  = open(log_pth, mode='w')
    for arg in (config_param):
        print(format(arg, '<20'), format(str(config_param[arg]), '<'), flush=True)
