import os
import pdb
import copy
from typing import Optional, Dict
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    OPTForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from kvcache.kvo import *
from longbench.dataset_model import *

ENABLE_KVCACHE_LM = {
    "llama2-7b-chat-4k": {
        'kvo':LlamaForCausalLMKVO,
    },
    "longchat-v1.5-7b-32k": {
        'kvo':LlamaForCausalLMKVO,
    },
    "vicuna-v1.5-7b-16k": {
        'kvo':LlamaForCausalLMKVO,
    },
}
ENABLE_KVCACHE_FUNCTIONS = {
    "llama2-7b-chat-4k": {
        'kvo':convert_kvcache_llama_kvo,
    },
    "longchat-v1.5-7b-32k": {
        'kvo':convert_kvcache_llama_kvo,
    },
    "vicuna-v1.5-7b-16k": {
        'kvo':convert_kvcache_llama_kvo,
    },
}
KVCACHE_TARGET_MODULE = {
    "llama2-7b-chat-4k": {
        'kvo':LlamaAttention_kvo,
    },
    "longchat-v1.5-7b-32k": {
        'kvo':convert_kvcache_llama_kvo,
    },
    "vicuna-v1.5-7b-16k": {
        'kvo':convert_kvcache_llama_kvo,
    },
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "vicuna-v1.5-7b-16k",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)

    return parser.parse_args(args)

def load_model_and_tokenizer(path, model_name, kvcache: Optional[dict]):
    if "llama2" in model_name:
        config = AutoConfig.from_pretrained(path,)
        config._flash_attn_2_enabled = False
        tokenizer = LlamaTokenizer.from_pretrained(path, )
        if kvcache['enable']:
            print('Enable KVCache Eviction or Merging')
            config.kvcache = {
                'start_budget':         kvcache['start_budget'],
                'recent_budget':        kvcache['recent_budget'],
                'window_diff_select':   kvcache['window_diff_select'],
                'key_quant':            kvcache['key_quant'],
                'value_quant':          kvcache['value_quant'],
            }
            CausalLM = ENABLE_KVCACHE_LM[model_name][kvcache['method']]
        else:
            CausalLM = LlamaForCausalLM
        model = CausalLM.from_pretrained(
            path, config=config, torch_dtype=torch.float16, device_map='auto',
        )
        # quant weight
        if kvcache['enable'] and (kvcache['method'] == 'qdfkv'):
            model.quant_weight()

    elif "longchat" in model_name or "vicuna" in model_name:
        config = AutoConfig.from_pretrained(path)
        config._flash_attn_2_enabled = False
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
        if kvcache['enable']:
            print('Enable KVCache Eviction or Merging')
            config.kvcache = {
                'start_budget':         kvcache['start_budget'],
                'recent_budget':        kvcache['recent_budget'],
                'alpha':                kvcache['alpha'],
                'window_diff_select':   kvcache['window_diff_select'],
                'key_quant':            kvcache['key_quant'],
                'value_quant':          kvcache['value_quant'],
            }
            CausalLM = ENABLE_KVCACHE_LM[model_name][kvcache['method']]
        else:
            CausalLM = AutoModelForCausalLM
        model = CausalLM.from_pretrained(
            path, trust_remote_code=True, config=config, torch_dtype=torch.float16, device_map='auto'
        )
    else:
        assert False, "Do not support this model type!"

    model = model.eval()

    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2-7b-chat-4k" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model, tokenizer, data, max_length, max_gen, prompt_format, dataset, model_name, kvcache: Optional[dict],
):

    preds   = []
    for json_obj in tqdm(data, desc=f"{dataset} eval"):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        

        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to('cuda')
            else:
                input = prompt.to('cuda')
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )

                past_key_values = output.past_key_values
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]

                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    pred_token_idx = (
                        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    )
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break
                
        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
        # kvcache mask reset
        if kvcache['enable']:
            for name, m in model.named_modules():
                if isinstance(m, KVCACHE_TARGET_MODULE[model_name][kvcache['method']]):
                    m._reset_masks()

    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def predict(model_name: str,
            datasets: list,
            e_enable: bool,
            kvcache: Optional[dict],
            longbench_path: str,
            result_path: str,):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define your model
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], model_name, kvcache
    )

    max_length = model2maxlen[model_name]
    for dataset in datasets:
        print()
        print(f"*******************************")
        print(f"{dataset} evaluation begins !!!")
        print(f"*******************************")
        if e_enable:
            # pls use your own path to define the cache_dir
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test", cache_dir="/home/lizy/Desktop/workspace/LLM2024/Private/dataset")
        else:
            # pls use your own path to define the cache_dir
            data = load_dataset("THUDM/LongBench", dataset, split="test", cache_dir="/home/lizy/Desktop/workspace/LLM2024/Private/dataset")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            kvcache,
        )
        if e_enable:
            if not os.path.exists(os.path.join(result_path, f"pred_e/")):
                os.makedirs(os.path.join(result_path, f"pred_e/"))
            out_path = os.path.join(result_path, f"pred_e/{dataset}.jsonl")
        else:
            if not os.path.exists(os.path.join(result_path, f"pred/")):
                os.makedirs(os.path.join(result_path, f"pred/"))
            out_path = os.path.join(result_path, f"pred/{dataset}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
