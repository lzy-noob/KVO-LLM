# KVO-LLM: Boosting Long-Context Generation Throughput for Batched LLM Inference (DAC'25)
## Abstract
With the widespread deployment of long-context large language models (LLMs), efficient and high-quality generation is becoming increasingly important.
Modern LLMs employ batching and key-value (KV) cache to improve generation throughput and quality. However, as the context length and batch size rise drastically, the KV cache incurs extreme external memory access (EMA) issues. Recent LLM accelerators face substantial processing element (PE) under-utilization due to the low arithmetic intensity of attention with KV cache, while existing KV cache compression algorithms struggle with hardware inefficiency or significant accuracy degradation. To address these issues, an algorithm-architecture co-optimization, KVO-LLM, is proposed for long-context batched LLM generation. At the algorithm level, we propose a KV cache quantization-aware pruning method that first adopts salient-token-aware quantization and then prunes KV channels and tokens by attention guided pruning based on salient tokens identified during quantization.  Achieving substantial savings on hardware overhead, our algorithm reduces the EMA of KV cache over 91\% with significant accuracy advantages  compared to previous KV cache compression algorithms. At the architecture level, we propose a multi-core jointly optimized accelerator that adopts operator fusion and cross-batch interleaving strategy, maximizing PE and DRAM bandwidth utilization. Compared to the state-of-the-art LLM accelerators, KVO-LLM improves generation throughput by up to 7.32x, and attains 5.52~8.38x better energy efficiency.

## Directory Structure
- kvcache: our algorithm implementation
  - kvquant: quantization method for kv cache
  - normalquant: quantization method for weight
- llm_eval: basic function such as build a model
- longbench: longbench dataset pred and eval function
- scripts: a basic script to run our code

## Run our code
```
# 1. create a conda env 
conda create -n kvo_llm python=3.9.19

# 2. install required package 
pip install -r requirements.txt
pip install prettytable

# 3. change the cache dir in our code
# 3.1 change the model2path in longbench/dataset_model
# 3.2 change the dataset cache_dir in longbench/pred line 289 and line 292

# 4. run our code
cd scripts
python main.py
```

The results might be slightly different depending on the GPU.