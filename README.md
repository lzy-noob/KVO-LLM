# KVO-LLM: Boosting Long-Context Generation Throughput for Batched LLM Inference (DAC'25)

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