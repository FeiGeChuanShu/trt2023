# -*- coding: utf-8 -*-
# Please pull the latest code to run the profiling.
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

checkpoint_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, resume_download=True)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    trust_remote_code=True,
    resume_download=True).half().cuda().eval()

model.generation_config = GenerationConfig.from_pretrained(
    checkpoint_path, trust_remote_code=True, resume_download=True,
)

model.generation_config.min_length = 513
model.generation_config.max_new_tokens = 512

max_experiment_times = 10
time_costs = []
context_str = 'Born in north-east France, Soyer trained as a'
max_gpu_memory_cost = 0
for _ in tqdm(range(max_experiment_times)):
    inputs = tokenizer(context_str, return_tensors='pt')
    inputs = inputs.to(model.device)
    t1 = time.time()
    pred = model.generate(**inputs)
    time_costs.append(time.time() - t1)
    #assert pred.shape[1] == model.generation_config.min_length
    max_gpu_memory_cost = max(max_gpu_memory_cost, torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()

print("Average generate speed (tokens/s): {}".format((max_experiment_times * 512) / sum(time_costs)))
print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
print("Experiment setting: ")
print(f"max_experiment_times = {max_experiment_times}")
