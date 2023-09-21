import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer
import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='trt_engines')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="./model",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'qwen_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    tokenizer_dir: str = None,
    num_beams: int = 1,
):
    tensorrt_llm.logger.set_level(log_level)

    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    #tokenizer = QWenTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True)
    
    input_ids = torch.IntTensor(tokenizer.encode(
        input_text)).cuda().unsqueeze(0)
    model_config = ModelConfig(num_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding)
    sampling_config = SamplingConfig(end_id=151643,
                                     pad_id=151643,
                                     num_beams=num_beams)
    input_lengths = torch.tensor(
        [input_ids.size(1) for _ in range(input_ids.size(0))]).int().cuda()
    engine_name = get_engine_name('qwen', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

   

    if remove_input_padding:
        decoder.setup(1, torch.max(input_lengths).item(), max_output_len)
    else:
        decoder.setup(input_ids.size(0), input_ids.size(1), max_output_len)

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    output_ids = output_ids.tolist()[0][0][input_ids.size(1):]
    output_text = tokenizer.decode(output_ids)
    print(f'Input: {args.input_text}')
    print(f'Output: {output_text}')


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
