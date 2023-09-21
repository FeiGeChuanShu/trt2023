'''
Convert huggingface Qwen model. Use https://huggingface.co/Qwen as demo.
'''
import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "qwen"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=1)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="qwen",
            type=str,
            help="Specify QWEN variants to convert checkpoints correctly",
            choices=["qwen"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))


@torch.no_grad()
def smooth_gpt_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, GPT2Block):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.c_fc"
        smoother = smooth_gemm(module.mlp.c_fc.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


# SantaCoder separates Q projection from KV projection
def concat_qkv_weight_bias(q, hf_key, hf_model):
    kv = hf_model.state_dict()[hf_key.replace("q_attn", "kv_attn")]
    return torch.cat([q, kv], dim=-1)


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = ["c_attn", "c_proj", "c_fc"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param

def hf_to_ft_name(orig_name):
    global_weights = {
        "transformer.wte.weight": "model.wte.weight",
        "transformer.ln_f.bias": "model.final_layernorm.bias",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "lm_head.weight": "model.lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.ln_1.bias": "input_layernorm.bias",
        "transformer.ln_1.weight": "input_layernorm.weight",
        "transformer.attn.c_attn.bias": "attention.query_key_value.bias",
        "transformer.attn.c_attn.weight": "attention.query_key_value.weight",
        "transformer.attn.c_proj.bias": "attention.dense.bias",
        "transformer.attn.c_proj.weight": "attention.dense.weight",
        "transformer.ln_2.bias": "post_attention_layernorm.bias",
        "transformer.ln_2.weight": "post_attention_layernorm.weight",
        "transformer.mlp.w1.bias": "mlp.w1.bias",
        "transformer.mlp.w1.weight": "mlp.w1.weight",
        "transformer.mlp.w2.bias": "mlp.w2.bias",
        "transformer.mlp.w2.weight": "mlp.w2.weight",
        "transformer.mlp.c_proj.bias": "mlp.c_proj.bias",
        "transformer.mlp.c_proj.weight": "mlp.c_proj.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"

@torch.no_grad()
def hf_qwen_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(args.in_file,
                                                 device_map="auto",
                                                 trust_remote_code=True)
    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, AutoTokenizer.from_pretrained(args.in_file, trust_remote_code=True), dataset)
        if args.smoothquant is not None:
            smooth_gpt_model(model, act_range, args.smoothquant)

   
    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wte.weight", "model.final_layernorm.bias",
        "model.final_layernorm.weight", "model.lm_head.weight"
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            print("Skip %s" % name)
            continue
        ft_name = hf_to_ft_name(name)

        if ft_name in global_ft_weights:
            continue
        else:
            starmap_args.append(
                (0, saved_dir, infer_tp, ft_name, param.to(storage_type),
                 storage_type, act_range.get(name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs
                 }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_qwen_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
