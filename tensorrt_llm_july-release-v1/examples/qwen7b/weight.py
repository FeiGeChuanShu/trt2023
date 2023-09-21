import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def load_from_hf_qwen(tensorrt_llm_qwen,
                       hf_qwen,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32"):
    tensorrt_llm.logger.info('Loading weights from HF QWen-7B...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2

    # # Do we use SmoothQuant?
    # use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # # Do we use quantization per token?
    # quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # # Do we use quantization per channel?
    # quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()
    
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()
  
    model_params = dict(hf_qwen.named_parameters())
    #for swiglu to fuse w1 and w2 weight to gate_up_proj
    for l in range(hf_qwen.config.num_hidden_layers):
        prefix = f'transformer.h.{l}.mlp.'
        w1_weight = model_params[prefix + 'w1.weight']
        w2_weight = model_params[prefix + 'w2.weight']
        assert w1_weight.shape[0] == w2_weight.shape[0]
        gate_up_proj_weight = torch.cat([w1_weight, w2_weight], dim=0)
        model_params[prefix + 'w1.weight'] = gate_up_proj_weight

    torch_dtype = str_dtype_to_torch(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = torch_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None
    
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'transformer.wte.weight' in k:
            tensorrt_llm_qwen.vocab_embedding.weight.value = v
        elif 'transformer.ln_f.weight' in k:
            tensorrt_llm_qwen.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_qwen.num_layers:
                continue
            if 'ln_1.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].input_layernorm.weight
                dst.value = v
            elif 'ln_2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'attn.c_attn.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.weight
                q_emb = v.shape[0] // 3
                model_emb = v.shape[1]
                v = v.reshape(3, q_emb, model_emb)
                split_v = split(v, tensor_parallel, rank, dim=1)
                split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                                            model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.qkv.bias
                dst.value = np.ascontiguousarray(v)
            elif 'attn.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attention.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w1.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.gate_up_proj.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[
                        idx].mlp.gate_up_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            if use_int8_kv_cache:
                t = fromfile(
                    './trt_engines/qwen/7B/trt_engines/int8_kv_cache/1-gpu/', 'model.layers.' + str(idx) +
                    '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                    np.float32)

                assert t is not None
                tensorrt_llm_qwen.layers[
                    idx].attention.kv_quantization_scale.value = 1.0 / t
                tensorrt_llm_qwen.layers[idx].attention.kv_dequantization_scale.value = t

            # elif 'mlp.w1.weight' in k:
            #     dst = tensorrt_llm_qwen.layers[idx].mlp.gate.weight
            #     split_v = split(v, tensor_parallel, rank, dim=0)
            #     if use_weight_only:
            #         v = np.ascontiguousarray(split_v.transpose())
            #         processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            #             torch.tensor(v), plugin_weight_only_quant_type)
            #         # workaround for trt not supporting int8 inputs in plugins currently
            #         dst.value = processed_torch_weights.view(
            #             dtype=torch.float32).numpy()
            #         scales = tensorrt_llm_qwen.layers[
            #             idx].mlp.gate.per_channel_scale
            #         scales.value = torch_weight_scales.numpy()
            #     else:
            #         dst.value = np.ascontiguousarray(split_v)
            # elif 'mlp.w2.weight' in k:
            #     dst = tensorrt_llm_qwen.layers[idx].mlp.fc.weight
            #     split_v = split(v, tensor_parallel, rank, dim=0)
            #     if use_weight_only:
            #         v = np.ascontiguousarray(split_v.transpose())
            #         processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            #             torch.tensor(v), plugin_weight_only_quant_type)
            #         # workaround for trt not supporting int8 inputs in plugins currently
            #         dst.value = processed_torch_weights.view(
            #             dtype=torch.float32).numpy()
            #         scales = tensorrt_llm_qwen.layers[
            #             idx].mlp.fc.per_channel_scale
            #         scales.value = torch_weight_scales.numpy()
            #     else:
            #         dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return

