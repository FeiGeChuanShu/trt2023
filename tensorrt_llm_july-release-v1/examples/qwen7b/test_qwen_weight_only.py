import os
import sys
import tempfile
import unittest
from itertools import product
from time import time_ns
import numpy as np
import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.quantization import QuantMode

from model.modeling_qwen import QWenLMHeadModel
from model.configuration_qwen import QWenConfig
from weight import load_from_hf_qwen

def compare_max_abs_error(ref, res, str):
    # calculate max abs error
    compare_HF = ref.cpu().numpy().flatten()
    compare_TRT_LLM = res.cpu().numpy().flatten()
    max_abs_error = np.max(abs(compare_TRT_LLM - compare_HF))
    print(str, "max abs error = ", max_abs_error)

def accuracy_compute(actual, desired, atol = 1e-5):
    res = np.all( np.abs(actual - desired) < atol )
    abs_err_mean = np.mean(np.abs(actual - desired))
    abs_err_median = np.median(np.abs(actual - desired))
    abs_err_max = np.max(np.abs(actual - desired))
    
    rel_err_mean = np.mean(np.abs(actual - desired) / (np.abs(desired) + atol))
    rel_err_median = np.median(np.abs(actual - desired) / (np.abs(desired) + atol))
    rel_err_max = np.max(np.abs(actual - desired) / (np.abs(desired) + atol))
    return {'total':      res,
            'abs_mean':   abs_err_mean,
            'abs_median': abs_err_median,
            'abs_max':    abs_err_max,
            'rel_mean':   rel_err_mean,
            'rel_median': rel_err_median,
            'rel_max':    rel_err_max
            }

class TestLLaMA(unittest.TestCase):
    def _gen_hf_qwen(self, dtype):
        qwen_config = QWenConfig()
        qwen_config.num_hidden_layers = 2
        qwen_config.seq_length = 2048
        qwen_config.fp16 = True


        hf_qwen = QWenLMHeadModel(qwen_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return qwen_config, hf_qwen
    
    def _gen_tensorrt_llm_network(self, network, hf_qwen,
                                  qwen_config: QWenConfig, batch_size,
                                  beam_width, input_len, output_len, dtype,
                                  rank, tensor_parallel):
        tensor_parallel_group = list(range(tensor_parallel))
        #rotary_dim = int((qwen_config.hidden_size // qwen_config.num_attention_heads) * qwen_config.rotary_pct)
        rotary_dim = int(qwen_config.kv_channels * qwen_config.rotary_pct)
        with net_guard(network):
            kv_dtype = str_dtype_to_trt(dtype)

            # Initialize model
            tensorrt_llm_qwen = tensorrt_llm.models.QWenLMHeadModel(
                num_layers=qwen_config.num_hidden_layers,
                num_heads=qwen_config.num_attention_heads,
                hidden_size=qwen_config.hidden_size,
                vocab_size=qwen_config.vocab_size,
                hidden_act=qwen_config.hidden_act,
                max_position_embeddings=qwen_config.max_position_embeddings,
                dtype=kv_dtype,
                use_dynamic_ntk=qwen_config.use_dynamic_ntk,
                use_logn_attn=qwen_config.use_logn_attn,
                mlp_hidden_size=qwen_config.intermediate_size//2,
                neox_rotary_style=True,
                rotary_dim = rotary_dim,
                tensor_parallel=tensor_parallel,  # TP only
                tensor_parallel_group=tensor_parallel_group  # TP only
            )
            tensorrt_llm_qwen = weight_only_quantize(tensorrt_llm_qwen,
                                                  QuantMode.use_weight_only())
            load_from_hf_qwen(tensorrt_llm_qwen,
                               hf_qwen,
                               dtype=dtype,
                               rank=rank,
                               tensor_parallel=tensor_parallel)

            with net_guard(network):
                # Prepare
                network.set_named_parameters(tensorrt_llm_qwen.named_parameters())
                inputs = tensorrt_llm_qwen.prepare_inputs(batch_size, input_len,
                                                        output_len, True,
                                                        beam_width)
                # Forward
                tensorrt_llm_qwen(*inputs)

                # mark as TRT network output
                # ----------------------------------------------------------------
                for k, v in tensorrt_llm_qwen.named_network_outputs():
                    print(k)
                    network._mark_output(v, k,
                                        tensorrt_llm.str_dtype_to_trt('float16'))
        

        return network

    def _gen_tensorrt_llm_engine(self,
                                 dtype,
                                 rank,
                                 world_size,
                                 qwen_config,
                                 hf_qwen,
                                 model_name,
                                 use_plugin,
                                 batch_size,
                                 beam_width,
                                 input_len,
                                 output_len,
                                 use_refit,
                                 fast_building=False,
                                 context_fmha_flag=ContextFMHAType.disabled,
                                 enable_remove_input_padding=False):

        builder = Builder()

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
                network.plugin_config.set_rmsnorm_plugin(dtype='float32')
            if fast_building:
                network.plugin_config.set_gemm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            network.plugin_config.set_weight_only_quant_matmul_plugin(dtype='float16')
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, hf_qwen, qwen_config,
                                           batch_size, beam_width, input_len,
                                           output_len, dtype, 
                                           rank, world_size)

            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                int8=QuantMode.use_weight_only().has_act_and_weight_quant()
            )

           
            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  qwen_config,
                                  hf_qwen,
                                  model_name,
                                  use_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping(world_size, rank)
        engine_buffer = self._gen_tensorrt_llm_engine(
            dtype, rank, world_size, qwen_config, hf_qwen, model_name,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    def load_test_cases():
        test_cases = list(
            product([False], [True], [ContextFMHAType.disabled], [False], ['float16']))
        # test_cases = list(
        #     product([False], [False, True], [
        #         ContextFMHAType.disabled, ContextFMHAType.enabled,
        #         ContextFMHAType.enabled_with_fp32_acc
        #     ], [False, True], ['float16'], [False]))
        #test_cases.append(
        #    (False, True, ContextFMHAType.disabled, False, 'bfloat16', False))
        # test_cases.append((False, True, ContextFMHAType.enabled, False,
        #                    'float16', False))  # needs transformers==4.31.0
        return test_cases

    @parameterized.expand(load_test_cases)
    def test_qwen(self, use_refit, fast_building, context_fmha_flag,
                   enable_remove_input_padding, dtype):
        model = 'qwen'
        log_level = 'error'
        use_plugin = True  # gpt plugin
        batch_size = 4
        beam_width = 1
        input_len = 128
        output_len = 128
        max_seq_len = input_len + output_len
        world_size = 1
        rank = 0
        
        qwen_config, hf_qwen = self._gen_hf_qwen(dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, qwen_config, hf_qwen, model,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding)
        key_value_cache_buffers = []
        head_size = qwen_config.hidden_size // qwen_config.num_attention_heads
        for i in range(qwen_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    qwen_config.num_attention_heads,
                    max_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
        sequence_length_buffer = torch.ones((batch_size, ),
                                            dtype=torch.int32,
                                            device='cuda')

        # compare context
        step = 0
        ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
        ctx_input_lengths = input_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_masked_tokens = torch.zeros((batch_size, input_len),
                                        dtype=torch.int32,
                                        device='cuda')
        ctx_position_ids = torch.IntTensor(range(input_len)).reshape(
            [1, input_len]).expand([batch_size, input_len]).cuda()
        ctx_last_token_ids = ctx_input_lengths.clone()
        ctx_max_input_length = torch.zeros((input_len, ),
                                           dtype=torch.int32,
                                           device='cuda')

        with torch.no_grad():
            hf_outputs = hf_qwen.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([1, batch_size * input_len])
            ctx_position_ids = ctx_position_ids.view(
                [1, batch_size * input_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        ctx_shape = {
            'input_ids': ctx_ids.shape,
            'input_lengths': ctx_input_lengths.shape,
            'masked_tokens': ctx_masked_tokens.shape,
            'position_ids': ctx_position_ids.shape,
            'last_token_ids': ctx_last_token_ids.shape,
            'max_input_length': ctx_max_input_length.shape,
            'cache_indirection': cache_indirections[0].shape,
        }
        ctx_buffer = {
            'input_ids': ctx_ids,
            'input_lengths': ctx_input_lengths,
            'masked_tokens': ctx_masked_tokens,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'max_input_length': ctx_max_input_length,
            'cache_indirection': cache_indirections[0],
        }
        kv_shape = (batch_size, 2, qwen_config.num_key_value_heads,
                    max_seq_len, qwen_config.hidden_size //
                    qwen_config.num_attention_heads)
        for i in range(qwen_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = kv_shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer * (input_len +
                                                                  step)
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_shape['past_key_value_length'] = (2, )
        ctx_buffer['past_key_value_length'] = torch.tensor([0, 1],
                                                           dtype=torch.int32)

        context = runtime.context_0
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        #warmup
        for n in range(50):
            runtime._run(context)

        torch.cuda.synchronize()
        t0 = time_ns()
        for n in range(10):
            runtime._run(context)
        torch.cuda.synchronize()
        t1 = time_ns()
        time_infer = (t1-t0)/1000/1000/10
        #print('trt_outputs: ', ctx_buffer.keys())
        #print('input_layernorm: ',ctx_buffer['layers.0.input_layernorm_output'],ctx_buffer['layers.0.input_layernorm_output'].shape)
        # print('qkv: ',ctx_buffer['layers.0.attention.qkv_output'],ctx_buffer['layers.0.attention.qkv_output'].shape)
        #print('attention: ',ctx_buffer['layers.0.attention_output'],ctx_buffer['layers.0.attention_output'].shape)
        # print('post_layernorm: ',ctx_buffer['layers.0.post_layernorm_output'],ctx_buffer['layers.0.post_layernorm_output'].shape)
        # print('mlp: ',ctx_buffer['layers.0.mlp_output'],ctx_buffer['layers.0.mlp_output'].shape)
        #print('hidden_states: ',ctx_buffer['layers.0.hidden_states'],ctx_buffer['layers.0.hidden_states'].shape)
        #print('hidden_states: ',ctx_buffer['layers.3.hidden_states'],ctx_buffer['layers.3.hidden_states'].shape)
        #print('before_lnf: ',ctx_buffer['before_ln_f_output'],ctx_buffer['before_ln_f_output'].shape)
        #print('lnf: ',ctx_buffer['ln_f_output'],ctx_buffer['ln_f_output'].shape)
        

        res = ctx_buffer['logits']

        acc_result = accuracy_compute(res.to(torch.float32).cpu().numpy(), 
                                      ref.to(torch.float32).cpu().numpy(), atol = 5e-2)
        print("context logits")
        print("Batch_size: {}\nLatency: {:.3f}\nThroughput: {:.3f}\n"
                "Absolute_mean: {:<.5f}\nAbsolute_median: {:.5f}\nAbsolute_max: {:.5f}\n"
                "Relative_mean: {:<.5f}\nRelative_median: {:.5f}\nRelative_max: {:.5f}\n".format(
                    1,
                    time_infer,
                    1/time_infer*1000,
                    acc_result['abs_mean'],
                    acc_result['abs_median'],
                    acc_result['abs_max'],
                    acc_result['rel_mean'],
                    acc_result['rel_median'],
                    acc_result['rel_max']))

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=5e-1)
  
        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_input_lengths = ctx_input_lengths.clone()
        gen_max_input_length = ctx_max_input_length.clone()
        gen_masked_tokens = torch.zeros((batch_size, max_seq_len),
                                        dtype=torch.int32,
                                        device='cuda')
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * input_len
        gen_last_token_ids = torch.zeros_like(gen_input_lengths).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_qwen.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            step1_id = step1_id.view([1, batch_size])
            gen_position_ids = gen_position_ids.view([1, batch_size])
            gen_last_token_ids = torch.ones_like(gen_input_lengths).int().cuda()
            gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()

        step1_shape = {
            'input_ids': step1_id.shape,
            'input_lengths': gen_input_lengths.shape,
            'masked_tokens': gen_masked_tokens.shape,
            'position_ids': gen_position_ids.shape,
            'last_token_ids': gen_last_token_ids.shape,
            'max_input_length': gen_max_input_length.shape,
            'cache_indirection': cache_indirections[1].shape,
        }
        step1_buffer = {
            'input_ids': step1_id,
            'input_lengths': gen_input_lengths,
            'masked_tokens': gen_masked_tokens,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'max_input_length': gen_max_input_length.contiguous(),
            'cache_indirection': cache_indirections[1],
        }
        for i in range(qwen_config.num_hidden_layers):
            step1_shape[f'past_key_value_{i}'] = kv_shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['past_key_value_length'] = (2, )
        for i in range(qwen_config.num_hidden_layers):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        step1_buffer['sequence_length'] = sequence_length_buffer * (input_len +
                                                                    step)
        step1_buffer['past_key_value_length'] = torch.tensor(
            [input_len + step - 1, 0], dtype=torch.int32)

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=5e-1)
        print("generation logits")
        acc_result = accuracy_compute(res.to(torch.float32).cpu().numpy(), 
                                      ref.to(torch.float32).cpu().numpy(), atol = 5e-2)
        print("Batch_size: {}\nLatency: {:.3f}\nThroughput: {:.3f}\n"
                "Absolute_mean: {:<.5f}\nAbsolute_median: {:.5f}\nAbsolute_max: {:.5f}\n"
                "Relative_mean: {:<.5f}\nRelative_median: {:.5f}\nRelative_max: {:.5f}\n".format(
                    1,
                    time_infer,
                    1/time_infer*1000,
                    acc_result['abs_mean'],
                    acc_result['abs_median'],
                    acc_result['abs_max'],
                    acc_result['rel_mean'],
                    acc_result['rel_median'],
                    acc_result['rel_max']))

if __name__ == '__main__':
    unittest.main()
