import unittest

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Parameter, Tensor
from tensorrt_llm._utils import torch_to_numpy

class Swiglu(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
     
    def forward(self, x):
        a, gate = x.chunk(2,dim=-1)
        out = a * torch.nn.functional.silu(gate)
        return out

class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')
        torch.manual_seed(42)

    #@parameterized.expand([['float16'], ['float32'], ['bfloat16']])
    @parameterized.expand([['float32']])
    def test_swiglu_plugin(self, dtype):
        # test data
        hidden_size = 22016
        x_data = torch.randn((1, 1, hidden_size),
                             dtype=torch.float64,
                             device="cuda")
        m = Swiglu()
  
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        m.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))
        x_data = x_data.to(tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_swiglu_plugin(dtype)
        
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
          
            output = tensorrt_llm.functional.swiglu(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16'),
                                bf16=(dtype == 'bfloat16')))
        assert build_engine is not None, "Build engine failed"
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu()})

        # compare diff
        dtype_atol = {"float16": 8e-3, "float32": 2e-6, "bfloat16": 8e-2}
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'].to(torch.float32),
                                   atol=dtype_atol[dtype])
if __name__ == "__main__":
    unittest.main()