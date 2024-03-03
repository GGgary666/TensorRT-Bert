#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

import pycuda.driver as cuda
import pycuda.autoinit

class TrtNetworkHelper():
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger):
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

    def set_layer_name(self, layer, name):
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    # 自己定义的，应该没问题
    def addLayerNorm(self, x, gamma, beta, layer_name=None, precision=None):
        # create your layer norm plugin
        # ln_plg_creator = self.plugin_registry.get_plugin_creator("LayerNorm", "1", "")

        # gamma_field = trt.PluginField("gamma_field", gamma, trt.PluginFieldType.FLOAT32)
        # beta_field = trt.PluginField("beta_field", beta, trt.PluginFieldType.FLOAT32)
        # field_collection = trt.PluginFieldCollection([gamma_field, beta_field])

        # plugin = ln_plg_creator.create_plugin("LayerNorm", field_collection)
        # trt_layer = self.network.add_plugin_v2([x], plugin)

        # if layer_name is None:
        #     layer_name = "Custom.LayerNorm"

        # self.layer_post_process(trt_layer, layer_name, precision)

        # x = trt_layer.get_output(0)

        pfc = trt.PluginFieldCollection([])
        LN_creator = self.plugin_registry.get_plugin_creator("LayerNorm", "1", "")
        LN_plug = LN_creator.create_plugin("LayerNorm", pfc)
        LN = self.network.add_plugin_v2([x], LN_plug)

        GAMMA = self.network.add_constant((1, 1, gamma.shape[0]), trt.Weights(np.ascontiguousarray(gamma, dtype=np.float32)))
        BETA = self.network.add_constant((1, 1, beta.shape[0]), trt.Weights(np.ascontiguousarray(beta, dtype=np.float32)))
        scale_layer = self.network.add_elementwise(LN.get_output(0), GAMMA.get_output(0), trt.ElementWiseOperation.PROD)
        trt_layer = self.network.add_elementwise(scale_layer.get_output(0), BETA.get_output(0), trt.ElementWiseOperation.SUM)

        if layer_name is None:
            layer_name = "LayerNorm"
        else:
            layer_name = "LayerNorm." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)
        return x

    # 自己定义的，没问题
    def addLinear(self, x, weight, bias, layer_name=None, precision=None):
        # add Linear

        # weight = self.addConstant(weight, layer_name="Custom.Linear.Weight")
        # weight = self.addShuffle(weight, None, (1, *weight.shape), None, "Custom.Linear.Weight_")
        # mm = self.addMatMul(x, weight, layer_name="Custom.Linear.MatMul_weight")

        # bias = self.addConstant(bias, layer_name="Custom.Linear.Bias")
        # bias = self.addShuffle(bias, None, mm.shape, None, "Custom.Linear.Bias_")
        # x = self.addAdd(mm, bias, layer_name="Custom.Linear.Add_bias")

        # W = self.network.add_constant((1, weight.shape[0], weight.shape[1]), trt.Weights(np.ascontiguousarray(weight, dtype=np.float32)))
        # B = self.network.add_constant((1, 1, bias.shape[0]), trt.Weights(np.ascontiguousarray(bias, dtype=np.float32)))
        # matmul_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, W.get_output(0), trt.MatrixOperation.NONE)
        # trt_layer = self.network.add_elementwise(matmul_layer.get_output(0), B.get_output(0), trt.ElementWiseOperation.SUM)
        # x = trt_layer.get_output(0)
        
        # if weight.shape[0] == 3072 and weight.shape[1] == 768:
        #     input_len = len(x.shape)
        #     print("((((((((((((((((((((((()))))))))))))))))))))))")
        #     print("input_len")
        #     print(input_len)
        #     print("x.shape")
        #     print(x.shape)
        #     print("weight.shape")
        #     print(weight.shape)
        #     print("bias.shape")
        #     print(bias.shape)
        #     if input_len < 3:
        #         raise RuntimeError("linaer x.shape.size must >= 3")
        #     if layer_name is None:
        #         layer_name = "nn.Linear"
            
        #     pre_reshape_dims = trt.Dims()
        #     after_reshape_dims = trt.Dims()
        #     if input_len == 3:
        #         pre_reshape_dims = (0, 0, 0, 1, 1)
        #         after_reshape_dims = (0, 0, 0)
        #     elif input_len == 4:
        #         pre_reshape_dims = (0, 0, 0, 0, 1, 1)
        #         after_reshape_dims = (0, 0, 0, 0)
        #     elif input_len == 5:
        #         pre_reshape_dims = (0, 0, 0, 0, 0, 1, 1)
        #         after_reshape_dims = (0, 0, 0, 0, 0)
        #     else:
        #         raise RuntimeError("addLinear x.shape.size > 5 not support!!!")
        #     trt_layer = self.network.add_shuffle(x)
        #     trt_layer.reshape_dims = pre_reshape_dims

        #     self.layer_post_process(trt_layer, layer_name+"_pre_reshape", precision)

        #     x = trt_layer.get_output(0)
        #     print("pre_reshape")
        #     print(x.shape)
        #     out_features = weight.shape[1]
        #     weight = trt.Weights(np.ascontiguousarray(weight))
        #     if bias is not None:
        #         bias = trt.Weights(np.ascontiguousarray(bias))
            
        #     trt_layer = self.network.add_fully_connected(x, out_features, weight, bias)
        #     self.layer_post_process(trt_layer, layer_name, precision)
        #     x = trt_layer.get_output(0)

        #     trt_layer = self.network.add_shuffle(x)
        #     trt_layer.reshape_dims = after_reshape_dims
        #     self.layer_post_process(trt_layer, layer_name+"_after_reshape", precision)
        #     x = trt_layer.get_output(0)
        #     print("after_reshape")
        #     print(x.shape)
        #     print("((((((((((((((((((((((()))))))))))))))))))))))")
        # else:
        W = self.network.add_constant((1, weight.shape[0], weight.shape[1]), trt.Weights(np.ascontiguousarray(weight, dtype=np.float32)))
        B = self.network.add_constant((1, 1, bias.shape[0]), trt.Weights(np.ascontiguousarray(bias, dtype=np.float32)))
        matmul_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, W.get_output(0), trt.MatrixOperation.NONE)
        trt_layer = self.network.add_elementwise(matmul_layer.get_output(0), B.get_output(0), trt.ElementWiseOperation.SUM)
        x = trt_layer.get_output(0)
        
        
        return x

    def addReLU(self, layer, x, layer_name=None, precision=None):
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    # 自己定义的，没问题
    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        # add softmax
        trt_layer = self.network.add_softmax(x)
        if dim == -1:
            trt_layer.axes = 1 << 3

        if layer_name is None:
            layer_name = "Custom.Softmax"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    # 自己定义的，没问题
    def addAdd(self, a, b, layer_name=None, precision=None):
        # add Add
        trt_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        if layer_name is None:
            layer_name = "Custom.Add"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    # 自己定义的，没问题
    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """scale"""
        # add scale
        scale = trt.Weights(np.ascontiguousarray([scale], dtype=np.float32))
        trt_layer = self.network.add_scale(x, trt.ScaleMode.UNIFORM, None, scale, None)
        if layer_name is None:
            layer_name = "Custom.Scale"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    # 自己定义的，没问题
    def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor:
        # add MatMul
        trt_layer = self.network.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.NONE)
        if layer_name is None:
            layer_name = "Custom.MatMul"

        self.layer_post_process(trt_layer, layer_name, precision=None)

        x = trt_layer.get_output(0)
        return x

    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        trt_layer = self.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x


class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list):
        nInput = len(inputs)

        bufferD = []
        # alloc memory
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        # for i in range(0, self.engine.num_bindings):
            # print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # warm up
        self.context.execute_v2(bufferD)

        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        for i in range(0, len(outputs)):
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
            # print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)












# # import torch
# import tensorrt as trt
# import numpy as np
# import ctypes
# import math
# import time

# from typing import Optional, Tuple

# import pycuda.driver as cuda
# import pycuda.autoinit

# class TrtNetworkHelper():
#     """TensorRT Network Definition helper for Pytorch"""
#     def __init__(self, network, plugin_registry, logger):
#         self.network = network
#         self.plugin_registry = plugin_registry
#         self.logger = logger

#         self.input_num = 0

#     def set_layer_name(self, layer, name):
#         """
#         Tool function. Set the name of trt layer or plugin and print output shapes.
#         """
#         if not layer:
#             raise RuntimeError("Could not name")

#         layer.name = str(self.network.num_layers) + "_" + name
#         for i in range(0, layer.num_outputs):
#             shape = layer.get_output(i).shape
#             self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

#         return None

#     def check_trt_layer(self, trt_layer):
#         """
#         Tool function. check trt layer,
#         """
#         if not trt_layer:
#             raise RuntimeError("add " + str(trt_layer) + " failed!")

#         for i in range(0, trt_layer.num_outputs):
#             shape = trt_layer.get_output(i).shape
#             # print(trt.volume(shape))

#             # if len(shape) is 1:
#                 # raise RuntimeError("add " + layer.name + " failed!")

#     def layer_post_process(self, trt_layer, layer_name, precision):
#         """
#         Tool function. set precision, set_layer_name and check_trt_layer
#         """
#         if precision is not None:
#             trt_layer.precision = precision

#         self.set_layer_name(trt_layer, layer_name)
#         self.check_trt_layer(trt_layer)

#     def addInput(self, name, dtype, shape):
#         if name is None:
#             name = "input" + str(self.input_num)

#         self.input_num = self.input_num + 1

#         trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
#         if not trt_input:
#             raise RuntimeError("addInput failed!")

#         self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

#         return trt_input

#     def markOutput(self, x: trt.ITensor):
#         self.network.mark_output(x)
#         self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

#     def addEmbedding(self, indices, weight, layer_name=None, precision=None):
#         constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
#         gather_layer = self.network.add_gather(constant_layer.get_output(0),
#                                                indices, axis=0)

#         if layer_name is None:
#             layer_name = "nn.Embedding"
#         else:
#             layer_name = "nn.Embedding." + layer_name

#         self.layer_post_process(gather_layer, layer_name, precision)

#         return gather_layer.get_output(0)

#     def addGELU(self, x, layer_name=None, precision=None):
#         POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
#         MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
#         SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
#         ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
#         HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
#         X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
#         X_pow_t = X_pow.get_output(0)
#         X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
#         X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
#         X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
#         X_sqrt_tensor = X_sqrt.get_output(0)
#         X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
#         X_tanh_tensor = X_tanh.get_output(0)
#         X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
#         CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
#         gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

#         if layer_name is None:
#             layer_name = "nn.GELU"
#         else:
#             layer_name = "nn.GELU." + layer_name

#         self.layer_post_process(gelu_layer, layer_name, precision)

#         return gelu_layer.get_output(0)

#     def addLayerNorm(self, x, gamma, beta, layer_name=None, precision=None):
#         # TODO: create your layer norm plugin

#         return trt_layer.get_output(0)

#     def addLinear(self, x, weight, bias, layer_name=None, precision=None):
#         # TODO: add Linear

#         return x

#     def addReLU(self, layer, x, layer_name=None, precision=None):
#         trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

#         if layer_name is None:
#             layer_name = "nn.ReLU"

#         self.layer_post_process(trt_layer, layer_name, precision)

#         x = trt_layer.get_output(0)
#         return x

#     def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
#         # TODO: add softmax
#         return x

#     ################## unary op ###################
#     def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
#         trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
#         if layer_name is None:
#             layer_name = "unary.log"
#         else:
#             layer_name = "unary.log." + layer_name

#         self.layer_post_process(trt_layer, layer_name, precision)

#         x = trt_layer.get_output(0)
#         return x

#     ################## elementwise op ###################
#     def addAdd(self, a, b, layer_name=None, precision=None):
#         # add Add
#         return x

#     # tensor and scalar op
#     def addScale(
#             self,
#             x: trt.ITensor,
#             scale: float,
#             layer_name: str = None,
#             precision: trt.DataType = None
#     ) -> trt.ITensor:
#         """scale"""
#         # TODOL add scale

#         return x

#     def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor:
#         # add MatMul
#         return x


#     def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
#         trt_layer = self.network.add_constant(w.shape, w)

#         if layer_name is None:
#             layer_name = "trt.Constant"
#         else:
#             layer_name = "trt.Constant." + layer_name

#         self.layer_post_process(trt_layer, layer_name, None)
#         x = trt_layer.get_output(0)
#         return x

#     def addShuffle(
#         self,
#         x: trt.ITensor,
#         first_transpose: trt.Permutation,
#         reshape_dims: trt.Dims,
#         second_transpose: trt.Permutation,
#         layer_name: Optional[str] = None
#     ) -> trt.ITensor:
#         """"""
#         trt_layer = self.network.add_shuffle(x)
#         if first_transpose is not None:
#             trt_layer.first_transpose = first_transpose

#         if reshape_dims is not None:
#             trt_layer.reshape_dims = reshape_dims

#         if second_transpose is not None:
#             trt_layer.second_transpose = second_transpose

#         if layer_name is None:
#             layer_name = "trt.Shuffle"
#         else:
#             layer_name = "trt.Shuffle." + layer_name

#         self.layer_post_process(trt_layer, layer_name, None)

#         x = trt_layer.get_output(0)
#         return x


# class InferHelper():
#     """"""
#     def __init__(self, plan_name, trt_logger):
#         """"""
#         self.logger = trt_logger
#         self.runtime = trt.Runtime(trt_logger)
#         with open(plan_name, 'rb') as f:
#             self.engine = self.runtime.deserialize_cuda_engine(f.read())
#             self.context = self.engine.create_execution_context()
#             self.context.active_optimization_profile = 0

#     def infer(self, inputs: list):
#         nInput = len(inputs)

#         bufferD = []
#         # alloc memory
#         for i in range(nInput):
#             bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
#             cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
#             self.context.set_binding_shape(i, tuple(inputs[i].shape))
#             # print(inputs[i].nbytes)

#         # for i in range(0, self.engine.num_bindings):
#             # print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

#         outputs = []
#         for i in range(len(inputs), self.engine.num_bindings):
#             outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

#         nOutput = len(outputs)
#         for i in range(nOutput):
#             bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
#             # print(outputs[i].nbytes)

#         for i in range(len(inputs), self.engine.num_bindings):
#             trt_output_shape = self.context.get_binding_shape(i)
#             output_idx = i - len(inputs)
#             if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
#                 self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
#                 self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
#                 self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
#                 assert(0)

#         # warm up
#         self.context.execute_v2(bufferD)

#         T1 = time.perf_counter()

#         self.context.execute_v2(bufferD)

#         T2 =time.perf_counter()
#         print("time=" + str((T2-T1) * 1000) + "ms")

#         for i in range(nInput, nInput + nOutput):
#             cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

#         for i in range(0, len(outputs)):
#             print("outputs.shape:" + str(outputs[i].shape))
#             print("outputs.sum:" + str(outputs[i].sum()))
#             # print(outputs[i])

#             # print("trt_output.shape:" + str(trt_output.shape))
#             # print("trt_output.sum:" + str(trt_output.sum()))
#             # print(trt_output.view(-1)[0:10])
#             # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
#             # print("====================")
#         return outputs
#         # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)
