
import jittor as jt
from jittor import Function, Var, init
from jittor import nn

from .util import compile_cuda_op

fused_bias_act = compile_cuda_op('fused_act')


class FusedLeakyReLUFunctionBackward(Function):
    def execute(self, grad_output, out, bias, negative_slope, scale):
        self._save_vars = out
        self._negative_slope = negative_slope
        self._scale = scale
        empty = jt.zeros(0, grad_output.dtype)
        grad_input: Var = fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if (grad_input.ndim > 2):
            dim += list(range(2, grad_input.ndim))
        if bias:
            grad_bias = grad_input.sum(tuple(dim)).detach()
        else:
            grad_bias = empty
        return (grad_input, grad_bias)

    def grad(self, gradgrad_input, gradgrad_bias):
        out = self._save_vars
        if gradgrad_bias is None:
            gradgrad_bias = jt.zeros_like(gradgrad_input)
        gradgrad_out = fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, self._negative_slope, self._scale)
        return (gradgrad_out, None, None, None, None)


class FusedLeakyReLUFunction(Function):
    def execute(self, input: Var, bias, negative_slope, scale):
        empty = jt.zeros(0, input.dtype)
        self._bias = (bias is not None)
        if (bias is None):
            bias = empty
        out = fused_bias_act(
            input, bias, empty, 3, 0, negative_slope, scale)
        self._save_vars = out
        self._negative_slope = negative_slope
        self._scale = scale
        return out

    def grad(self, grad_output):
        out = self._save_vars
        (grad_input, grad_bias) = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, self._bias, self._negative_slope, self._scale)
        return (grad_input, grad_bias, None, None)


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=(2 ** 0.5)):
        super().__init__()
        if bias:
            self.bias = jt.start_grad(jt.array(jt.zeros(channel)))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def execute(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=(2 ** 0.5)):
    if (bias is not None):
        rest_dim = ([1] * ((input.ndim - bias.ndim) - 1))
        return (nn.leaky_relu((input + bias.view((1, bias.shape[0], *rest_dim))), negative_slope) * scale)
    else:
        return (nn.leaky_relu(input, negative_slope) * scale)

