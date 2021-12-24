
import jittor as jt
from jittor import Function, init
from jittor import nn


from .util import compile_cuda_op

upfirdn2d_gpu = compile_cuda_op('upfirdn2d')


class UpFirDn2dBackward(Function):
    def execute(self, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        (up_x, up_y) = up
        (down_x, down_y) = down
        (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1) = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_gpu(
            grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(
            (in_size[0], in_size[1], in_size[2], in_size[3]))
        self._save_vars = kernel
        (pad_x0, pad_x1, pad_y0, pad_y1) = pad
        self._up_x = up_x
        self._up_y = up_y
        self._down_x = down_x
        self._down_y = down_y
        self._pad_x0 = pad_x0
        self._pad_x1 = pad_x1
        self._pad_y0 = pad_y0
        self._pad_y1 = pad_y1
        self._in_size = in_size
        self._out_size = out_size
        return grad_input

    def grad(self, gradgrad_input):
        kernel = self._save_vars
        gradgrad_input = gradgrad_input.reshape(-1,
                                                self._in_size[2], self._in_size[3], 1)
        gradgrad_out = upfirdn2d_gpu(
            gradgrad_input, kernel, self._up_x, self._up_y, self._down_x, self._down_y, self._pad_x0, self._pad_x1, self._pad_y0, self._pad_y1)
        gradgrad_out = gradgrad_out.view(
            (self._in_size[0], self._in_size[1], self._out_size[0], self._out_size[1]))
        return (gradgrad_out, None, None, None, None, None, None, None, None)


class UpFirDn2d(Function):
    def execute(self, input, kernel, up, down, pad):
        (up_x, up_y) = up
        (down_x, down_y) = down
        (pad_x0, pad_x1, pad_y0, pad_y1) = pad
        (kernel_h, kernel_w) = kernel.shape
        (batch, channel, in_h, in_w) = input.shape
        self._in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
     
        self._save_vars = (kernel, jt.flip(kernel, [0, 1]))

        out_h = ((((((in_h * up_y) + pad_y0) + pad_y1) - kernel_h) // down_y) + 1)
        out_w = ((((((in_w * up_x) + pad_x0) + pad_x1) - kernel_w) // down_x) + 1)
        self._out_size = (out_h, out_w)
        self._up = (up_x, up_y)
        self._down = (down_x, down_y)
        self._pad = (pad_x0, pad_x1, pad_y0, pad_y1)
        g_pad_x0 = ((kernel_w - pad_x0) - 1)
        g_pad_y0 = ((kernel_h - pad_y0) - 1)
        g_pad_x1 = (((((in_w * up_x) - (out_w * down_x)) + pad_x0) - up_x) + 1)
        g_pad_y1 = (((((in_h * up_y) - (out_h * down_y)) + pad_y0) - up_y) + 1)
        self._g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        out = upfirdn2d_gpu(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(((- 1), channel, out_h, out_w))
        return out

    def grad(self, grad_output):
        (kernel, grad_kernel) = self._save_vars
        grad_input = UpFirDn2dBackward.apply(
            grad_output, kernel, grad_kernel, self._up, self._down, self._pad, self._g_pad, self._in_size, self._out_size)
        return (grad_input, None, None, None, None)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):

    out = UpFirDn2d.apply(input, kernel, (up, up),
                          (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out

