#include "fused_act_op.h"
#include "var.h"

namespace jittor {

#ifndef JIT
FusedActOp::FusedActOp(Var *input, Var *bias, Var *refer, int act, int grad,
                           float64 alpha, float64 scale) {
  flags.set(NodeFlags::_cuda, 1);
  flags.set(NodeFlags::_cpu, 0);
  output = create_output(input->shape, input->dtype());
  this->input = input, this->bias = bias, this->refer = refer;
  this->act = act, this->grad = grad;
  this->alpha = alpha, this->scale = scale;
}

void FusedActOp::jit_prepare(JK &jk) {
  add_jit_define(jk, "T", output->dtype());
}

#else  // JIT
#ifdef JIT_cpu
void FusedActOp::jit_run() {
  fprintf(stderr, "[fused_act] Unable to run this op on CPU, please check JIT setting\n");
  exit(EXIT_FAILURE);
}
#else
template <typename scalar_t>
static __global__ void
fused_bias_act_kernel(scalar_t *out, const scalar_t *p_x, const scalar_t *p_b,
                      const scalar_t *p_ref, int act, int grad, scalar_t alpha,
                      scalar_t scale, int loop_x, int size_x, int step_b,
                      int size_b, int use_bias, int use_ref) {
  int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

  scalar_t zero = 0.0;

  for (int loop_idx = 0; loop_idx < loop_x && xi < size_x;
       loop_idx++, xi += blockDim.x) {
    scalar_t x = p_x[xi];

    if (use_bias) {
      x += p_b[(xi / step_b) % size_b];
    }

    scalar_t ref = use_ref ? p_ref[xi] : zero;

    scalar_t y;

    switch (act * 10 + grad) {
    default:
    case 10:
      y = x;
      break;
    case 11:
      y = x;
      break;
    case 12:
      y = 0.0;
      break;

    case 30:
      y = (x > 0.0) ? x : x * alpha;
      break;
    case 31:
      y = (ref > 0.0) ? x : x * alpha;
      break;
    case 32:
      y = 0.0;
      break;
    }

    out[xi] = y * scale;
  }
}

void FusedActOp::jit_run() {
  auto x = input;
  auto b = bias;
  auto ref = refer;

  int use_bias = b->numel() ? 1 : 0;
  int use_ref = ref->numel() ? 1 : 0;

  int size_x = x->numel();
  int size_b = b->numel();
  int step_b = 1;
  for (int i = 2; i < x->shape.size(); ++i) {
    step_b *= x->shape[i];
  }

  int loop_x = 4;
  int block_size = 4 * 32;
  int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

  fused_bias_act_kernel<T><<<grid_size, block_size, 0>>>(
      output->ptr<T>(), x->ptr<T>(), b->ptr<T>(), ref->ptr<T>(), act, grad,
      alpha, scale, loop_x, size_x, step_b, size_b, use_bias, use_ref);
}
#endif // JIT_cpu
#endif

} // namespace jittor