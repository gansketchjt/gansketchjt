#pragma once
#include "op.h"

struct UpFirDn2DKernelParams {
  int up_x;
  int up_y;
  int down_x;
  int down_y;
  int pad_x0;
  int pad_x1;
  int pad_y0;
  int pad_y1;

  int major_dim;
  int in_h;
  int in_w;
  int minor_dim;
  int kernel_h;
  int kernel_w;
  int out_h;
  int out_w;
  int loop_major;
  int loop_x;
};

namespace jittor {

struct Upfirdn2dOp : Op {
  Var *output;
  Upfirdn2dOp(Var* input, Var* kernel, int up_x, int up_y, int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0, int pad_y1);

  const char *name() const override { return "upfirdn2d"; }

  UpFirDn2DKernelParams p;

  Var *x, *k;
  int up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1;

  DECLARE_jit_run;
};

} // namespace jittor