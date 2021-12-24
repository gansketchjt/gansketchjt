#pragma once
#include "op.h"

namespace jittor {

struct FusedActOp : Op {
  Var *output;
  FusedActOp(Var* input, Var* bias, Var* refer, int act, int grad, float64 alpha, float64 scale);

  const char *name() const override { return "fused_act"; }

  Var *input, *bias, *refer;
  int act, grad;
  float alpha, scale;

  DECLARE_jit_run;
};

} // namespace jittor