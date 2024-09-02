#include "hetu/autograd/ops/Optimizer.h"
#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

bool OptimizerUpdateOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // 这块内容本质上就是在做grad的reduce, 在新版本中, 通过autograd处理后返回的grad已经是用allreduce处理过的了(partial->duplicate), 因此这一块代码可以省略
  // TODO: reuse还需要检查要不要在autograd里加
  // if (pg.num_devices() > 1) {
  //   Tensor& grad = _inputs[1];
  //   HT_ASSERT(grad->producer()->placement_group() == pg)
  //     << "Currently we only support gathering gradients in data parallel";

  //   // add all reduce op; re-use if possible
  //   bool reused = false;
  //   for (size_t i = 0; i < grad->num_consumers(); i++) {
  //     Operator& consumer_op = grad->consumer(i);
  //     if (is_all_reduce_op(consumer_op) &&
  //         consumer_op->placement_group() == pg) {
  //       ReplaceInput(1, consumer_op->output(0));
  //       reused = true;
  //       break;
  //     }
  //   }
  //   if (!reused) {
  //     // TODO: make the all reduce op in-place so that `get_grad` of VariableOp
  //     // returns the reduced gradient
  //     AllReduceOp all_reduce_op(
  //       grad,
  //       OpMeta().set_device_group(pg).set_name(grad->name() + "_AllReduce"));
  //     all_reduce_op->MapToParallelDevices(pg);
  //     ReplaceInput(1, all_reduce_op->output(0));
  //   }
  // }
  return OperatorDef::DoMapToParallelDevices(pg);
}

void SGDUpdateOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  NDArray param = inputs.at(0);
  NDArray grad = inputs.at(1);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::SGDUpdate, grad, param, velocity,
                                  learning_rate(), 0, false, stream());
  // HT_LOG_INFO << "after update_op: param: " << param << ", grad: " << grad;
}

void MomemtumUpdateOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  NDArray param = inputs.at(0);
  NDArray grad = inputs.at(1);
  NDArray velocity = inputs.at(2);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::SGDUpdate, grad, param, velocity,
    learning_rate(), momentum(), nesterov(), stream());
}

} // namespace autograd
} // namespace hetu
