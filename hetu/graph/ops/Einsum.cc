#include "hetu/graph/ops/Einsum.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/stream/CPUStream.h"
#include <bitset>

namespace hetu {
namespace graph {

bool operator==(const EinsumParameters& l, const EinsumParameters& r) {
    return (l._msg == r._msg
            && l._input_msgs == r._input_msgs
            && l._output_msg == r._output_msg
            && l.input_dims == r.input_dims
            && l.output_dims == r.output_dims
            && l.num_labels == r.num_labels
            && l.output_labels_idx == r.output_labels_idx
            && l.undefined_labels == r.undefined_labels
            && l.output_size == r.output_size
            && l.num_output_labels == r.num_output_labels
            && l.elli_len == r.elli_len
            && l.input_elli_len == r.input_elli_len
            && l.elli_pos == r.elli_pos);
}

NDArray sumproduct_pair(Operator& op, NDArray& left_, NDArray& right_, HTShape sum_dims_,
                        bool keepdim) {
  HT_ASSERT(left_->ndim() == right_->ndim())
    << "number of dimensions must match";
  if (sum_dims_.size() == 0)
    return NDArray::mul(left_, right_, op->instantiation_ctx().stream_index);
  int64_t dim = left_->ndim();

  constexpr size_t dim_bitset_size = 64;
  HT_ASSERT(dim <= (int64_t) dim_bitset_size)
    << "only tensors with up to " << dim_bitset_size << " dims are supported";
  std::bitset<dim_bitset_size> sum_dims;
  for (size_t i = 0; i < sum_dims_.size(); ++i) {
    size_t d = sum_dims_[i];
    HT_ASSERT(!sum_dims[d])
      << "dim " << d << " appears multiple times in the list of dims";
    sum_dims[d] = true;
  }

  // dimensions that will be part of the output (i.e. not summed over) in three
  // vectors dims in lro appear in left, right and output, similarly lo: left
  // and output, ro: right and output also the sizes are kept track of for
  // reshaping
  HTShape lro, lo, ro;
  int32_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  NDArray left = NDArray::copy(left_, op->instantiation_ctx().stream_index);
  NDArray right = NDArray::copy(right_, op->instantiation_ctx().stream_index);
  for (int i = 0; i < dim; ++i) {
    auto sl = left->shape(i) > 1;
    auto sr = right->shape(i) > 1;
    if (sum_dims[i]) { // first dimensions that will be summed over after
                       // multiplication
      if (sl && sr) { // dimensions nontrivially in both left and right must be
                      // of the same size
        HT_ASSERT(left->shape(i) == right->shape(i))
          << "non-broadcast dimensions must match";
        sum_size *= left->shape(i);
      } else if (sl) { // if it is only in one of left and right, we can sum
                       // right away
        left = NDArray::sum(left, {i}, {true}, op->instantiation_ctx().stream_index);
      } else if (sr) {
        right = NDArray::sum(right, {i}, {true}, op->instantiation_ctx().stream_index);
      }
    } else if (sl && sr) { // now deal with dimensions  dimensions that will be
                           // in the output
      // dimensions nontrivially in both left and right must be of the same size
      HT_ASSERT(left->shape(i) == right->shape(i))
        << "non-broadcast dimensions must match";
      lro.push_back(i);
      lro_size *= left->shape(i);
    } else if (sl) { // keep track of dimensions appearing only once
      lo.push_back(i);
      lo_size *= left->shape(i);
    } else {
      ro.push_back(i);
      ro_size *= right->shape(i);
    }
  }

  HTShape out_size;
  for (auto& d : lro)
    out_size.push_back(left->shape(d));
  for (auto& d : lo)
    out_size.push_back(left->shape(d));
  for (auto& d : sum_dims_) {
    out_size.push_back(1);
    (void) (d);
  }; // avoid warning about not using d
  for (auto& d : ro)
    out_size.push_back(right->shape(d));

  HTShape lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  HTShape rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  HTShape opermutation(lro.size() + lo.size() + sum_dims_.size() + ro.size(),
                       -1);
  {
    int32_t i = 0;

    for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
  }
  // now we can execute the operations above
  left = NDArray::permute(left, lpermutation, op->instantiation_ctx().stream_index);
  HTShape ls(3);
  ls[0] = lro_size;
  ls[1] = lo_size;
  ls[2] = sum_size;

  left = NDArray::reshape(left, ls, op->instantiation_ctx().stream_index);

  right = NDArray::permute(right, rpermutation, op->instantiation_ctx().stream_index);
  HTShape rs(3);
  rs[0] = lro_size;
  rs[1] = sum_size;
  rs[2] = ro_size;

  right = NDArray::reshape(right, rs, op->instantiation_ctx().stream_index);

  NDArray result = NDArray::bmm(left, right, false, false, op->instantiation_ctx().stream_index);
  HTShape os(out_size.size());
  for (size_t i = 0; i < out_size.size(); ++i) {
    os[i] = out_size[i];
  }

  result = NDArray::reshape(result, os, op->instantiation_ctx().stream_index);
  result = NDArray::permute(result, opermutation, op->instantiation_ctx().stream_index);

  // finally squeeze summed dimensions if desired
  if (!keepdim) {
    HTShape sizes = result->shape();
    for (int i = dim - 1; i >= 0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }

    result = NDArray::reshape(result, sizes, op->instantiation_ctx().stream_index);
  }
  return result;
}

void EinsumOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArrayList permuted_inputs = {};
  EinsumParameters para = params();
  for (size_t i = 0; i < inputs.size(); ++i) {
    HTShape perm_shape(para.num_output_labels, -1);
    LabelMap label_dim;
    OpDim input_labels;
    HTShape input_shape;
    NDArray input_tensor = NDArray::copy(inputs.at(i), op->instantiation_ctx().stream_index);
    input_labels = para.input_dims[i];
    input_shape = inputs.at(i)->shape();

    int j = 0;
    for (const auto& label : input_labels) {
      if (label == "...") {
        // Add missing dimensions covered by the ellipsis
        int missing_dims =
          para.elli_len - (input_shape.size() - input_labels.size() + 1);
        for (int k = 0; k < missing_dims; ++k) {
          input_tensor = NDArray::unsqueeze(input_tensor, j);
        }
        for (int k = 0; k < para.elli_len; ++k) {
          perm_shape[para.elli_pos + k] = j++;
        }
      } else if (label_dim.find(label) != label_dim.end()) {
        // Repeated label, take diagonal
        int dim = label_dim[label];
        HT_ASSERT(input_tensor->shape(j) == input_tensor->shape(dim))
          << j << ":" << input_tensor->shape(j) << "," << dim << ":"
          << input_tensor->shape(dim);
        // HT_LOG_INFO << input_tensor;
        input_tensor = NDArray::diagonal(input_tensor, dim, j, 0, op->instantiation_ctx().stream_index);
        // HT_LOG_INFO << input_tensor;
        input_tensor = NDArray::movedim(input_tensor, -1, dim, op->instantiation_ctx().stream_index);
      } else {
        // Lookup output index for label
        label_dim[label] = j;
        perm_shape[para.output_labels_idx.find(label)->second] = j++;
      }
    }

    // Add dimensions for missing labels
    for (int64_t& index : perm_shape) {
      if (index == -1) {
        input_tensor = NDArray::unsqueeze(input_tensor, -1);
        index = j++;
      }
    }
    NDArray permuted_input = NDArray::permute(input_tensor, perm_shape, op->instantiation_ctx().stream_index);
    permuted_inputs.emplace_back(permuted_input);
  }


  // Check if operands broadcast and keep track of last operand with
  // dimension size != 1 for optimizing reductions
  std::vector<std::size_t> dim_last_op(para.num_output_labels, 0);
  bool has_zero_size_dim = false;

  for (int dim = 0; dim < para.num_output_labels; dim++) {
    int output_dim_size = permuted_inputs[0]->shape(dim);
    for (size_t i = 1; i < inputs.size(); ++i) {
      int input_dim_size = permuted_inputs[i]->shape(dim);
      HT_ASSERT(output_dim_size == input_dim_size || output_dim_size == 1 ||
                input_dim_size == 1)
        << "input" << i << "cannot broadcast to outshape."
        << "input" << i << "'s shape:" << permuted_inputs[i]->shape()
        << "output shape:" << permuted_inputs[0]->shape();
      if (input_dim_size != 1) {
        output_dim_size = input_dim_size;
        dim_last_op[dim] = i;
      }
    }
    if (output_dim_size == 0)
      has_zero_size_dim = true;
  }

  // Compute result
  // Fast path for when an operand has zero sized dim
  if (has_zero_size_dim) {
    return;
  }

  // Sum out or squeeze dimensions that are size 1 for all later operands
  NDArray output_tensor = NDArray::copy(permuted_inputs[0], op->instantiation_ctx().stream_index);
  int dim = para.output_size;
  for (int i = dim; i < para.num_output_labels; ++i, ++dim) {
    if (dim_last_op[i] == 0) {
      if (output_tensor->shape(dim) == 1) {
        output_tensor = NDArray::squeeze(output_tensor, dim--);
      } else {
        output_tensor = NDArray::sum(output_tensor, {dim--}, {false}, op->instantiation_ctx().stream_index);
      }
    }
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    NDArray permuted_input = NDArray::copy(permuted_inputs[i], op->instantiation_ctx().stream_index);
    HTShape sum_dims;
    // Sum out or squeeze dimensions that are size 1 for all later operands
    dim = para.output_size;
    for (int j = dim; j < para.num_output_labels; ++j, ++dim) {
      if (dim_last_op[j] < i) {
        permuted_input = NDArray::squeeze(permuted_input, dim--);
      } else if (dim_last_op[j] == i) {
        if (output_tensor->shape(dim) == 1) {
          permuted_input = NDArray::sum(permuted_input, {dim}, {false}, op->instantiation_ctx().stream_index);
          output_tensor = NDArray::squeeze(output_tensor, dim--);
        } else {
          sum_dims.push_back(dim);
        }
      }
    }
    // Multiply tensors and sum out dimensions in sum_dims
    if (sum_dims.empty()) {
      output_tensor = NDArray::mul(output_tensor, permuted_input, op->instantiation_ctx().stream_index);
    } else if (sum_dims.size() == output_tensor->ndim()) {
      NDArray flatten_input = NDArray::flatten(permuted_input, 0, -1);
      NDArray flatten_output = NDArray::flatten(output_tensor, 0, -1);
      output_tensor =
        NDArray::sum(NDArray::dot(flatten_input, flatten_output), {0}, {false}, op->instantiation_ctx().stream_index);
    } else {
      output_tensor =
        sumproduct_pair(op, output_tensor, permuted_input, sum_dims, false);
    }
  }
  outputs[0] = NDArray::reshape(output_tensor, op->output(0)->shape(), op->instantiation_ctx().stream_index);
}

TensorList EinsumOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  EinsumParameters para = params();
  int len = op->num_inputs();
  TensorList grad_inputs = {};
  std::string ori_cmd = fetch_msg();
  auto g_op_meta = op->grad_op_meta();
  for (int i = 0; i < len; ++i) {
    std::string grad_msg;
    TensorList grad_oplinkers = {};
    for (int j = 0; j < len - 1; ++j) {
      if (i == j) {
        grad_msg = grad_msg + para._output_msg + ",";
        grad_oplinkers.emplace_back(grad_outputs.at(0));
      } else {
        grad_msg = grad_msg + para._input_msgs[j] + ",";
        grad_oplinkers.emplace_back(op->input(j));
      }
    }
    if (i == len - 1) {
      grad_msg = grad_msg + para._output_msg + "->";
      grad_oplinkers.emplace_back(grad_outputs.at(0));
    } else {
      grad_msg = grad_msg + para._input_msgs[len - 1] + "->";
      grad_oplinkers.emplace_back(op->input(len - 1));
    }
    grad_msg = grad_msg + para._input_msgs[i];
    auto grad_input = op->requires_grad(i) ? MakeEinsumGradientOp(grad_msg, grad_oplinkers, op->output(0), op->input(i),
                                            g_op_meta.set_name(op->grad_name(i)))
                                          : Tensor();
    grad_inputs.emplace_back(grad_input);
  }
  return grad_inputs;
}

HTShapeList EinsumOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  EinsumParameters para = params();
  LabelMap label_to_size;
  HTShape output_shape(para.output_size);
  std::vector<int> elli_size(para.elli_len, -1);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    int input_idx = 0;
    HTShape perm_shape(para.num_output_labels, 0);
    OpDim input_labels = para.input_dims[i];
    HTShape input_shape = input_shapes.at(i);
    for (const auto& label : input_labels) {
      if (label == "...") {
        if (para.input_elli_len[i] == para.elli_len) {
          for (int k = 0; k < para.elli_len; ++k) {
            if (elli_size[k] == -1) {
              elli_size[k] = input_shape[input_idx + k];
            } else {
              // HT_ASSERT(elli_size[k] == input_shape[input_idx + k]);
            }
          }
        }
        input_idx += para.elli_len;
      } else {
        if (label_to_size.find(label) == label_to_size.end()) {
          label_to_size[label] = input_shape[input_idx];
        } else {
          HT_ASSERT(label_to_size[label] == input_shape[input_idx])
            << label << ":" << label_to_size[label] << ","
            << input_shape[input_idx] << std::endl;
        }
        input_idx += 1;
      }
    }
  }
  if (para.output_dims.empty()) {
    output_shape = {1};
  } else {
    int output_idx = 0;
    for (const auto& label : para.output_dims.at(0)) {
      if (label == "...") {
        for (int k = 0; k < para.elli_len; ++k) {
          output_shape[para.elli_pos + k] = elli_size[k];
        }
        output_idx += para.elli_len;
      } else {
        output_shape[output_idx] = label_to_size[label];
        output_idx += 1;
      }
    }
    if (output_shape.size() == 0) {
      output_shape = {1};
    }
  }
  return {output_shape};
}

void EinsumGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs, NDArrayList& outputs, 
                                     RuntimeContext& ctx) const {
  EinsumParameters para = params();
  NDArrayList permuted_inputs = {};
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    HTShape perm_shape(para.num_output_labels, -1);
    LabelMap label_dim;
    OpDim input_labels;
    HTShape input_shape;
    NDArray input_tensor;
    input_labels = para.input_dims[i];
    input_shape = inputs.at(i)->shape();
    input_tensor = NDArray::copy(inputs.at(i), op->instantiation_ctx().stream_index);

    int j = 0;
    for (const auto& label : input_labels) {
      if (label == "...") {
        // Add missing dimensions covered by the ellipsis
        int missing_dims =
          para.elli_len - (input_shape.size() - input_labels.size() + 1);
        for (int k = 0; k < missing_dims; ++k) {
          input_tensor = NDArray::unsqueeze(input_tensor, j);
        }
        for (int k = 0; k < para.elli_len; ++k) {
          perm_shape[para.elli_pos + k] = j++;
        }
      } else if (label_dim.find(label) != label_dim.end()) {
        // Repeated label, take diagonal
        int dim = label_dim[label];
        HT_ASSERT(input_shape[j] == input_shape[dim]);

        input_tensor = NDArray::diagonal(input_tensor, dim, j, 0, op->instantiation_ctx().stream_index);

        input_tensor = NDArray::movedim(input_tensor, -1, dim, op->instantiation_ctx().stream_index);
      } else {
        // Lookup output index for label
        label_dim[label] = j;
        perm_shape[para.output_labels_idx.find(label)->second] = j++;
      }
    }

    // Add dimensions for missing labels
    for (int64_t& index : perm_shape) {
      if (index == -1) {
        input_tensor = NDArray::unsqueeze(input_tensor, -1);
        index = j++;
      }
    }
    NDArray permuted_input = NDArray::permute(input_tensor, perm_shape, op->instantiation_ctx().stream_index);
    permuted_inputs.emplace_back(permuted_input);
  }
  // Check if operands broadcast and keep track of last operand with
  // dimension size != 1 for optimizing reductions
  std::vector<std::size_t> dim_last_op(para.num_output_labels, 0);
  bool has_zero_size_dim = false;

  for (int dim = 0; dim < para.num_output_labels; dim++) {
    int output_dim_size = permuted_inputs[0]->shape(dim);
    for (size_t i = 1; i < inputs.size() - 1; ++i) {
      int input_dim_size = permuted_inputs[i]->shape(dim);
      HT_ASSERT(output_dim_size == input_dim_size || output_dim_size == 1 ||
                input_dim_size == 1)
        << "input" << i << "cannot broadcast to outshape."
        << "input" << i << "'s shape:" << inputs.at(0)->shape()
        << "output shape:" << outputs.at(0)->shape();
      if (input_dim_size != 1) {
        output_dim_size = input_dim_size;
        dim_last_op[dim] = i;
      }
    }
    if (output_dim_size == 0)
      has_zero_size_dim = true;
  }

  if (has_zero_size_dim) {
    return;
  }

  // Sum out or squeeze dimensions that are size 1 for all later operands
  NDArray output_tensor = NDArray::copy(permuted_inputs[0], op->instantiation_ctx().stream_index);
  int dim = para.output_size;
  for (int i = dim; i < para.num_output_labels; ++i, ++dim) {
    if (dim_last_op[i] == 0) {
      if (output_tensor->shape(dim) == 1) {
        output_tensor = NDArray::squeeze(output_tensor, dim--);
        // output_tensor.squeeze_(dim--);
      } else {
        output_tensor = NDArray::sum(output_tensor, {dim--}, {false}, op->instantiation_ctx().stream_index);
      }
    }
  }

  for (size_t i = 1; i < inputs.size() - 1; ++i) {
    NDArray permuted_input = permuted_inputs[i];
    HTShape sum_dims;

    // Sum out or squeeze dimensions that are size 1 for all later operands
    dim = para.output_size;
    for (int j = dim; j < para.num_output_labels; ++j, ++dim) {
      if (dim_last_op[j] < i) {
        permuted_input = NDArray::squeeze(permuted_input, dim--);
      } else if (dim_last_op[j] == i) {
        if (output_tensor->shape(dim) == 1) {
          permuted_input = NDArray::sum(permuted_input, {dim}, {false}, op->instantiation_ctx().stream_index);
          output_tensor = NDArray::squeeze(output_tensor, dim--);
        } else {
          sum_dims.push_back(dim);
        }
      }
    }
    // Multiply tensors and sum out dimensions in sum_dims
    if (sum_dims.empty()) {
      output_tensor = NDArray::mul(output_tensor, permuted_input, op->instantiation_ctx().stream_index);
    } else if (sum_dims.size() == output_tensor->ndim()) {
      NDArray flatten_input = NDArray::flatten(permuted_input, 0, -1);
      NDArray flatten_output = NDArray::flatten(output_tensor, 0, -1);
      output_tensor =
        NDArray::sum(NDArray::dot(flatten_input, flatten_output, op->instantiation_ctx().stream_index), {0}, {false}, op->instantiation_ctx().stream_index);
    } else {
      output_tensor =
        sumproduct_pair(op, output_tensor, permuted_input, sum_dims, false);
    }
  }
  LabelMap first_output_idx;
  int output_idx = 0;
  for (const auto& label : para.output_dims.at(0)) {
    if (label == "...") {
      output_idx += para.elli_len;
    } else {
      if (first_output_idx.find(label) == first_output_idx.end()) {
        if (para.undefined_labels.find(label) != para.undefined_labels.end()) {
          output_tensor = NDArray::adddim(output_tensor, output_idx,
                                          outputs.at(0)->shape(output_idx),
                                          op->instantiation_ctx().stream_index);
        }
        first_output_idx.emplace(label, output_idx);
      } else {
        int first_idx = first_output_idx.find(label)->second;
        output_tensor =
          NDArray::diagonal_grad(NDArray::contiguous(output_tensor, op->instantiation_ctx().stream_index),
                                 first_idx, output_idx, op->instantiation_ctx().stream_index);
      }
      output_idx += 1;
    }
  }
  outputs[0] = NDArray::reshape(output_tensor, op->output(0)->shape(), op->instantiation_ctx().stream_index);
}

HTShapeList EinsumGradientOpImpl::DoInferShape(Operator& op,
                                               const HTShapeList& input_shapes,
                                               RuntimeContext& ctx) const {
  // HT_LOG_INFO << _params._msg << " " << input_shapes[input_shapes.size() - 1];
  return {input_shapes[input_shapes.size() - 1]};
}

EinsumParameters EinsumParseMsg(const TensorList& inputs,
                                std::string cmd) {
  EinsumParameters params;
  params._msg = cmd;
  params.input_dims = {};
  params.output_dims = {};
  params._input_msgs.resize(0);
  size_t pos = cmd.find("->");
  std::string input_cmd = cmd.substr(0, pos);
  std::string output_cmd;
  if (pos != cmd.npos)
    output_cmd = cmd.substr(pos + 2);
  else
    output_cmd = "";
  params._output_msg = output_cmd;
  int num_ellipsis = 0;
  int i = 0;
  bool flag = true;
  while (flag) {
    size_t mid = input_cmd.find(",");
    std::string tmp_input;
    if (mid == input_cmd.npos) {
      tmp_input = input_cmd;
      flag = false;
    } else {
      tmp_input = input_cmd.substr(0, mid);
      input_cmd = input_cmd.substr(mid + 1);
    }
    params._input_msgs.emplace_back(tmp_input);
    int input_len = tmp_input.length();
    num_ellipsis = 0;
    i = 0;
    OpDim tmp_dim = {};
    while (i < input_len) {
      switch (tmp_input[i]) {
        case '.':
          HT_ASSERT(num_ellipsis == 0 && tmp_input[i + 1] == '.' &&
                    tmp_input[i + 2] == '.')
            << "Invalid command.More than one ellipsis";
          num_ellipsis++;
          tmp_dim.emplace_back("...");
          i += 3;
          break;
        case ' ': i++; break;
        default:
          HT_ASSERT(std::isalpha(tmp_input[i]))
            << "Invalid command.Use invalid charactor.Charactors should in [a-zA-Z].";
          tmp_dim.emplace_back(tmp_input.substr(i, 1));
          i++;
      }
    }
    params.input_dims.emplace_back(tmp_dim);
  }
  HT_ASSERT(params.input_dims.size() == inputs.size())
    << "The number of tensors is incorrect, expected " << inputs.size()
    << ", got " << params.input_dims.size();

  params.elli_len = 0;
  params.input_elli_len = std::vector<int>(inputs.size(), -1);
  for (size_t i = 0; i < inputs.size(); ++i) {
    HTShape input_shape = inputs[i]->shape();
    OpDim input_dim = params.input_dims.at(i);
    int ndims = input_shape.size();
    int nlabels = input_dim.size();
    bool has_ellipsis = false;
    for (const auto& dim_label : input_dim) {
      if (dim_label == "...") {
        nlabels--;
        has_ellipsis = true;
        params.input_elli_len[i] = ndims - nlabels;
        params.elli_len = std::max(params.elli_len, ndims - nlabels);
      } else {
        if (params.num_labels.find(dim_label) != params.num_labels.end()) {
          params.num_labels[dim_label]++;
        } else {
          params.num_labels.emplace(dim_label, 1);
        }
      }
    }
    if (has_ellipsis) {
      HT_ASSERT(nlabels <= ndims)
        << "num of dims is not equal to num of labels.";
    } else {
      HT_ASSERT(nlabels == ndims)
        << "num of dims is not equal to num of labels."
        << "num dims:" << ndims << ",labels:" << input_dim;
    }
  }
  int output_idx = 0;
  params.elli_pos = 0;
  bool has_elli = false;
  if (pos != cmd.npos) {
    std::string tmp_output;
    tmp_output = output_cmd;
    int output_len = tmp_output.length();
    num_ellipsis = 0;
    i = 0;
    OpDim tmp_dim = {};
    // int label_idx = 0;
    while (i < output_len) {
      switch (tmp_output[i]) {
        case '.':
          HT_ASSERT(num_ellipsis == 0 && tmp_output[i + 1] == '.' &&
                    tmp_output[i + 2] == '.')
            << "Invalid command.more than one ellipsis";
          num_ellipsis++;
          tmp_dim.emplace_back("...");
          i += 3;
          params.elli_pos = output_idx;
          output_idx += params.elli_len;
          has_elli = true;
          break;
        case ' ': i++; break;
        default:
          HT_ASSERT(std::isalpha(tmp_output[i]))
            << "Invalid command.Use invalid charactor.Charactors should in [a-zA-Z].";
          std::string label = tmp_output.substr(i, 1);
          HT_ASSERT(params.num_labels.find(label) != params.num_labels.end())
            << "This label occurs didn't occur in inputs:" << label;
          HT_ASSERT(params.output_labels_idx.find(label) == params.output_labels_idx.end())
            << "This label occurs more than once in output:" << label;
          params.output_labels_idx.emplace(label, output_idx++);
          tmp_dim.emplace_back(tmp_output.substr(i, 1));
          i++;
      }
    }
    params.output_dims.emplace_back(tmp_dim);
  } else {
    output_idx = params.elli_len;
    has_elli = true;
    params._output_msg = "";
    OpDim tmp_dim = {};
    if (params.elli_len > 0) {
      params._output_msg += "...";
      tmp_dim.emplace_back("...");
    }
    for (auto it = params.num_labels.begin(); it != params.num_labels.end(); ++it) {
      if (it->second == 1) {
        params.output_labels_idx[it->first] = output_idx++;
        tmp_dim.emplace_back(it->first);
      }
    }
    // sort by dict order
    int len = tmp_dim.size();
    for (int i = 0; i < len - 1; ++i) {
      for (int j = i + 1; j < len; ++j) {
        if (tmp_dim[i] > tmp_dim[j]) {
          std::string tmp = tmp_dim[i];
          tmp_dim[i] = tmp_dim[j];
          tmp_dim[j] = tmp;
        }
      }
    }
    for (int i = 0; i < len; ++i) {
      params.output_labels_idx[tmp_dim[i]] = i;
      params._output_msg += tmp_dim[i];
    }
    params.output_dims.emplace_back(tmp_dim);
  }

  params.output_size = output_idx;
  if (!has_elli) {
    params.elli_pos = output_idx;
    output_idx += params.elli_len;
  }
  for (auto it = params.num_labels.begin(); it != params.num_labels.end(); ++it) {
    if (it->second > 0 && params.output_labels_idx.find(it->first) == params.output_labels_idx.end()) {
      params.output_labels_idx[it->first] = output_idx++;
    }
  }
  params.num_output_labels = output_idx;
  return std::move(params);
}
 

EinsumParameters EinsumGradientParseMsg(const TensorList& inputs,
                                        std::string cmd, size_t outdim) {
  EinsumParameters param;
  param._msg = cmd;
  param.input_dims = {};
  param.output_dims = {};
  param._input_msgs = {};
  param._output_msg = {};
  param.undefined_labels = {};
  param.num_labels = {};
  param.output_labels_idx = {};
  param.input_elli_len = {};
  param._input_msgs.resize(0);
  size_t pos = cmd.find("->");
  std::string input_cmd = cmd.substr(0, pos);
  std::string output_cmd;
  if (pos != cmd.npos)
    output_cmd = cmd.substr(pos + 2);
  else
    output_cmd = "";
  param._output_msg = output_cmd;
  int num_ellipsis = 0;
  int i = 0;
  bool flag = true;
  while (flag) {
    size_t mid = input_cmd.find(",");
    std::string tmp_input;
    if (mid == input_cmd.npos) {
      tmp_input = input_cmd;
      flag = false;
    } else {
      tmp_input = input_cmd.substr(0, mid);
      input_cmd = input_cmd.substr(mid + 1);
    }
    param._input_msgs.emplace_back(tmp_input);
    // special signal for constant output
    if (tmp_input == "") {
      tmp_input = "$";
    }
    int input_len = tmp_input.length();
    num_ellipsis = 0;
    i = 0;
    OpDim tmp_dim = {};
    while (i < input_len) {
      switch (tmp_input[i]) {
        case '.':
          HT_ASSERT(num_ellipsis == 0 && tmp_input[i + 1] == '.' &&
                    tmp_input[i + 2] == '.')
            << "Invalid command.More than one ellipsis";
          num_ellipsis++;
          tmp_dim.emplace_back("...");
          i += 3;
          break;
        case ' ': i++; break;
        default: tmp_dim.emplace_back(tmp_input.substr(i, 1)); i++;
      }
    }
    param.input_dims.emplace_back(tmp_dim);
  }
  HT_ASSERT(param.input_dims.size() == inputs.size())
    << "The number of Tensors is incorrect, expected " << inputs.size()
    << ", got " << param.input_dims.size();

  param.elli_len = 0;
  param.input_elli_len.resize(0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    HTShape input_shape;
    if (inputs[i]->has_shape())
      input_shape = inputs[i]->shape();
    else {
      HT_LOG_ERROR << "wrong.";
    }
    OpDim input_dim = param.input_dims.at(i);
    int ndims = input_shape.size();
    int nlabels = input_dim.size();
    bool has_ellipsis = false;
    for (const auto& dim_label : input_dim) {
      if (dim_label == "...") {
        nlabels--;
        has_ellipsis = true;
        param.input_elli_len.emplace_back(ndims - nlabels);
        param.elli_len = std::max(param.elli_len, ndims - nlabels);
      } else {
        if (param.num_labels.find(dim_label) != param.num_labels.end()) {
          param.num_labels[dim_label]++;
        } else {
          param.num_labels.emplace(dim_label, 1);
        }
      }
    }
    if (has_ellipsis) {
      HT_ASSERT(nlabels <= ndims)
        << "num of dims is not equal to num of labels.";
    } else {
      HT_ASSERT(nlabels == ndims)
        << "num of dims is not equal to num of labels."
        << nlabels << "vs" << ndims << " of input" << i;
    }
  }

  int output_idx = 0;
  int repeat_dims = 0;
  param.elli_pos = 0;
  bool has_elli = false;
  if (pos != cmd.npos) {
    std::string tmp_output;
    tmp_output = output_cmd;
    int output_len = tmp_output.length();
    num_ellipsis = 0;
    i = 0;
    int out_labels = 0;
    while (i < output_len) {
      switch (tmp_output[i]) {
        case '.':
          HT_ASSERT(num_ellipsis == 0 && tmp_output[i + 1] == '.' &&
                    tmp_output[i + 2] == '.')
            << "Invalid command.more than one ellipsis";
          i += 3;
          break;
        case ' ': i++; break;
        default: out_labels++; i++;
      }
    }
    int output_ellis = outdim - out_labels;
    i = 0;
    OpDim tmp_dim = {};
    param.undefined_labels = {};
    // int label_idx = 0;
    while (i < output_len) {
      switch (tmp_output[i]) {
        case '.':
          HT_ASSERT(num_ellipsis == 0 && tmp_output[i + 1] == '.' &&
                    tmp_output[i + 2] == '.')
            << "Invalid command.More than one ellipsis";
          num_ellipsis++;
          tmp_dim.emplace_back("...");
          i += 3;
          param.elli_pos = output_idx;
          output_idx += param.elli_len;
          if (output_ellis > param.elli_len) {
            for (int idx = 0; idx < output_ellis - param.elli_len; ++idx) {
              std::string label = "UNDEFINED" + std::to_string(idx);
              param.undefined_labels.emplace(label, 0);
              param.output_labels_idx.emplace(label, output_idx++);
              tmp_dim.emplace_back(label);
            }
          }
          has_elli = true;
          break;
        case ' ': i++; break;
        default:
          HT_ASSERT(std::isalpha(tmp_output[i]))
            << "Invalid command.Use invalid charactor.Charactors should in [a-zA-Z].";
          std::string label = tmp_output.substr(i, 1);
          if (param.num_labels.find(label) == param.num_labels.end()) {
            param.undefined_labels.emplace(label, 0);
          }
          if (param.output_labels_idx.find(label) == param.output_labels_idx.end()) {
            param.output_labels_idx.emplace(label, output_idx++);
          } else {
            repeat_dims++;
          }
          tmp_dim.emplace_back(tmp_output.substr(i, 1));
          i++;
      }
    }
    param.output_dims.emplace_back(tmp_dim);
  } else {
    output_idx = param.elli_len;
    has_elli = true;
    for (auto it = param.num_labels.begin(); it != param.num_labels.end(); ++it) {
      if (it->second == 1) {
        param.output_labels_idx[it->first] = output_idx++;
      }
    }
  }
  param.output_size = output_idx;
  if (!has_elli) {
    param.elli_pos = output_idx;
    output_idx += param.elli_len;
    if (param.elli_len > 0)
      param.output_dims.at(0).emplace_back("...");
  }

  for (auto it = param.num_labels.begin(); it != param.num_labels.end(); ++it) {
    if (it->second > 0 &&
        param.output_labels_idx.find(it->first) == param.output_labels_idx.end()) {
      param.output_labels_idx[it->first] = output_idx++;
    }
  }

  param.num_output_labels = output_idx;
  return std::move(param);
}


Tensor MakeEinsumOp(const std::string& msg, TensorList inputs,
                    OpMeta op_meta) {
  for (auto& input: inputs) {
    HT_ASSERT(input->has_shape());
  }
  auto params = EinsumParseMsg(inputs, msg);
  return Graph::MakeOp(
          std::make_shared<EinsumOpImpl>(params),
          std::move(inputs),
          std::move(op_meta))->output(0);  
}

Tensor MakeEinsumGradientOp(const std::string& msg, TensorList inputs,
                            Tensor ori_output, Tensor ori_input,
                            OpMeta op_meta) {
  auto params = EinsumGradientParseMsg(inputs, msg, ori_input->ndim());
  inputs.emplace_back(ori_input);
  return Graph::MakeOp(
          std::make_shared<EinsumGradientOpImpl>(params),
          std::move(inputs),
          std::move(op_meta))->output(0);  
}

} // namespace graph
} // namespace hetu
