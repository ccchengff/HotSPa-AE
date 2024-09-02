#include "hetu/autograd/ops/Einsum.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <bitset>

namespace hetu {
namespace autograd {

NDArray sumproduct_pair(NDArray& left_, NDArray& right_, HTShape sum_dims_,
                        bool keepdim) {
  // assumes that tensors have been pre-unsqueezed (so that all dimensions match
  // - after broadcasting) but makes no other assumptions on the order of
  // dimensions
  HT_ASSERT(left_->ndim() == right_->ndim())
    << "number of dimensions must match";
  if (sum_dims_.size() == 0)
    return NDArray::mul(left_, right_);
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
  NDArray left = NDArray::copy(left_);
  NDArray right = NDArray::copy(right_);
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
        left = NDArray::sum(left, {i}, {true});
      } else if (sr) {
        right = NDArray::sum(right, {i}, {true});
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

  // we now work with the following permutations / shapes.
  // the pipeline is permute inputs -> reshape inputs -> batch matrix mul ->
  // reshape(view) output -> permute output output: "lro, lo, 1-for-summed-dims,
  // ro" with orgiginal shape dimensions left: "lro, lo, summed" permuted with
  // lpermutation and the three flattened right:  "lro, summed, ro" permuted
  // with rpermutation and the three flattened then the permuted output is a
  // view of bmm(left, right) finally, opermutation reverts the permutation to
  // the original order of dimensions
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
  left = NDArray::permute(left, lpermutation);
  HTShape ls(3);
  ls[0] = lro_size;
  ls[1] = lo_size;
  ls[2] = sum_size;

  left = NDArray::reshape(left, ls);

  right = NDArray::permute(right, rpermutation);
  HTShape rs(3);
  rs[0] = lro_size;
  rs[1] = sum_size;
  rs[2] = ro_size;

  right = NDArray::reshape(right, rs);

  NDArray result = NDArray::bmm(left, right, false, false);
  HTShape os(out_size.size());
  for (size_t i = 0; i < out_size.size(); ++i) {
    os[i] = out_size[i];
  }

  result = NDArray::view(result, os);
  result = NDArray::permute(result, opermutation);

  // finally squeeze summed dimensions if desired
  if (!keepdim) {
    HTShape sizes = result->shape();
    for (int i = dim - 1; i >= 0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }

    result = NDArray::view(result, sizes);
  }
  return result;
}

void EinsumOpDef::ParseMsg() {
  input_dims = {};
  output_dims = {};
  std::string cmd = fetch_msg();
  _input_msgs.resize(0);
  size_t pos = cmd.find("->");
  std::string input_cmd = cmd.substr(0, pos);
  std::string output_cmd;
  if (pos != cmd.npos)
    output_cmd = cmd.substr(pos + 2);
  else
    output_cmd = "";
  _output_msg = output_cmd;
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
    _input_msgs.emplace_back(tmp_input);
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
    input_dims.emplace_back(tmp_dim);
  }
  HT_ASSERT(input_dims.size() == num_inputs())
    << "The number of tensors is incorrect, expected " << num_inputs()
    << ", got " << input_dims.size();

  elli_len = 0;
  input_elli_len = std::vector<int>(num_inputs(), -1);
  for (size_t i = 0; i < num_inputs(); ++i) {
    HTShape input_shape = _inputs[i]->shape();
    OpDim input_dim = input_dims.at(i);
    int ndims = input_shape.size();
    int nlabels = input_dim.size();
    bool has_ellipsis = false;
    for (const auto& dim_label : input_dim) {
      if (dim_label == "...") {
        nlabels--;
        has_ellipsis = true;
        input_elli_len[i] = ndims - nlabels;
        elli_len = std::max(elli_len, ndims - nlabels);
      } else {
        if (num_labels.find(dim_label) != num_labels.end()) {
          num_labels[dim_label]++;
        } else {
          num_labels.emplace(dim_label, 1);
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
  elli_pos = 0;
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
          elli_pos = output_idx;
          output_idx += elli_len;
          has_elli = true;
          break;
        case ' ': i++; break;
        default:
          HT_ASSERT(std::isalpha(tmp_output[i]))
            << "Invalid command.Use invalid charactor.Charactors should in [a-zA-Z].";
          std::string label = tmp_output.substr(i, 1);
          HT_ASSERT(num_labels.find(label) != num_labels.end())
            << "This label occurs didn't occur in inputs:" << label;
          HT_ASSERT(output_labels_idx.find(label) == output_labels_idx.end())
            << "This label occurs more than once in output:" << label;
          output_labels_idx.emplace(label, output_idx++);
          tmp_dim.emplace_back(tmp_output.substr(i, 1));
          i++;
      }
    }
    output_dims.emplace_back(tmp_dim);
  } else {
    output_idx = elli_len;
    has_elli = true;
    _output_msg = "";
    OpDim tmp_dim = {};
    if (elli_len > 0) {
      _output_msg += "...";
      tmp_dim.emplace_back("...");
    }
    for (auto it = num_labels.begin(); it != num_labels.end(); ++it) {
      if (it->second == 1) {
        output_labels_idx[it->first] = output_idx++;
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
      output_labels_idx[tmp_dim[i]] = i;
      _output_msg += tmp_dim[i];
    }
    output_dims.emplace_back(tmp_dim);
  }

  output_size = output_idx;
  if (!has_elli) {
    elli_pos = output_idx;
    output_idx += elli_len;
  }
  for (auto it = num_labels.begin(); it != num_labels.end(); ++it) {
    if (it->second > 0 &&
        output_labels_idx.find(it->first) == output_labels_idx.end()) {
      output_labels_idx[it->first] = output_idx++;
    }
  }
  num_output_labels = output_idx;
}

void EinsumOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  NDArrayList permuted_inputs = {};
  for (size_t i = 0; i < num_inputs(); ++i) {
    HTShape perm_shape(num_output_labels, -1);
    LabelMap label_dim;
    OpDim input_labels;
    HTShape input_shape;
    NDArray input_tensor = NDArray::copy(inputs.at(i));
    input_labels = input_dims[i];
    input_shape = inputs.at(i)->shape();

    int j = 0;
    for (const auto& label : input_labels) {
      if (label == "...") {
        // Add missing dimensions covered by the ellipsis
        int missing_dims =
          elli_len - (input_shape.size() - input_labels.size() + 1);
        for (int k = 0; k < missing_dims; ++k) {
          input_tensor = NDArray::unsqueeze(input_tensor, j);
        }
        for (int k = 0; k < elli_len; ++k) {
          perm_shape[elli_pos + k] = j++;
        }
      } else if (label_dim.find(label) != label_dim.end()) {
        // Repeated label, take diagonal
        int dim = label_dim[label];
        HT_ASSERT(input_tensor->shape(j) == input_tensor->shape(dim))
          << j << ":" << input_tensor->shape(j) << "," << dim << ":"
          << input_tensor->shape(dim);

        input_tensor = NDArray::diagonal(input_tensor, dim, j);

        input_tensor = NDArray::movedim(input_tensor, -1, dim);
      } else {
        // Lookup output index for label
        label_dim[label] = j;
        perm_shape[output_labels_idx[label]] = j++;
      }
    }

    // Add dimensions for missing labels
    for (int64_t& index : perm_shape) {
      if (index == -1) {
        input_tensor = NDArray::unsqueeze(input_tensor, -1);
        index = j++;
      }
    }
    NDArray permuted_input = NDArray::permute(input_tensor, perm_shape);
    permuted_inputs.emplace_back(permuted_input);
  }

  // Check if operands broadcast and keep track of last operand with
  // dimension size != 1 for optimizing reductions
  std::vector<std::size_t> dim_last_op(num_output_labels, 0);
  bool has_zero_size_dim = false;

  for (int dim = 0; dim < num_output_labels; dim++) {
    int output_dim_size = permuted_inputs[0]->shape(dim);
    for (size_t i = 1; i < num_inputs(); ++i) {
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
  NDArray output_tensor = NDArray::copy(permuted_inputs[0]);
  int dim = output_size;
  for (int i = dim; i < num_output_labels; ++i, ++dim) {
    if (dim_last_op[i] == 0) {
      if (output_tensor->shape(dim) == 1) {
        output_tensor = NDArray::squeeze(output_tensor, dim--);
      } else {
        output_tensor = NDArray::sum(output_tensor, {dim--}, {false});
      }
    }
  }
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed, start, stop);
  for (size_t i = 1; i < num_inputs(); ++i) {
    NDArray permuted_input = NDArray::copy(permuted_inputs[i]);
    HTShape sum_dims;
    // Sum out or squeeze dimensions that are size 1 for all later operands
    dim = output_size;
    for (int j = dim; j < num_output_labels; ++j, ++dim) {
      if (dim_last_op[j] < i) {
        permuted_input = NDArray::squeeze(permuted_input, dim--);
      } else if (dim_last_op[j] == i) {
        if (output_tensor->shape(dim) == 1) {
          permuted_input = NDArray::sum(permuted_input, {dim}, {false});
          output_tensor = NDArray::squeeze(output_tensor, dim--);
        } else {
          sum_dims.push_back(dim);
        }
      }
    }
    // Multiply tensors and sum out dimensions in sum_dims
    if (sum_dims.empty()) {
      output_tensor = NDArray::mul(output_tensor, permuted_input);
    } else if (sum_dims.size() == output_tensor->ndim()) {
      NDArray flatten_input = NDArray::flatten(permuted_input, 0, -1);
      NDArray flatten_output = NDArray::flatten(output_tensor, 0, -1);
      output_tensor =
        NDArray::sum(NDArray::dot(flatten_input, flatten_output), {0}, {false});
    } else {
      output_tensor =
        sumproduct_pair(output_tensor, permuted_input, sum_dims, false);
    }
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, output_tensor,
                                  outputs.at(0), stream());
}

TensorList EinsumOpDef::DoGradient(const TensorList& grad_outputs) {
  int len = num_inputs();
  TensorList grad_inputs = {};
  std::string ori_cmd = fetch_msg();
  auto g_op_meta = grad_op_meta();
  for (int i = 0; i < len; ++i) {
    std::string grad_msg;
    TensorList grad_oplinkers = {};
    for (int j = 0; j < len - 1; ++j) {
      if (i == j) {
        grad_msg = grad_msg + _output_msg + ",";
        grad_oplinkers.emplace_back(grad_outputs.at(0));
      } else {
        grad_msg = grad_msg + _input_msgs[j] + ",";
        grad_oplinkers.emplace_back(_inputs[j]);
      }
    }
    if (i == len - 1) {
      grad_msg = grad_msg + _output_msg + "->";
      grad_oplinkers.emplace_back(grad_outputs.at(0));
    } else {
      grad_msg = grad_msg + _input_msgs[len - 1] + "->";
      grad_oplinkers.emplace_back(_inputs[len - 1]);
    }
    grad_msg = grad_msg + _input_msgs[i];
    auto grad_input =
      EinsumGradientOp(grad_msg, grad_oplinkers, _outputs[0], _inputs[i],
                       g_op_meta.set_name(grad_name(i)))
        ->output(0);
    grad_inputs.emplace_back(grad_input);
  }
  return grad_inputs;
}

void EinsumOpDef::DoInferMeta() {
  ParseMsg();
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_device(_inputs[0]->device()));
}

HTShapeList EinsumOpDef::DoInferShape(const HTShapeList& input_shapes) {
  LabelMap label_to_size;
  HTShape output_shape(output_size);
  std::vector<int> elli_size(elli_len, -1);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    int input_idx = 0;
    HTShape perm_shape(num_output_labels, 0);
    OpDim input_labels = input_dims[i];
    HTShape input_shape = input_shapes.at(i);
    for (const auto& label : input_labels) {
      if (label == "...") {
        if (input_elli_len[i] == elli_len) {
          for (int k = 0; k < elli_len; ++k) {
            if (elli_size[k] == -1) {
              elli_size[k] = input_shape[input_idx + k];
            } else {
              // HT_ASSERT(elli_size[k] == input_shape[input_idx + k]);
            }
          }
        }
        input_idx += elli_len;
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
  if (output_dims.empty()) {
    output_shape = {1};
  } else {
    int output_idx = 0;
    for (const auto& label : output_dims.at(0)) {
      if (label == "...") {
        for (int k = 0; k < elli_len; ++k) {
          output_shape[elli_pos + k] = elli_size[k];
        }
        output_idx += elli_len;
      } else {
        output_shape[output_idx] = label_to_size[label];
        output_idx += 1;
      }
    }
    if (output_shape.size() == 0) {
      output_shape = {1};
    }
  }
  set_grad_shape(output_shape);
  return {output_shape};
}

void EinsumGradientOpDef::ParseMsg(const HTShapeList& input_shapes) {
  input_dims = {};
  output_dims = {};
  _input_msgs = {};
  _output_msg = {};
  undefined_labels = {};
  num_labels = {};
  output_labels_idx = {};
  input_elli_len = {};
  std::string cmd = fetch_msg();
  _input_msgs.resize(0);
  size_t pos = cmd.find("->");
  std::string input_cmd = cmd.substr(0, pos);
  std::string output_cmd;
  if (pos != cmd.npos)
    output_cmd = cmd.substr(pos + 2);
  else
    output_cmd = "";
  _output_msg = output_cmd;
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
    _input_msgs.emplace_back(tmp_input);
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
    input_dims.emplace_back(tmp_dim);
  }
  HT_ASSERT(input_dims.size() == num_inputs())
    << "The number of Tensors is incorrect, expected " << num_inputs()
    << ", got " << input_dims.size();

  elli_len = 0;
  input_elli_len.resize(0);
  for (size_t i = 0; i < num_inputs(); ++i) {
    HTShape input_shape;
    if (_inputs[i]->has_shape())
      input_shape = _inputs[i]->shape();
    else {
      input_shape = input_shapes.at(i);
    }
    OpDim input_dim = input_dims.at(i);
    int ndims = input_shape.size();
    int nlabels = input_dim.size();
    bool has_ellipsis = false;
    for (const auto& dim_label : input_dim) {
      if (dim_label == "...") {
        nlabels--;
        has_ellipsis = true;
        input_elli_len.emplace_back(ndims - nlabels);
        elli_len = std::max(elli_len, ndims - nlabels);
      } else {
        if (num_labels.find(dim_label) != num_labels.end()) {
          num_labels[dim_label]++;
        } else {
          num_labels.emplace(dim_label, 1);
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
  elli_pos = 0;
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
    int output_ellis = pred_in->ndim() - out_labels;
    i = 0;
    OpDim tmp_dim = {};
    undefined_labels = {};
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
          elli_pos = output_idx;
          output_idx += elli_len;
          if (output_ellis > elli_len) {
            for (int64_t idx = 0; idx < output_ellis - elli_len; ++idx) {
              std::string label = "UNDEFINED" + std::to_string(idx);
              undefined_labels.emplace(label, 0);
              output_labels_idx.emplace(label, output_idx++);
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
          if (num_labels.find(label) == num_labels.end()) {
            undefined_labels.emplace(label, 0);
          }
          if (output_labels_idx.find(label) == output_labels_idx.end()) {
            output_labels_idx.emplace(label, output_idx++);
          } else {
            repeat_dims++;
          }
          tmp_dim.emplace_back(tmp_output.substr(i, 1));
          i++;
      }
    }
    output_dims.emplace_back(tmp_dim);
  } else {
    output_idx = elli_len;
    has_elli = true;
    for (auto it = num_labels.begin(); it != num_labels.end(); ++it) {
      if (it->second == 1) {
        output_labels_idx[it->first] = output_idx++;
      }
    }
  }
  output_size = output_idx;
  ori_output_size = output_idx + repeat_dims;
  if (!has_elli) {
    elli_pos = output_idx;
    output_idx += elli_len;
    if (elli_len > 0)
      output_dims.at(0).emplace_back("...");
  }

  for (auto it = num_labels.begin(); it != num_labels.end(); ++it) {
    if (it->second > 0 &&
        output_labels_idx.find(it->first) == output_labels_idx.end()) {
      output_labels_idx[it->first] = output_idx++;
    }
  }

  num_output_labels = output_idx;
}

void EinsumGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  NDArrayList permuted_inputs = {};
  for (size_t i = 0; i < num_inputs(); ++i) {
    HTShape perm_shape(num_output_labels, -1);
    LabelMap label_dim;
    OpDim input_labels;
    HTShape input_shape;
    NDArray input_tensor;
    input_labels = input_dims[i];
    input_shape = inputs.at(i)->shape();
    input_tensor = NDArray::copy(inputs.at(i));

    int j = 0;
    for (const auto& label : input_labels) {
      if (label == "...") {
        // Add missing dimensions covered by the ellipsis
        int missing_dims =
          elli_len - (input_shape.size() - input_labels.size() + 1);
        for (int k = 0; k < missing_dims; ++k) {
          input_tensor = NDArray::unsqueeze(input_tensor, j);
        }
        for (int k = 0; k < elli_len; ++k) {
          perm_shape[elli_pos + k] = j++;
        }
      } else if (label_dim.find(label) != label_dim.end()) {
        // Repeated label, take diagonal
        int dim = label_dim[label];
        HT_ASSERT(input_shape[j] == input_shape[dim]);

        input_tensor = NDArray::diagonal(input_tensor, dim, j);

        input_tensor = NDArray::movedim(input_tensor, -1, dim);
      } else {
        // Lookup output index for label
        label_dim[label] = j;
        perm_shape[output_labels_idx[label]] = j++;
      }
    }

    // Add dimensions for missing labels
    for (int64_t& index : perm_shape) {
      if (index == -1) {
        input_tensor = NDArray::unsqueeze(input_tensor, -1);
        index = j++;
      }
    }
    NDArray permuted_input = NDArray::permute(input_tensor, perm_shape);
    permuted_inputs.emplace_back(permuted_input);
  }
  // Check if operands broadcast and keep track of last operand with
  // dimension size != 1 for optimizing reductions
  std::vector<std::size_t> dim_last_op(num_output_labels, 0);
  bool has_zero_size_dim = false;

  for (int dim = 0; dim < num_output_labels; dim++) {
    int output_dim_size = permuted_inputs[0]->shape(dim);
    for (size_t i = 1; i < num_inputs(); ++i) {
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
  NDArray output_tensor = NDArray::copy(permuted_inputs[0]);
  int dim = output_size;
  for (int i = dim; i < num_output_labels; ++i, ++dim) {
    if (dim_last_op[i] == 0) {
      if (output_tensor->shape(dim) == 1) {
        output_tensor = NDArray::squeeze(output_tensor, dim--);
        // output_tensor.squeeze_(dim--);
      } else {
        output_tensor = NDArray::sum(output_tensor, {dim--}, {false});
      }
    }
  }

  for (size_t i = 1; i < num_inputs(); ++i) {
    NDArray permuted_input = permuted_inputs[i];
    HTShape sum_dims;

    // Sum out or squeeze dimensions that are size 1 for all later operands
    dim = output_size;
    for (int j = dim; j < num_output_labels; ++j, ++dim) {
      if (dim_last_op[j] < i) {
        permuted_input = NDArray::squeeze(permuted_input, dim--);
      } else if (dim_last_op[j] == i) {
        if (output_tensor->shape(dim) == 1) {
          permuted_input = NDArray::sum(permuted_input, {dim}, {false});
          output_tensor = NDArray::squeeze(output_tensor, dim--);
        } else {
          sum_dims.push_back(dim);
        }
      }
    }
    // Multiply tensors and sum out dimensions in sum_dims
    if (sum_dims.empty()) {
      output_tensor = NDArray::mul(output_tensor, permuted_input);
    } else if (sum_dims.size() == output_tensor->ndim()) {
      NDArray flatten_input = NDArray::flatten(permuted_input, 0, -1);
      NDArray flatten_output = NDArray::flatten(output_tensor, 0, -1);
      output_tensor =
        NDArray::sum(NDArray::dot(flatten_input, flatten_output), {0}, {false});
    } else {
      output_tensor =
        sumproduct_pair(output_tensor, permuted_input, sum_dims, false);
    }
  }
  LabelMap first_output_idx;
  int output_idx = 0;
  for (const auto& label : output_dims.at(0)) {
    if (label == "...") {
      output_idx += elli_len;
    } else {
      if (first_output_idx.find(label) == first_output_idx.end()) {
        if (undefined_labels.find(label) != undefined_labels.end()) {
          output_tensor = NDArray::adddim(output_tensor, output_idx,
                                          outputs.at(0)->shape(output_idx));
        }
        first_output_idx.emplace(label, output_idx);
      } else {
        int first_idx = first_output_idx.find(label)->second;
        output_tensor =
          NDArray::diagonal_grad(output_tensor, first_idx, output_idx);
      }
      output_idx += 1;
    }
  }
  
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, output_tensor,
                                  outputs.at(0), stream());
}

void EinsumGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype())
                         .set_device(_inputs[0]->device())
                         .set_shape(pred_in->shape()));
}

HTShapeList EinsumGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  ParseMsg(input_shapes);
  return {pred_in->shape()};
}

} // namespace autograd
} // namespace hetu
