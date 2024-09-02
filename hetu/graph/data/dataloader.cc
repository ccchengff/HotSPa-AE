#include "hetu/graph/operator.h"
#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/data/dataloader.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>
#include <numeric>

namespace hetu {
namespace graph {

void Dataloader::init_states() {
  auto& inst_ctx = instantiation_ctx();
  inst_ctx.placement = Device(kCPU, 0);
  inst_ctx.stream_index = 0;
  inst_ctx.start[0] = std::make_unique<hetu::impl::CPUEvent>();
  inst_ctx.stop[0] = std::make_unique<hetu::impl::CPUEvent>();

  if (_dp_nrank != -1) {
    int cur_size = _data->shape(0);
    int start = cur_size * _dp_rank;
    int ending = start + cur_size;
    _data = reshape_tensor(start, ending);
  }
  _samples_num = _data->shape(0);
  _queue_size = 3;
  _batch_size = std::min(_batch_size, _samples_num / _queue_size);
  HT_ASSERT(_batch_size > 0) << "Invalid batch size.";
  _batch_num = _drop_last ? (_samples_num / _batch_size)
                          : std::ceil(double(_samples_num) / _batch_size);
  _shape = {};
  _shape.emplace_back(_batch_size);
  for (size_t i = 1; i < _data->ndim(); ++i) {
    _shape.emplace_back(_data->shape(i));
  }
  _seq = {};
  _seq.resize(_samples_num);
  _index = 0;
  _arrs = {};
  processers = std::vector<std::future<void>>();
  _arr_map = {};
  _arrs.resize(_queue_size);
  processers.resize(_queue_size);
  hetu::impl::CPUStream cpu_stream(instantiation_ctx().stream());
  if (_shuffle) {
    shuffled.resize(_batch_num);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    std::random_shuffle(shuffled.begin(), shuffled.end());
    HT_LOG_INFO << shuffled;
  }
  for (int i = 0; i < _queue_size; ++i) {
    int next_idx = _index + _batch_size;
    int cur_idx = _index;
    if (_shuffle) {
      cur_idx = _batch_size * shuffled[i];
      next_idx = _batch_size * (shuffled[i] + 1);
    }
    processers[i] = cpu_stream.EnqueueTask(
                    [this, cur_idx, next_idx, i]() {
                      this->pre_load(cur_idx, next_idx, i);
                    },
                    "Preload");
    _index = next_idx;
    _arr_map[i] = i;
  }
  _max_key = _queue_size - 1;
  _min_key = 0;
  _batch_idx = 0;
}

void Dataloader::pre_load(int cur_index, int next_index, int temp_id) {
  if (next_index <= _samples_num) {
    _arrs[temp_id] = reshape_tensor(cur_index, next_index);
  } else {
    HT_ASSERT(!_drop_last);
    _arrs[temp_id] = reshape_tensor(cur_index, _samples_num);
  }
  // HT_LOG_INFO << temp_id << " " << _arrs[temp_id];
}

NDArray Dataloader::_get_arr(int batch_idx) {
  int temp_id = _arr_map[_min_key];
  // HT_LOG_INFO << batch_idx << "," << _min_key << "," << temp_id;
  if (processers[temp_id].valid())
    processers[temp_id].wait();
  HT_ASSERT(_arr_map.find(batch_idx) != _arr_map.end());
  _max_key = (_max_key + 1) % _batch_num;
  if ((_index >= _samples_num) ||
      (_drop_last && _index + _batch_size > _samples_num)) {
    _index = 0;
  }
  int cur_index = _index;
  int next_index = _index + _batch_size;
  HT_LOG_INFO << _shuffle;
  if (_shuffle) {
    HT_LOG_INFO << shuffled[batch_idx];
    cur_index = _batch_size * shuffled[(batch_idx + _queue_size) % _batch_num];
    next_index = _batch_size * (shuffled[(batch_idx + _queue_size) % _batch_num] + 1);
  }
  _index = next_index;
  _arr_map[_max_key] = temp_id;
  NDArray res = _arrs[_arr_map[batch_idx]];
  hetu::impl::CPUStream cpu_stream(instantiation_ctx().stream());
  HT_LOG_INFO << temp_id << " " <<  batch_idx <<" "<< cur_index << " " << next_index << " " << res;
  processers[temp_id] = cpu_stream.EnqueueTask(
  [this, cur_index, next_index, temp_id]() {
    this->pre_load(cur_index, next_index, temp_id);
  },
  "Preload");
  _min_key = (_min_key + 1) % _batch_num;
  return std::move(res);
}

Tensor Dataloader::get_arr() {
  if (_batch_idx == _batch_num)
    return Tensor();
  NDArray res = _get_arr(_batch_idx);
  _last_batch_size = res->shape(0); 
  _batch_idx = (_batch_idx + 1) ;
  return MakeVariableOp(res, false, res->meta().dtype, false, DistributedStatesHierarchy(), OpMeta().set_eager_device(res->meta().device));
}

Tensor Dataloader::get_next_arr() {
  NDArray res = _get_arr(_batch_idx);
  return MakeVariableOp(res, false, res->meta().dtype, false, DistributedStatesHierarchy(), OpMeta().set_eager_device(res->meta().device));
}

void Dataloader::set_dp_rank(int dp_rank, int dp_nrank) {
  if (_dp_nrank != -1) {
    HT_ASSERT(dp_rank == _dp_rank);
    HT_ASSERT(dp_nrank == _dp_nrank);
  }
  _dp_rank = dp_rank;
  _dp_nrank = dp_nrank;
}

void Dataloader::set_mp_parts(int cur_part, int parts) {
  if (_parts != -1) {
    HT_ASSERT(cur_part == _cur_part);
    HT_ASSERT(parts == _parts);
  }
  _cur_part = cur_part;
  _parts = parts;
}

NDArray Dataloader::reshape_tensor(int begin_pos, int end_pos) {
  NDArrayMeta meta = _data->meta();
  int size = std::min(end_pos, (int) _data->numel()) - begin_pos;
  int stride = 1;
  HTShape res_shape = {size};
  HTShape shape = _data->shape();
  for (size_t i = 1; i < shape.size(); ++i) {
    stride *= shape[i];
  }
  for (size_t i = 1; i < shape.size(); ++i) {
    res_shape.emplace_back(shape[i]);
  }
  meta.set_shape(res_shape);
  NDArray res(meta, _data->storage(),
              _data->storage_offset() + stride * begin_pos);
  return res;
}

HTShape Dataloader::get_cur_shape() {
  return _arrs[_arr_map[_batch_idx]]->shape();
}

} // namespace autograd
} // namespace hetu
