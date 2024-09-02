#include "hetu/autograd/operator.h"
#include "hetu/autograd/ops/DataTransfer.h"
#include "hetu/autograd/ops/Group.h"
#include "hetu/autograd/dataloader.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <cmath>

namespace hetu {
namespace autograd {

void DataloaderDef::init_states() {
  if (_dp_nrank != -1) {
    int cur_size = _data->shape(0);
    int start = cur_size * _dp_rank;
    int ending = start + cur_size;
    _data = reshape_tensor(start, ending);
  }
  _samples_num = _data->shape(0);
  _queue_size = 3;
  _pool.Start(_queue_size);
  _batch_size = std::min(_batch_size, _samples_num / _queue_size);
  HT_ASSERT(_batch_size > 0) << "Invalid batch size.";
  _batch_num = _drop_last ? (_samples_num / _batch_size)
                          : std::ceil(_samples_num / _batch_size);
  _shape = {};
  _shape.emplace_back(_batch_size);
  for (size_t i = 1; i < _data->ndim(); ++i) {
    _shape.emplace_back(_data->shape(i));
  }
  _seq = {};
  _seq.resize(_samples_num);
  _index = 0;
  _arrs = {};
  _arr_map = {};
  for (int i = 0; i < _queue_size; ++i) {
    int next_idx = _index + _batch_size;
    _arrs.emplace_back(reshape_tensor(_index, next_idx));
    _index = next_idx;
    _arr_map[i] = i;
  }
  _max_key = _queue_size - 1;
  _min_key = 0;
  if (!_drop_last) {
    int res_num = _samples_num % _batch_size;
    if (res_num > 0) {
      _arrs.emplace_back(NDArray());
    }
    _rest = _queue_size;
  }
  _batch_idx = 0;
}

void DataloaderDef::pre_load(int cur_index, int next_index, int min_key,
                             int max_key) {
  if (next_index <= _samples_num) {
    int temp_ind = _arr_map[min_key];
    //_arr_map.erase(_min_key);
    if (temp_ind == _queue_size && !_drop_last) {
      temp_ind = _rest;
      _rest = _queue_size;
    }
    _arr_map[max_key] = temp_ind;
    _arrs[temp_ind] = reshape_tensor(cur_index, next_index);
  } else {
    HT_ASSERT(!_drop_last);
    _arrs[_arrs.size() - 1] = reshape_tensor(cur_index, next_index);
    _rest = _arr_map[min_key];
    //_arr_map.erase(_min_key);
    _arr_map[max_key] = _queue_size;
  }
}

NDArray DataloaderDef::_get_arr(int batch_idx) {
  HT_ASSERT(_arr_map.find(batch_idx) != _arr_map.end());
  NDArray res = _arrs[_arr_map[batch_idx]];
  //_global_mutex.lock();
  if (batch_idx > _min_key) {
    _max_key = (_max_key + 1) % _samples_num;
    _min_key = (_min_key + 1) % _samples_num;
    if ((_index >= _samples_num) ||
        (_drop_last && _index + _batch_size > _samples_num)) {
      _index = 0;
    }
    int cur_index = _index;
    int next_index = _index + _batch_size;
    _index = next_index;
    //_global_mutex.unlock();
    _pool.AddTask(std::bind(&DataloaderDef::pre_load, this,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, std::placeholders::_4),
                  cur_index, next_index, _min_key, _max_key);
  }
  // else
  //   _global_mutex.unlock();
  return res;
}

NDArray DataloaderDef::get_arr() {
  NDArray res = _get_arr(_batch_idx);
  _last_batch_size = res->shape(0);
  _batch_idx = (_batch_idx + 1) % _samples_num;
  return res;
}

NDArray DataloaderDef::get_next_arr() {
  NDArray res = _get_arr(_batch_idx);
  return res;
}

void DataloaderDef::set_dp_rank(int dp_rank, int dp_nrank) {
  if (_dp_nrank != -1) {
    HT_ASSERT(dp_rank == _dp_rank);
    HT_ASSERT(dp_nrank == _dp_nrank);
  }
  _dp_rank = dp_rank;
  _dp_nrank = dp_nrank;
}

void DataloaderDef::set_mp_parts(int cur_part, int parts) {
  if (_parts != -1) {
    HT_ASSERT(cur_part == _cur_part);
    HT_ASSERT(parts == _parts);
  }
  _cur_part = cur_part;
  _parts = parts;
}

NDArray DataloaderDef::reshape_tensor(int begin_pos, int end_pos) {
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

HTShape DataloaderDef::get_cur_shape() {
  return _arrs[_arr_map[_batch_idx]]->shape();
}

TensorList DataloaderOpDef::DoGradient(const TensorList& grad_outputs) {
  return {};
}

HTShapeList DataloaderOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {};
}

void DataloaderOpDef::set_dp_rank(int dp_rank, int dp_nrank) {
  for (auto it = dataloaders().begin(); it != dataloaders().end(); ++it) {
    it->second->set_dp_rank(dp_rank, dp_nrank);
  }
}

void DataloaderOpDef::set_mp_parts(int cur_part, int parts) {
  for (auto it = dataloaders().begin(); it != dataloaders().end(); ++it) {
    it->second->set_mp_parts(cur_part, parts);
  }
}

int DataloaderOpDef::get_batch_num(DataloaderName name) {
  return dataloaders(name)->batch_num();
}

NDArray DataloaderOpDef::get_arr(DataloaderName name) {
  return dataloaders(name)->get_arr();
}

NDArray DataloaderOpDef::get_next_arr(DataloaderName name) {
  return dataloaders(name)->get_next_arr();
}

HTShape DataloaderOpDef::get_cur_shape(DataloaderName name) {
  return dataloaders(name)->get_cur_shape();
}

} // namespace autograd
} // namespace hetu
