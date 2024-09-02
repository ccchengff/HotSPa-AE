#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/common/macros.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/graph/operator.h"
#include "math.h"

namespace hetu {
namespace graph {

class Dataloader;
using DataloaderName = std::string;
using Dataloaders = std::vector<Dataloader>;
using Dataloaderinfo = std::unordered_map<DataloaderName, Dataloader>;

class Dataloader{
 public:
  Dataloader() = default;
  Dataloader(NDArray raw_data, int batch_size, int num_workers = 0,
             DataloaderName name = "default", bool shuffle = false,
             bool drop_last = true):
  _data(std::move(raw_data)),
  _num_workers(num_workers),
  _batch_size(batch_size),
  _name(name),
  _shuffle(shuffle),
  _drop_last(drop_last),
  _dp_rank(-1),
  _dp_nrank(-1) {
    init_states();
  }

  ~Dataloader() {
    int len = processers.size();
    for (int i = 0; i < len; ++i) {
      if (processers[len - 1 - i].valid())
        processers[len - 1 - i].wait();
      processers.pop_back();
    }
  }

  Dataloader(Dataloader&& resource) {
    _data = std::move(resource._data);
    _num_workers = resource._num_workers;
    _batch_size = resource._batch_size;
    _name = resource._name;
    _shuffle = resource._shuffle;
    shuffled = std::move(resource.shuffled);
    _drop_last = resource._drop_last;
    _dp_rank = -1;
    _dp_nrank = -1;
    _name = resource._name;
    _samples_num = resource._samples_num;
    _queue_size = resource._queue_size;
    _batch_num = resource._batch_num;
    _shape = std::move(resource._shape);
    _seq = std::move(resource._seq);
    _index = resource._index;
    _max_key = resource._max_key;
    _min_key = resource._min_key;
    _arrs = std::move(resource._arrs);
    _arr_map = std::move(resource._arr_map);
    _batch_idx = resource._batch_idx;
    _last_batch_size = resource._last_batch_size; 
    _cur_part = resource._cur_part;
    _parts = resource._parts;
    auto& inst_ctx = instantiation_ctx();
    inst_ctx.placement = Device(kCPU, 0);
    inst_ctx.stream_index = 0;
    inst_ctx.start[0] = std::make_unique<hetu::impl::CPUEvent>();
    inst_ctx.stop[0] = std::make_unique<hetu::impl::CPUEvent>();
    processers = std::vector<std::future<void>>();
    processers.resize(resource.processers.size());
    for (int i = 0; i < _queue_size; ++i) {
      processers[i] = std::move(resource.processers[i]);
    }
    resource.~Dataloader();
  }

  Dataloader(const Dataloader& resource) {
    _data = resource._data;
    _num_workers = resource._num_workers;
    _batch_size = resource._batch_size;
    _name = resource._name;
    _shuffle = resource._shuffle;
    _drop_last = resource._drop_last;
    _dp_rank = -1;
    _dp_nrank = -1;
    init_states();
  }

  Dataloader& operator=(const Dataloader& resource) {
    _data = resource._data;
    _num_workers = resource._num_workers;
    _batch_size = resource._batch_size;
    _name = resource._name;
    _shuffle = resource._shuffle;
    _drop_last = resource._drop_last;
    _dp_rank = -1;
    _dp_nrank = -1;
    init_states();
  }

  void init_states();

  void pre_load(int cur_index, int next_index, int temp_id);

  NDArray _get_arr(int batch_idx);

  Tensor get_arr();

  Tensor get_next_arr();

  void set_dp_rank(int dp_rank, int dp_nrank);

  void set_mp_parts(int cur_part, int parts);

  NDArray reshape_tensor(int begin_pos, int end_pos);

  HTShape get_cur_shape();

  int batch_num() const {
    return _batch_num;
  }

  int sample_num() const {
    return _samples_num;
  }

  int batch_size() const {
    return _batch_size;
  }

  int num_workers() const {
    return _num_workers;
  }

  DataType dtype() const {
    return _data->dtype();
  }

  DataloaderName name() const {
    return _name;
  }

  NDArray data() const {
    return _data;
  }

  const OpInstantiationContext& instantiation_ctx() const {
    return _inst_ctx;
  }

  OpInstantiationContext& instantiation_ctx() {
    return _inst_ctx;
  }

  void Sync() {
    instantiation_ctx().stop[0]->Sync();
  }

 protected:

  NDArray _data;

  int _batch_size;

  int _num_workers;

  DataloaderName _name;

  bool _shuffle;

  bool _drop_last;

  int _samples_num;

  int _queue_size;

  int _batch_num;

  HTShape _shape;

  HTShape _seq;

  int _index;

  int _max_key;

  int _min_key;

  NDArrayList _arrs;

  std::unordered_map<int, int> _arr_map;

  int _batch_idx;

  int _rest;

  int _last_batch_size;

  int _dp_rank;

  int _dp_nrank;

  int _cur_part;

  int _parts;

  OpInstantiationContext _inst_ctx;

  std::vector<std::future<void>> processers;

  std::vector<int> shuffled;
};


} // namespace autograd
} // namespace hetu
