#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/common/macros.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/autograd/runtime_context.h"
#include "hetu/autograd/operator.h"
#include "math.h"
#include "hetu/autograd/threadpool.h"

namespace hetu {
namespace autograd {

class DataloaderDef;
class Dataloader;
using DataloaderName = std::string;
class DataloaderOpDef;
class DataloaderOp;
using Dataloaders = std::vector<Dataloader>;
using Dataloaderinfo = std::unordered_map<DataloaderName, Dataloader>;

class DataloaderDef : public shared_ptr_target {
 public:
  DataloaderDef(NDArray& raw_data, int batch_size,
                DataloaderName name = "default", bool shuffle = false,
                bool drop_last = true) {
    _data = raw_data;
    _batch_size = batch_size;
    _name = name;
    _drop_last = drop_last;
    _dp_rank = -1;
    _dp_nrank = -1;
  }

  ~DataloaderDef() {
    if (_pool.is_started()) {
      _pool.Stop();
    }
  }

  void init_states();

  void pre_load(int cur_index, int next_index, int min_key, int max_key);

  NDArray _get_arr(int batch_idx);

  NDArray get_arr();

  NDArray get_next_arr();

  void set_dp_rank(int dp_rank, int dp_nrank);

  void set_mp_parts(int cur_part, int parts);

  NDArray reshape_tensor(int begin_pos, int end_pos);

  HTShape get_cur_shape();

  int batch_num() const {
    return _batch_num;
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

 protected:
  ThreadPool _pool;

  std::mutex _global_mutex;

  NDArray _data;

  int _batch_size;

  DataloaderName _name;

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
};

class Dataloader : shared_ptr_wrapper<DataloaderDef> {
 public:
  Dataloader(NDArray& raw_data, int batch_size, DataloaderName name = "default",
             bool drop_last = true)
  : shared_ptr_wrapper<DataloaderDef>() {
    this->_ptr = make_ptr<DataloaderDef>(raw_data, batch_size, name, drop_last);
    HT_ASSERT(this->_ptr) << "Passing a nullptr of OpDef "
                          << "to the constructor of Operator is not allowed. "
                          << "If you wish to declare an empty operator, "
                          << "call Operator() instead";
  }

  std::shared_ptr<DataloaderDef> operator->() const noexcept {
    return _ptr;
  }
};

class DataloaderOpDef;
class DataloaderOp;

class DataloaderOpDef : public OperatorDef {
 private:
  friend class DataloaderOp;
  struct constrcutor_access_key {};

 public:
  DataloaderOpDef(const constrcutor_access_key&, Dataloaders dataloaders,
                  const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(DataloaderOp), TensorList(), op_meta) {
    int len = dataloaders.size();
    for (int i = 0; i < len; ++i) {
      _dataloaders.insert(std::pair<DataloaderName, Dataloader>(
        dataloaders.at(i)->name(), dataloaders.at(i)));
      HT_ASSERT_NE(dataloaders.at(i)->dtype(), kUndeterminedDataType)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Data type is not prodived and cannot be inferred.";
      if (!dataloaders.at(i)->data().is_defined()) {
        AddOutput(
          NDArrayMeta().set_dtype(dataloaders.at(i)->dtype()).set_shape({}));
      } else {
        AddOutput(dataloaders.at(i)->data()->meta());
      }
    }
  }

  Dataloader dataloaders(DataloaderName name) {
    return _dataloaders.find(name)->second;
  }

  Dataloaderinfo dataloaders() {
    return _dataloaders;
  }

  uint64_t op_indicator() const noexcept {
    return DATA_LOADER_OP;
  }

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  void set_dp_rank(int dp_rank, int dp_nrank);

  void set_mp_parts(int cur_part, int parts);

  int get_batch_num(DataloaderName name);

  NDArray get_arr(DataloaderName name);

  NDArray get_next_arr(DataloaderName name);

  HTShape get_cur_shape(DataloaderName name);

 protected:
  Dataloaderinfo _dataloaders;
};

class DataloaderOp final : public OpWrapper<DataloaderOpDef> {
 public:
  DataloaderOp(Dataloaders dataloaders, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DataloaderOpDef>(make_ptr<DataloaderOpDef>(
      DataloaderOpDef::constrcutor_access_key(), dataloaders, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
