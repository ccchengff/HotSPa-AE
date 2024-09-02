#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include <numeric>
#include <mutex>

namespace hetu {
namespace impl {
namespace comm {

using hetu::operator<<;

DECLARE_HT_EXCEPTION(mpi_error);

#define MPI_CALL(f)                                                            \
  for (auto status = (f); status != MPI_SUCCESS; status = MPI_SUCCESS)         \
  __HT_FATAL_SILENT(hetu::impl::comm::mpi_error)                               \
    << "MPI call " << #f << " failed with status: " << std::to_string(status)

namespace {

inline MPI_Op to_MPI_Op(ReductionType red_type) {
  switch (red_type) {
    case kSUM: return MPI_SUM;
    case kPROD: return MPI_PROD;
    case kMAX: return MPI_MAX;
    case kMIN: return MPI_MIN;
    case kNONE:
      HT_NOT_IMPLEMENTED << "Reduction type cannot be none";
      __builtin_unreachable();
    default:
      HT_NOT_IMPLEMENTED << "Reduction type " << red_type
                         << " is not supported for MPI.";
      __builtin_unreachable();
  }
}

inline MPI_Datatype to_MPI_Datatype(DataType dtype) {
  switch (dtype) {
    case kUInt8: return MPI_UNSIGNED_CHAR;
    case kInt8: return MPI_CHAR;
    case kInt16: return MPI_SHORT;
    case kInt32: return MPI_INT;
    case kInt64: return MPI_LONG;
    case kFloat32: return MPI_FLOAT;
    case kFloat64: return MPI_DOUBLE;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for MPI.";
      __builtin_unreachable();
  }
}

inline int to_num_bytes(DataType dtype) {
  switch (dtype) {
    case kUInt8: return 1;
    case kInt8: return 1;
    case kInt16: return 2;
    case kInt32: return 4;
    case kInt64: return 8;
    case kFloat32: return 4;
    case kFloat64: return 8;
    default:
      HT_NOT_IMPLEMENTED << "Data type " << dtype
                         << " is not supported for MPI.";
      __builtin_unreachable();
  }
}

static std::once_flag mpi_init_flag;
static int mpi_world_rank = -1;
static int mpi_world_size = -1;
static std::mutex mpi_call_mutex;
static std::mutex mpi_create_group_mutex;
static std::vector<std::map<std::vector<int>, MPICommunicationGroup>>
  mpi_comm_groups((HT_NUM_STREAMS_PER_DEVICE) + 1);
static std::vector<std::once_flag>
  worldwide_mpi_comm_group_reg_flags((HT_NUM_STREAMS_PER_DEVICE) + 1);
// The worldwide groups would be excessively accessed. Cache them here.
static std::vector<MPICommunicationGroup>
  worldwide_mpi_comm_groups((HT_NUM_STREAMS_PER_DEVICE) + 1);

} // namespace

struct MPICallGuard {
  // MPI_THREAD_SERIALIZED requires all MPI calls are sequential,
  // so we need to lock on a global mutex here.
  MPICallGuard() : lock(mpi_call_mutex) {}
  std::lock_guard<std::mutex> lock;
};

static void MPI_Init_Once() {
  std::call_once(mpi_init_flag, []() {
    // init mpi
    int32_t mpi_provided;
    MPI_CALL(
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpi_provided));
    HT_ASSERT(mpi_provided >= MPI_THREAD_SERIALIZED)
      << "The installed MPI cannot support MPI_THREAD_SERIALIZED.";
    // get world rank and size
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
    HT_ASSERT(mpi_world_rank >= 0 && mpi_world_rank < mpi_world_size)
      << "Failed to get the world rank and/or size. "
      << "(Got rank " << mpi_world_rank << " and size " << mpi_world_size
      << ".)";
    // register exit handler
    HT_ASSERT(std::atexit([]() {
                MPICallGuard guard;
                HT_LOG_DEBUG << "Destructing MPI comm groups...";
                mpi_comm_groups.clear();
                worldwide_mpi_comm_groups.clear();
                MPI_CALL(MPI_Finalize());
                HT_LOG_DEBUG << "Destructed MPI comm groups";
              }) == 0)
      << "Failed to register the exit function for MPI.";
  });
}

MPICommunicationGroupDef::MPICommunicationGroupDef(
  const std::vector<int>& world_ranks, const Stream& stream)
: CommunicationGroupDef(world_ranks, stream) {
  HT_ASSERT(_stream.device().is_cpu())
    << "MPI communication group must be initialized with "
    << "a stream related with CUDA. Got " << _stream << ".";
  MPI_Init_Once();
  // TODO: Currently we cannot get the world size in super's constructor
  // so we need to perform following check here.
  // Shall move it to super's constructor in the future.
  HT_ASSERT(_world_ranks.back() < mpi_world_size)
    << "Invalid ranks " << _world_ranks << " for world size " << mpi_world_size
    << ".";

  {
    MPICallGuard mpi_guard;

    if (_world_ranks.size() == static_cast<size_t>(mpi_world_size)) {
      // communication group for the world
      MPI_CALL(MPI_Comm_dup(MPI_COMM_WORLD, &_comm));
      HT_ASSERT(_comm != MPI_COMM_NULL) << "Failed to duplicate communicator.";
      MPI_CALL(MPI_Comm_rank(_comm, &_rank));
      MPI_CALL(MPI_Comm_size(_comm, &_size));
      HT_ASSERT(_rank == mpi_world_rank && _size == mpi_world_size)
        << "Failed to get the rank and/or size of duplicated MPI_COMM_WORLD. "
        << "(Got rank " << _rank << " and size " << _size << ".)";
    } else {
      // communication group for the provided ranks
      HT_ASSERT(std::find(_world_ranks.begin(), _world_ranks.end(),
                          mpi_world_rank) != _world_ranks.end())
        << "The current rank " << mpi_world_rank
        << " is not included in the group " << _world_ranks << ".";
      MPI_Group world_group, small_group;
      MPI_CALL(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
      MPI_CALL(MPI_Group_incl(world_group, _world_ranks.size(),
                              _world_ranks.data(), &small_group));
      const int max_attempts = 10;
      int num_attemps = 0;
      while (true) {
        auto status =
          MPI_Comm_create_group(MPI_COMM_WORLD, small_group, 0, &_comm);
        if (status == MPI_SUCCESS) {
          HT_LOG_TRACE << "MPI_Comm_create succeeded for " << _world_ranks;
          HT_ASSERT(_comm != MPI_COMM_NULL) << "Failed to create communicator.";
          break;
        } else if (++num_attemps < max_attempts) {
          HT_LOG_WARN << "MPI_Comm_create failed for " << num_attemps
                      << " attempt(s), retrying...";
          continue;
        } else {
          HT_RUNTIME_ERROR
            << "MPI_Comm_create failed for " << num_attemps
            << " attempt(s), which reaches the maximum number of attempts.";
        }
      }
      MPI_CALL(MPI_Group_free(&world_group));
      MPI_CALL(MPI_Group_free(&small_group));
      HT_ASSERT(_comm != MPI_COMM_NULL) << "Failed to create communicator.";
      MPI_CALL(MPI_Comm_rank(_comm, &_rank));
      MPI_CALL(MPI_Comm_size(_comm, &_size));
      HT_ASSERT(static_cast<size_t>(_size) == _world_ranks.size())
        << "Group sizes mismatch: " << _size << " vs. " << _world_ranks.size()
        << ".";
    }
  }

  HT_ASSERT(_rank >= 0 && _rank < _size)
    << "Failed to get rank and/or size. "
    << "(Got rank " << _rank << " and size " << _size << ".)";

  HT_LOG_DEBUG << "Initialized MPI comm group for " << _world_ranks
               << " with stream " << _stream << ".";
}

MPICommunicationGroupDef::~MPICommunicationGroupDef() {
  Sync();
  if (_comm != MPI_COMM_WORLD && _comm != MPI_COMM_NULL)
    MPI_Comm_free(&_comm);
}

void MPICommunicationGroupDef::Broadcast(NDArray& data, int broadcaster) {
  HT_ASSERT_CPU_DEVICE(data);
  void* buf = data->raw_data_ptr();
  auto numel = data->numel();
  auto mpi_dtype = to_MPI_Datatype(data->dtype());
  int root = world_to_group_rank(broadcaster);
  _latest_future = CPUStream(_stream).EnqueueTask(
    [buf, numel, mpi_dtype, root, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Bcast(buf, numel, mpi_dtype, root, _comm));
    },
    "MPI_Broadcast(broadcaster=" + std::to_string(broadcaster) + ")");
  NDArray::MarkUsedBy(data, _stream);
}

void MPICommunicationGroupDef::AllReduce(const NDArray& input, NDArray& output,
                                         ReductionType red_type) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto numel = input->numel();
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  auto mpi_red_op = to_MPI_Op(red_type);
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, numel, mpi_dtype, mpi_red_op, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Allreduce(send_buf == recv_buf ? MPI_IN_PLACE : send_buf,
                             recv_buf, numel, mpi_dtype, mpi_red_op, _comm));
    },
    "MPI_AllReduce(reduction=" + ReductionType2Str(red_type) + ")");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::AllReduceCoalesce(const NDArrayList& inputs,
                                                 NDArrayList& outputs,
                                                 NDArray contiguous_buffers,
                                                 ReductionType red_type) {
  size_t n_bytes = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    HT_ASSERT_CPU_DEVICE(inputs[i]);
    HT_ASSERT_CPU_DEVICE(outputs[i]);
    HT_ASSERT_EXCHANGABLE(inputs[i], outputs[i]);
    n_bytes += inputs[i]->numel();
  }
  HT_ASSERT(contiguous_buffers->numel() >= n_bytes);
  auto mpi_dtype = to_MPI_Datatype(inputs[0]->dtype());
  auto mpi_red_op = to_MPI_Op(red_type);

  if (contiguous_buffers->numel() > 0) {
    void* buffer_ptr = contiguous_buffers->raw_data_ptr();
    _latest_future = CPUStream(_stream).EnqueueTask(
      [inputs, outputs, buffer_ptr, mpi_dtype, mpi_red_op, this]() {
        int offset = 0;
        int num_bytes_per_element = to_num_bytes(inputs[0]->dtype());
        for (size_t i = 0; i < inputs.size(); i++) {
          void* send_buf = inputs[i]->raw_data_ptr();
          int num_bytes = inputs[i]->numel() * num_bytes_per_element;
          std::memcpy(buffer_ptr + offset, send_buf, num_bytes);
          offset += num_bytes;
        }
        MPICallGuard mpi_guard;
        MPI_CALL(MPI_Allreduce(MPI_IN_PLACE, buffer_ptr,
                               int(offset / num_bytes_per_element), mpi_dtype,
                               mpi_red_op, _comm));
        offset = 0;
        for (size_t i = 0; i < outputs.size(); i++) {
          void* recv_buf = outputs[i]->raw_data_ptr();
          int num_bytes = outputs[i]->numel() * num_bytes_per_element;
          std::memcpy(recv_buf, buffer_ptr + offset, num_bytes);
          offset += num_bytes;
        }
      },
      "AllReduceCoalesce(reduction=" + ReductionType2Str(red_type) + ")");
  } else {
    _latest_future = CPUStream(_stream).EnqueueTask(
      [inputs, outputs, mpi_dtype, mpi_red_op, this]() {
        MPICallGuard mpi_guard;
        for (size_t i = 0; i < inputs.size(); i++) {
          void* send_buf = inputs[i]->raw_data_ptr();
          void* recv_buf = outputs[i]->raw_data_ptr();
          auto numel = inputs[i]->numel();
          MPI_CALL(MPI_Allreduce(send_buf == recv_buf ? MPI_IN_PLACE : send_buf,
                                 recv_buf, numel, mpi_dtype, mpi_red_op,
                                 _comm));
        }
      },
      "MPI_AllReduce(reduction=" + ReductionType2Str(red_type) + ")");
  }
  NDArray::MarkUsedBy(inputs, _stream);
  NDArray::MarkUsedBy(outputs, _stream);
}

void MPICommunicationGroupDef::AlltoAll(const NDArray& input, NDArray& output) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto send_numl = input->numel() / _size;
  auto recv_numl = output->numel() / _size;
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, send_numl, recv_numl, mpi_dtype, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Alltoall(send_buf, send_numl, mpi_dtype, recv_buf, recv_numl,
                            mpi_dtype, _comm));
    },
    "MPI_AlltoAll");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::Reduce(const NDArray& input, NDArray& output,
                                      int reducer, ReductionType red_type) {
  HT_ASSERT_CPU_DEVICE(input);
  int root = world_to_group_rank(reducer);
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = nullptr;
  if (_rank == root) {
    HT_ASSERT_CPU_DEVICE(output);
    HT_ASSERT_EXCHANGABLE(input, output);
    recv_buf = output->raw_data_ptr();
    send_buf = send_buf == recv_buf ? MPI_IN_PLACE : send_buf;
  }
  auto numel = input->numel();
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  auto mpi_red_op = to_MPI_Op(red_type);
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, numel, mpi_dtype, mpi_red_op, root, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Reduce(send_buf, recv_buf, numel, mpi_dtype, mpi_red_op,
                          root, _comm));
    },
    "MPI_Reduce(reduction=" + ReductionType2Str(red_type) + ")");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::AllGather(const NDArray& input,
                                         NDArray& output) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT_SAME_DTYPE(input, output);
  size_t input_size = input->numel();
  size_t output_size = output->numel();
  HT_ASSERT(input->shape(0) * _size == output->shape(0) &&
            input_size * _size == output_size)
    << "Invalid shapes for AllGather: "
    << "(send) " << input->shape() << " vs. "
    << "(recv) " << output->shape() << ".";
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, input_size, mpi_dtype, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Allgather(send_buf, input_size, mpi_dtype, recv_buf,
                             input_size, mpi_dtype, _comm));
    },
    "MPI_AllGather");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::ReduceScatter(const NDArray& input,
                                             NDArray& output,
                                             ReductionType red_type) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT_SAME_DTYPE(input, output);
  size_t input_size = input->numel();
  size_t output_size = output->numel();
  HT_ASSERT(input->shape(0) == output->shape(0) * _size &&
            input_size == output_size * _size)
    << "Invalid shapes for ReduceScatter: "
    << "(send) " << input->shape() << " vs. "
    << "(recv) " << output->shape() << ".";
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = output->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  auto mpi_red_op = to_MPI_Op(red_type);
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, output_size, mpi_dtype, mpi_red_op, this]() {
      MPICallGuard mpi_guard;
      int recv_cnts[_size];
      for (int i = 0; i < _size; i++)
        recv_cnts[i] = output_size;
      MPI_CALL(
        MPI_Reduce_scatter(send_buf == recv_buf ? MPI_IN_PLACE : send_buf,
                           recv_buf, recv_cnts, mpi_dtype, mpi_red_op, _comm));
    },
    "MPI_ReduceScatter(reduction=" + ReductionType2Str(red_type) + ")");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::Gather(const NDArray& input, NDArray& output,
                                      int gatherer) {
  HT_ASSERT_CPU_DEVICE(input);
  int root = world_to_group_rank(gatherer);
  bool is_gatherer = root == _rank;
  size_t input_size = input->numel();
  if (is_gatherer) {
    HT_ASSERT_CPU_DEVICE(output);
    HT_ASSERT_SAME_DTYPE(input, output);
    HT_ASSERT(input->shape(0) * _size == output->shape(0) &&
              input_size * _size == output->numel())
      << "Invalid shapes for Gather: "
      << "(send) " << input->shape() << " vs. "
      << "(recv) " << output->shape() << ".";
  }
  void* send_buf = input->raw_data_ptr();
  void* recv_buf = is_gatherer ? output->raw_data_ptr() : nullptr;
  auto mpi_dtype = to_MPI_Datatype(input->dtype());
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, input_size, mpi_dtype, root, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Gather(send_buf, input_size, mpi_dtype, recv_buf, input_size,
                          mpi_dtype, root, _comm));
    },
    "MPI_Gather(gatherer" + std::to_string(gatherer) + ")");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::Scatter(const NDArray& input, NDArray& output,
                                       int scatterer) {
  HT_ASSERT_CPU_DEVICE(output);
  int root = world_to_group_rank(scatterer);
  bool is_scatterer = root == _rank;
  size_t output_size = output->numel();
  if (is_scatterer) {
    HT_ASSERT_CPU_DEVICE(input);
    HT_ASSERT_SAME_DTYPE(input, output);
    HT_ASSERT(input->shape(0) == output->shape(0) * _size &&
              input->numel() == output_size * _size)
      << "Invalid shapes for Scatter: "
      << "(send) " << input->shape() << " vs. "
      << "(recv) " << output->shape() << ".";
  }
  void* send_buf = is_scatterer ? input->raw_data_ptr() : nullptr;
  void* recv_buf = output->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(output->dtype());
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, recv_buf, output_size, mpi_dtype, root, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Scatter(send_buf, output_size, mpi_dtype, recv_buf,
                           output_size, mpi_dtype, root, _comm));
    },
    "MPI_Scatter(scatterer=" + std::to_string(scatterer) + ")");
  NDArray::MarkUsedBy({input, output}, _stream);
}

void MPICommunicationGroupDef::Send(const NDArray& data, int receiver) {
  int dst = world_to_group_rank(receiver);
  HT_ASSERT(dst != _rank) << "Cannot send to self.";
  size_t size = data->numel();
  if (size == 0)
    return;
  void* send_buf = data->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(data->dtype());
  int tag = static_cast<int>(data->dtype()); // simply use type as tag
  _latest_future = CPUStream(_stream).EnqueueTask(
    [send_buf, size, mpi_dtype, dst, tag, this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Send(send_buf, size, mpi_dtype, dst, tag, _comm));
    },
    "MPI_Send(receiver=" + std::to_string(receiver) + ")");
  NDArray::MarkUsedBy(data, _stream);
}

void MPICommunicationGroupDef::Recv(NDArray& data, int sender) {
  int src = world_to_group_rank(sender);
  HT_ASSERT(src != _rank) << "Cannot receive from self.";
  size_t size = data->numel();
  if (size == 0)
    return;
  void* recv_buf = data->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(data->dtype());
  int tag = static_cast<int>(data->dtype()); // simply use type as tag
  _latest_future = CPUStream(_stream).EnqueueTask(
    [recv_buf, size, mpi_dtype, src, tag, this]() {
      MPICallGuard mpi_guard;
      MPI_Status s;
      memset(&s, 0, sizeof(MPI_Status));
      MPI_CALL(MPI_Recv(recv_buf, size, mpi_dtype, src, tag, _comm, &s));
      HT_ASSERT(s.MPI_ERROR == MPI_SUCCESS) << "Failed in MPI_RECV.";
    },
    "MPI_Recv(sender=" + std::to_string(sender) + ")");
  NDArray::MarkUsedBy(data, _stream);
}

CommTask MPICommunicationGroupDef::ISend(const NDArray& data, int receiver) {
  int dst = world_to_group_rank(receiver);
  HT_ASSERT(dst != _rank) << "Cannot send to self.";
  size_t size = data->numel();
  if (size == 0)
    return CommTask();
  void* send_buf = data->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(data->dtype());
  int tag = static_cast<int>(data->dtype()); // simply use type as tag
  auto task = CommTask(
    std::function<void()>([data, send_buf, size, mpi_dtype, dst, tag, this]() {
      MPI_CALL(MPI_Send(send_buf, size, mpi_dtype, dst, tag, _comm));
    }),
    {data});
  return task;
}

CommTask MPICommunicationGroupDef::IRecv(NDArray& data, int sender) {
  int src = world_to_group_rank(sender);
  HT_ASSERT(src != _rank) << "Cannot receive from self.";
  size_t size = data->numel();
  if (size == 0)
    return CommTask();
  void* recv_buf = data->raw_data_ptr();
  auto mpi_dtype = to_MPI_Datatype(data->dtype());
  int tag = static_cast<int>(data->dtype()); // simply use type as tag
  auto task = CommTask(
    std::function<void()>([data, recv_buf, size, mpi_dtype, src, tag, this]() {
      MPI_Status s;
      memset(&s, 0, sizeof(MPI_Status));
      MPI_CALL(MPI_Recv(recv_buf, size, mpi_dtype, src, tag, _comm, &s));
      HT_ASSERT(s.MPI_ERROR == MPI_SUCCESS) << "Failed in MPI_RECV.";
    }),
    {data});
  return task;
}

void MPICommunicationGroupDef::BatchedISendIRecv(
  const std::vector<CommTask>& tasks) {
  _latest_future = CPUStream(_stream).EnqueueTask(
    [tasks, this]() {
      MPICallGuard mpi_guard;
      for (auto& task : tasks) {
        task.fn();
      }
    },
    "MPI_BatchedISendIRecv");
  for (auto& task : tasks) {
    NDArray::MarkUsedBy(task.data, _stream);
  }
}

void MPICommunicationGroupDef::Barrier(bool sync) {
  _latest_future = CPUStream(_stream).EnqueueTask(
    [this]() {
      MPICallGuard mpi_guard;
      MPI_CALL(MPI_Barrier(_comm));
    },
    "MPI Barrier Wait");
  if (sync)
    Sync();
}

void MPICommunicationGroupDef::Sync() {
  if (_latest_future.valid())
    _latest_future.wait();
}

void MPICommunicationGroupDef::WaitRequest(MPI_Request request) {
  MPICallGuard mpi_guard;
  MPI_Status s;
  memset(&s, 0, sizeof(MPI_Status));
  MPI_CALL(MPI_Wait(&request, &s));
  HT_ASSERT(s.MPI_ERROR == MPI_SUCCESS)
    << "Failed to wait for the MPI request.";
}

MPICommunicationGroup&
MPICommunicationGroup::GetOrCreate(const std::vector<int>& world_ranks,
                                   const Stream& stream) {
  HT_ASSERT(stream.device().is_cpu())
    << "The argument \"stream\" for "
    << "MPICommunicationGroup::GetOrCreate "
    << "must be a CPU stream. Got " << stream << ".";
  // Note: stream id could be -1, we shall shift it by one when accessing
  int stream_id = static_cast<int>(stream.stream_index());

  MPI_Init_Once();

  HT_ASSERT(world_ranks.empty() ||
            CommunicationGroupDef::IsRanksValid(world_ranks))
    << "Invalid world ranks: " << world_ranks;
  if (world_ranks.empty() ||
      static_cast<int>(world_ranks.size()) == mpi_world_size) {
    if (!worldwide_mpi_comm_groups[stream_id + 1].is_defined()) {
      std::unique_lock<std::mutex> lock(mpi_create_group_mutex);
      // double check for thread-safety
      if (!worldwide_mpi_comm_groups[stream_id + 1].is_defined()) {
        std::vector<int> all_world_ranks(mpi_world_size);
        std::iota(all_world_ranks.begin(), all_world_ranks.end(), 0);
        worldwide_mpi_comm_groups[stream_id + 1] =
          MPICommunicationGroup(all_world_ranks, stream);
        mpi_comm_groups[stream_id + 1].insert(
          {all_world_ranks, worldwide_mpi_comm_groups[stream_id + 1]});
      }
    }
    return worldwide_mpi_comm_groups[stream_id + 1];
  } else {
    HT_ASSERT(GetGroupRank(world_ranks) != -1)
      << "Cannot get comm group " << world_ranks << " on rank "
      << GetWorldRank() << ".";
    auto it = mpi_comm_groups[stream_id + 1].find(world_ranks);
    if (it == mpi_comm_groups[stream_id + 1].end()) {
      std::unique_lock<std::mutex> lock(mpi_create_group_mutex);
      // double check for thread-safety
      it = mpi_comm_groups[stream_id + 1].find(world_ranks);
      if (it == mpi_comm_groups[stream_id + 1].end()) {
        MPICommunicationGroup comm_group(world_ranks, stream);
        auto insertion = mpi_comm_groups[stream_id + 1].insert(
          {comm_group->world_ranks(), comm_group});
        HT_ASSERT(insertion.second)
          << "Failed to insert MPICommunicationGroup for ranks "
          << comm_group->world_ranks() << ".";
        it = insertion.first;
      }
    }
    return it->second;
  }
}

MPICommunicationGroup&
MPICommunicationGroup::GetOrCreateWorldwide(const Stream& stream) {
  HT_ASSERT(stream.device().is_cpu())
    << "The argument \"stream\" for "
    << "MPICommunicationGroup::GetOrCreateWorldwide "
    << "must be a CPU stream. Got " << stream << ".";
  // Note: stream id could be -1, we shall shift it by one when accessing
  int stream_id = static_cast<int>(stream.stream_index());

  if (worldwide_mpi_comm_groups[stream_id + 1].is_defined())
    return worldwide_mpi_comm_groups[stream_id + 1];
  else
    return GetOrCreate({});
}

int GetWorldRank() {
  MPI_Init_Once();
  return mpi_world_rank;
}

int GetWorldSize() {
  MPI_Init_Once();
  return mpi_world_size;
}

int GetGroupRank(const std::vector<int>& world_ranks) {
  int my_world_rank = GetWorldRank();
  auto it = std::find(world_ranks.begin(), world_ranks.end(), my_world_rank);
  return it != world_ranks.end() ? std::distance(world_ranks.begin(), it) : -1;
}

/******************************************************
 * Maintaining the mapping between ranks and devices
 ******************************************************/

namespace {
static std::once_flag device_mapping_init_flag;
std::unordered_map<Device, int> device_to_rank_mapping;
std::vector<Device> rank_to_device_mapping;
DeviceGroup global_device_group;

std::vector<std::string> AllGatherHostnames(MPICommunicationGroup& comm) {
  // Walkaround: communication groups handle ndarrays only
  NDArray local_arr = NDArray::empty({1, 1 + HT_MAX_HOSTNAME_LENGTH}, kCPU,
                                     kUInt8, kBlockingStream);
  NDArray gathered_arr = NDArray::empty(
    {comm->size(), 1 + HT_MAX_HOSTNAME_LENGTH}, kCPU, kUInt8, kBlockingStream);
  auto* local_ptr = local_arr->data_ptr<uint8_t>();
  std::string local_hostname = Device::GetLocalHostname();
  local_ptr[0] = static_cast<uint8_t>(local_hostname.length());
  if (!local_hostname.empty())
    local_hostname.copy(reinterpret_cast<char*>(local_ptr + 1),
                        local_hostname.length());
  comm->AllGather(local_arr, gathered_arr);
  comm->Sync();

  std::vector<std::string> hostnames;
  hostnames.reserve(comm->size());
  auto rank_arrs = NDArray::split(gathered_arr, comm->size(), 0);
  for (int rank = 0; rank < comm->size(); rank++) {
    auto* rank_ptr = rank_arrs[rank]->data_ptr<uint8_t>();
    size_t len = static_cast<size_t>(rank_ptr[0]);
    hostnames.emplace_back(reinterpret_cast<char*>(rank_ptr + 1), len);
  }
  return hostnames;
}

void SetUpDeviceMappingWithAssignedLocalDevice(const Device& local_device) {
  HT_ASSERT(local_device.local()) << "Device is not local: " << local_device;
  auto& comm = MPICommunicationGroup::GetOrCreateWorldwide();
  auto hostnames = AllGatherHostnames(comm);
  // Walkaround: communication groups handle ndarrays only
  auto world_size = GetWorldSize();
  NDArray local_arr = NDArray::empty({1, 3}, kCPU, kInt8, kBlockingStream);
  NDArray gathered_arr =
    NDArray::empty({comm->size(), 3}, kCPU, kInt8, kBlockingStream);
  auto* local_ptr = local_arr->data_ptr<int8_t>();
  local_ptr[0] = static_cast<int8_t>(local_device.type());
  local_ptr[1] = static_cast<int8_t>(local_device.index());
  local_ptr[2] = static_cast<int8_t>(local_device.multiplex());
  comm->AllGather(local_arr, gathered_arr);
  comm->Sync();

  device_to_rank_mapping.reserve(world_size);
  rank_to_device_mapping.reserve(world_size);
  auto rank_arrs = NDArray::split(gathered_arr, world_size, 0);
  for (int rank = 0; rank < world_size; rank++) {
    auto* rank_ptr = rank_arrs[rank]->data_ptr<int8_t>();
    Device rank_device(static_cast<DeviceType>(rank_ptr[0]),
                       static_cast<DeviceIndex>(rank_ptr[1]), hostnames[rank],
                       static_cast<uint8_t>(rank_ptr[2]));
    HT_LOG_DEBUG << "Device of rank[" << rank << "]: " << rank_device;
    HT_ASSERT(device_to_rank_mapping.find(rank_device) ==
                device_to_rank_mapping.end() ||
              rank_device.is_cpu())
      << "Device " << rank_device << " is duplicated.";
    device_to_rank_mapping.insert({rank_device, rank});
    rank_to_device_mapping.emplace_back(rank_device);
  }
  global_device_group = DeviceGroup(rank_to_device_mapping);
  HT_LOG_DEBUG << "Global devices: " << global_device_group;
}

void SetUpDeviceMappingAndAssignLocalDevice(
  const std::map<DeviceType, int>& resources) {
  auto& comm = MPICommunicationGroup::GetOrCreateWorldwide();
  auto hostnames = AllGatherHostnames(comm);
  auto local_hostname = Device::GetLocalHostname();
  HT_ASSERT(hostnames[comm->rank()] == local_hostname)
    << "Local hostname mismatched after gathering: " << hostnames[comm->rank()]
    << " vs. " << local_hostname;
  int local_rank = 0;
  for (int i = 0; i < comm->rank(); i++)
    if (hostnames[i] == local_hostname)
      local_rank++;
  HT_LOG_DEBUG << "local host = " << local_hostname << ", rank = " << comm->rank() 
               << ", all hosts = " << hostnames << ", world ranks = " << comm->world_ranks()
               << ", world size = " << GetWorldSize() << ", local rank = " << local_rank;
  Device local_device;
  if (resources.find(kCUDA) == resources.end() || resources.at(kCUDA) == 0) {
    // Question: do we need to set the multiplex field for CPU?
    local_device = Device(kCPU);
  } else {
    auto device_id = local_rank % resources.at(kCUDA);
    auto multiplex = local_rank / resources.at(kCUDA);
    local_device = Device(kCUDA, device_id, local_hostname, multiplex);
  }
  SetUpDeviceMappingWithAssignedLocalDevice(local_device);
}

} // namespace

void SetUpDeviceMappingWithAssignedLocalDeviceOnce(const Device& local_device) {
  if (!device_to_rank_mapping.empty()) {
    HT_LOG_WARN << "Device mapping has been set up.";
    return;
  }
  std::call_once(device_mapping_init_flag,
                 SetUpDeviceMappingWithAssignedLocalDevice, local_device);
}

Device SetUpDeviceMappingAndAssignLocalDeviceOnce(
  const std::map<DeviceType, int>& resources) {
  if (!device_to_rank_mapping.empty()) {
    HT_LOG_WARN << "Device mapping has been set up.";
    return rank_to_device_mapping[GetWorldRank()];
  }
  std::call_once(device_mapping_init_flag,
                 SetUpDeviceMappingAndAssignLocalDevice, resources);
  return rank_to_device_mapping[GetWorldRank()];
}

bool IsGlobalDeviceGroupReady() {
  return !global_device_group.empty();
}

const DeviceGroup& GetGlobalDeviceGroup() {
  HT_ASSERT(!device_to_rank_mapping.empty())
    << "Please set up the device mapping in advance.";
  return global_device_group;
}

const Device& GetLocalDevice() {
  if (rank_to_device_mapping.empty())
    return Device();
  HT_ASSERT(!rank_to_device_mapping.empty())
    << "Please set up the device mapping in advance.";
  return rank_to_device_mapping.at(GetWorldRank());
}

int GetRankOfLocalHost() {
  int world_rank = GetWorldRank(), local_rank = 0;
  for (int i = 0; i < world_rank; i++)
    if (rank_to_device_mapping[i].local())
      local_rank++;
  return local_rank;
}

int DeviceToWorldRank(const Device& device) {
  auto it = device_to_rank_mapping.find(device);
  HT_ASSERT(it != device_to_rank_mapping.end())
    << "Cannot find device " << device << ".";
  return it->second;
}

std::vector<int> DeviceGroupToWorldRanks(const DeviceGroup& device_group) {
  std::vector<int> ranks;
  ranks.reserve(device_group.num_devices());
  for (const auto& device : device_group.devices())
    ranks.push_back(DeviceToWorldRank(device));
  std::sort(ranks.begin(), ranks.end());
  return ranks;
}

} // namespace comm
} // namespace impl
} // namespace hetu
