#pragma once

#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <nccl.h>

namespace hetu {
namespace impl {
namespace comm {

void EmptyNCCLCache();

class NCCLCommunicationGroupDef;
class NCCLCommunicationGroup;

class NCCLCommunicationGroupDef : public CommunicationGroupDef {
 protected:
  friend class NCCLCommunicationGroup;
  struct constrcutor_access_key {};
  NCCLCommunicationGroupDef(const std::vector<int>& world_ranks,
                            const Stream& stream);

 public:
  NCCLCommunicationGroupDef(const constrcutor_access_key&,
                            const std::vector<int>& world_ranks,
                            const Stream& stream)
  : NCCLCommunicationGroupDef(world_ranks, stream) {}

  ~NCCLCommunicationGroupDef();

  const ncclComm_t& GetComm() const {
    return _comm;
  }

  void Broadcast(NDArray& data, int broadcaster) override;

  void AllReduce(const NDArray& input, NDArray& output,
                 ReductionType red_type = kSUM) override;

  void AllReduceCoalesce(const NDArrayList& inputs, NDArrayList& outputs,
                         NDArray contiguous_buffers,
                         ReductionType red_type = kSUM) override;

  void AlltoAll(const NDArray& input, NDArray& output) override;                 

  void Reduce(const NDArray& input, NDArray& output, int reducer,
              ReductionType red_type = kSUM) override;

  void AllGather(const NDArray& input, NDArray& output) override;

  void ReduceScatter(const NDArray& input, NDArray& output,
                     ReductionType red_type = kSUM) override;

  void Gather(const NDArray& input, NDArray& output, int gatherer) override;

  void Scatter(const NDArray& input, NDArray& output, int scatterer) override;

  void Send(const NDArray& data, int receiver) override;

  void Recv(NDArray& data, int sender) override;

  CommTask ISend(const NDArray& data, int receiver) override;

  CommTask IRecv(NDArray& data, int sender) override;

  void BatchedISendIRecv(const std::vector<CommTask>& tasks) override;  

  void Barrier(bool sync = false) override;

  void Sync() override;

  std::string backend() const override {
    return "NCCL";
  }

 protected:
  static void CreateNCCLUniqueId(const std::vector<int>& world_ranks,
                                 ncclUniqueId& id);

  ncclUniqueId _unique_id;
  ncclComm_t _comm;
  NDArray _barrier_arr;
};

class NCCLCommunicationGroup final
: public CommGroupWrapper<NCCLCommunicationGroupDef> {
 protected:
  NCCLCommunicationGroup(const std::vector<int>& world_ranks,
                         const Stream& stream)
  : CommGroupWrapper<NCCLCommunicationGroupDef>(
      make_ptr<NCCLCommunicationGroupDef>(
        NCCLCommunicationGroupDef::constrcutor_access_key(), world_ranks,
        stream)) {}

 public:
  NCCLCommunicationGroup() = default;

  static NCCLCommunicationGroup&
  GetOrCreate(const std::vector<int>& world_ranks, const Stream& stream);

  static NCCLCommunicationGroup&
  GetOrCreate(const std::vector<int>& world_ranks, Device device) {
    return GetOrCreate(
      world_ranks,
      Stream(device, world_ranks.size() != 2 ? kCollectiveStream : kP2PStream));
  }

  static NCCLCommunicationGroup& GetOrCreateWorldwide(const Stream& stream);

  static NCCLCommunicationGroup& GetOrCreateWorldwide(Device device) {
    return GetOrCreateWorldwide(Stream(device, kCollectiveStream));
  }
};

} // namespace comm
} // namespace impl
} // namespace hetu
