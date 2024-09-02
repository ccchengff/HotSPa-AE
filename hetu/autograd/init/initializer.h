#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"

namespace hetu {
namespace autograd {

class Initializer {
 public:
  Initializer() {}

  virtual void Init(NDArray& data, uint64_t seed = 0,
                    StreamIndex stream_id = NDArray::DEFAULT_STREAM) const = 0;

  virtual Initializer* copy() const = 0;

};

class VoidifiedInitializer : public Initializer {
 public:
  VoidifiedInitializer() : Initializer() {}

  void Init(NDArray& data, uint64_t seed = 0,
            StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override {
    // suppress un-used warning
    (void) data;
    (void) seed;
    (void) stream_id;
  }

  Initializer* copy() const override {
    return new VoidifiedInitializer();
  }
};

class ConstantInitializer : public Initializer {
 public:
  ConstantInitializer(double value) : Initializer(), _value(value) {}

  virtual void
  Init(NDArray& data, uint64_t seed = 0,
       StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override {
    (void) seed; // suppress un-used warning
    NDArray::full_(data, value(), stream_id);
  }

  virtual Initializer* copy() const {
    return new ConstantInitializer(value());
  }

  double value() const noexcept {
    return _value;
  }

 protected:
  const double _value;
};

class ZerosInitializer : public ConstantInitializer {
 public:
  ZerosInitializer() : ConstantInitializer(0) {}
};

class OnesInitializer : public ConstantInitializer {
 public:
  OnesInitializer() : ConstantInitializer(1) {}
};

class UniformInitializer : public Initializer {
 public:
  UniformInitializer(double lb = 0, double ub = 1)
  : Initializer(), _lb(lb), _ub(ub) {}

  virtual void
  Init(NDArray& data, uint64_t seed = 0,
       StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override {
    NDArray::uniform_(data, lb(), ub(), seed, stream_id);
  }

  virtual Initializer* copy() const {
    return new UniformInitializer(lb(), ub());
  }

  double lb() const noexcept {
    return _lb;
  }

  double ub() const noexcept {
    return _ub;
  }

 protected:
  const double _lb;
  const double _ub;
};

class NormalInitializer : public Initializer {
 public:
  NormalInitializer(double mean = 0, double stddev = 1)
  : Initializer(), _mean(mean), _stddev(stddev) {}

  virtual void
  Init(NDArray& data, uint64_t seed = 0,
       StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override {
    NDArray::normal_(data, mean(), stddev(), seed, stream_id);
  }

  virtual Initializer* copy() const {
    return new NormalInitializer(mean(), stddev());
  }

  double mean() const noexcept {
    return _mean;
  }

  double stddev() const noexcept {
    return _stddev;
  }

 protected:
  const double _mean;
  const double _stddev;
};

class TruncatedNormalInitializer : public Initializer {
 public:
  TruncatedNormalInitializer(double mean = 0, double stddev = 1, double lb = -2,
                             double ub = 2)
  : Initializer(), _mean(mean), _stddev(stddev), _lb(lb), _ub(ub) {}

  virtual void
  Init(NDArray& data, uint64_t seed = 0,
       StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override {
    NDArray::truncated_normal_(data, mean(), stddev(), lb(), ub(), seed,
                               stream_id);
  }

  virtual Initializer* copy() const {
    return new TruncatedNormalInitializer(mean(), stddev(), lb(), ub());
  }

  double mean() const noexcept {
    return _mean;
  }

  double stddev() const noexcept {
    return _stddev;
  }

  double lb() const noexcept {
    return _lb;
  }

  double ub() const noexcept {
    return _ub;
  }

 protected:
  const double _mean;
  const double _stddev;
  const double _lb;
  const double _ub;
};

class GeneralizedXavierInitializer : public Initializer {
 protected:
  GeneralizedXavierInitializer(const std::string& dist, const std::string& mode,
                               double gain)
  : Initializer(), _dist(dist), _mode(mode), _gain(gain) {
    HT_ASSERT(_dist == "uniform" || _dist == "normal")
      << "Invalid dist: " << _dist;
    HT_ASSERT(_mode == "fan_in" || _mode == "fan_out" || _mode == "avg")
      << "Invalid mode: " << _mode;
    HT_ASSERT(_gain > 0 && _gain != std::numeric_limits<double>::infinity())
      << "Invalid gain: " << _gain;
  }

 public:
  virtual void
  Init(NDArray& data, uint64_t seed = 0,
       StreamIndex stream_id = NDArray::DEFAULT_STREAM) const override;

  virtual Initializer* copy() const {
    return new GeneralizedXavierInitializer(dist(), mode(), gain());
  }

  const std::string& dist() const noexcept {
    return _dist;
  }

  const std::string& mode() const noexcept {
    return _mode;
  }

  double gain() const noexcept {
    return _gain;
  }

 protected:
  const std::string _dist;
  const std::string _mode;
  const double _gain;
};

class XavierUniformInitializer : public GeneralizedXavierInitializer {
 public:
  XavierUniformInitializer(double gain = 3.0)
  : GeneralizedXavierInitializer("uniform", "avg", gain) {}
};

class XavierNormalInitializer : public GeneralizedXavierInitializer {
 public:
  XavierNormalInitializer(double gain = 1.0)
  : GeneralizedXavierInitializer("normal", "avg", gain) {}
};

class HeUniformInitializer : public GeneralizedXavierInitializer {
 public:
  HeUniformInitializer(double gain = 6.0)
  : GeneralizedXavierInitializer("uniform", "fan_in", gain) {}
};

class HeNormalInitializer : public GeneralizedXavierInitializer {
 public:
  HeNormalInitializer(double gain = 2.0)
  : GeneralizedXavierInitializer("normal", "fan_in", gain) {}
};

class LecunUniformInitializer : public GeneralizedXavierInitializer {
 public:
  LecunUniformInitializer(double gain = 3.0)
  : GeneralizedXavierInitializer("uniform", "fan_in", gain) {}
};

class LecunNormalInitializer : public GeneralizedXavierInitializer {
 public:
  LecunNormalInitializer(double gain = 1.0)
  : GeneralizedXavierInitializer("normal", "fan_in", gain) {}
};

} // namespace autograd
} // namespace hetu
