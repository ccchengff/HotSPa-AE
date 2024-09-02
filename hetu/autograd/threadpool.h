#pragma once

#include "math.h"
#include <vector>
#include <deque>
#include <thread>
#include <functional>
#include <condition_variable>

namespace hetu {
namespace autograd {

class ThreadPool {
  using Task = std::function<void(int, int, int, int)>;

  struct Taskinfo {
    Task task;
    int cur_index;
    int next_index;
    int min_key;
    int max_key;
    Taskinfo() {}
    Taskinfo(Task task_, int cur_index_, int next_index_, int min_key_,
             int max_key_) {
      task = task_;
      cur_index = cur_index_;
      next_index = next_index_;
      min_key = min_key_;
      max_key = max_key_;
    }
  };

  using TaskList = std::deque<Taskinfo>;

  using WorkThreadQueue = std::vector<std::thread*>;

 public:
  ThreadPool();

  ~ThreadPool();

  bool Start(uint ThreadNum = 1);

  void Stop();

  void AddTask(const Task&, int cur_index, int next_index, int min_key,
               int max_key);

  inline uint thread_num() const {
    return thread_num_;
  }

  inline bool is_started() const {
    return is_started_;
  }

 private:
  void ThreadLoop();

  Taskinfo AcceptTask();

  uint thread_num_;

  bool is_started_;

  WorkThreadQueue work_thread_list_;

  TaskList task_list_;

  std::mutex thread_pool_mutex_;

  std::condition_variable condition_variable_;
};

} // namespace autograd
} // namespace hetu
