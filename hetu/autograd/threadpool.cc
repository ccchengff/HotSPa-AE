#include "hetu/autograd/threadpool.h"
#include "math.h"
#include <iostream>

namespace hetu {
namespace autograd {

ThreadPool::ThreadPool() : thread_num_(1), is_started_(false) {}

ThreadPool::~ThreadPool() {
  if (true == is_started_) {
    Stop();
  }
}

bool ThreadPool::Start(uint ThreadNum) {
  thread_num_ = ThreadNum;
  if (false == work_thread_list_.empty()) {
    return false;
  }
  is_started_ = true;
  work_thread_list_.reserve(thread_num_);
  for (uint i = 0; i < thread_num_; ++i) {
    work_thread_list_.push_back(
      new std::thread(std::bind(&ThreadPool::ThreadLoop, this)));
  }
  return true;
}

void ThreadPool::Stop() {
  // std::lock_guard<std::mutex> Lock(thread_pool_mutex_);
  is_started_ = false;
  condition_variable_.notify_all();
  int idx = 0;
  for (WorkThreadQueue::iterator it = work_thread_list_.begin();
       it != work_thread_list_.end(); ++it, ++idx) {
    (*it)->join();
    delete *it;
  }
  work_thread_list_.clear();
}

void ThreadPool::ThreadLoop() {
  while (true == is_started_) {
    Taskinfo NewTask = AcceptTask();
    if (NewTask.task) {
      NewTask.task(NewTask.cur_index, NewTask.next_index, NewTask.min_key,
                   NewTask.max_key);
    }
  }
}

void ThreadPool::AddTask(const Task& NewTask, int cur_index, int next_index,
                         int min_key, int max_key) {
  std::lock_guard<std::mutex> Lock(thread_pool_mutex_);
  Taskinfo taskinfo(NewTask, cur_index, next_index, min_key, max_key);
  task_list_.push_back(taskinfo);
  condition_variable_.notify_one();
}

ThreadPool::Taskinfo ThreadPool::AcceptTask() {
  std::unique_lock<std::mutex> Lock(thread_pool_mutex_);
  while (task_list_.empty() && is_started_) {
    condition_variable_.wait(Lock);
  }
  Taskinfo NewTask;
  TaskList::size_type size = task_list_.size();
  if (!task_list_.empty() && is_started_) {
    NewTask = task_list_.front();
    task_list_.pop_front();
  }
  return NewTask;
}

} // namespace autograd
} // namespace hetu
