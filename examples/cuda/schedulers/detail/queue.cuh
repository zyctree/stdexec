/*
 * Copyright (c) NVIDIA
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <execution.hpp>
#include <type_traits>

#include <cuda/atomic>

namespace example::cuda::stream::detail {

namespace queue {
  struct task_base_t {
    using fn_t = void(task_base_t*) noexcept;

    task_base_t* next_{};
    fn_t* execute_{};
  };

  struct host_deleter_t {
    template <class T>
    void operator()(T *ptr) {
      cudaFreeHost(ptr);
    }
  };

  template <class T>
  using host_ptr = std::unique_ptr<T, host_deleter_t>;
  using atomic_task_ref = ::cuda::atomic_ref<task_base_t*, ::cuda::thread_scope_system>;

  template <class T, class... As>
  host_ptr<T> make_host(As&&... as) {
    T* ptr{};
    cudaMallocHost(&ptr, sizeof(T));
    new (ptr) T((As&&)as...);
    return host_ptr<T>(ptr);
  }

  struct producer_t {
    task_base_t** tail_;

    void operator()(task_base_t* task) {
      atomic_task_ref tail_ref(*tail_);
      task_base_t* old_tail = tail_ref.load(::cuda::memory_order_acquire);

      while (true) {
        atomic_task_ref next_ref(old_tail->next_);

        task_base_t* expected = nullptr;
        if (next_ref.compare_exchange_weak(
              expected, task, ::cuda::memory_order_release, ::cuda::memory_order_relaxed)) {
          break;
        }

        old_tail = tail_ref.load(::cuda::memory_order_acquire);
      }

      tail_ref.compare_exchange_strong(
          old_tail, task, ::cuda::memory_order_release, ::cuda::memory_order_relaxed);
    }
  };

  struct root_task_t : task_base_t {
    root_task_t() {
      this->execute_ = [](task_base_t* t) noexcept {};
      this->next_ = nullptr;
    }
  };

  struct poller_t {
    task_base_t *head_;
    task_base_t *sentinel_;
    producer_t producer_;
    std::thread poller_;

    poller_t(task_base_t* head, producer_t producer)
      : head_(head)
      , sentinel_(head)
      , producer_(producer) {
      poller_ = std::thread([this] {
        task_base_t* current = head_;

        while (true) {
          atomic_task_ref next_ref(current->next_);

          while(next_ref.load(::cuda::memory_order_relaxed) == nullptr) {
            std::this_thread::yield();
          }
          current = next_ref.load(::cuda::memory_order_acquire);

          if (current == sentinel_) {
            return;
          }

          current->execute_(current);
        }
      });
    }

    ~poller_t() {
      producer_(sentinel_);
      poller_.join();
    }
  };

  struct task_hub_t {
    host_ptr<root_task_t> head_;
    host_ptr<task_base_t*> tail_ptr_;
    poller_t poller_;

    task_hub_t()
      : head_(make_host<root_task_t>())
      , tail_ptr_(make_host<task_base_t*>(head_.get()))
      , poller_(head_.get(), producer()) {
    }

    producer_t producer() {
      return producer_t{tail_ptr_.get()};
    }
  };
}

}

