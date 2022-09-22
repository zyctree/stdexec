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

#include <iostream>

#include "queue.cuh"

namespace example::cuda {

enum class device_type {
  host,
  device
};

#if defined(__clang__) && defined(__CUDA__)
__host__ inline device_type get_device_type() { return device_type::host; }
__device__ inline device_type get_device_type() { return device_type::device; }
#else 
__host__ __device__ inline device_type get_device_type() {
  NV_IF_TARGET(NV_IS_HOST,
               (return device_type::host;),
               (return device_type::device;));
}
#endif

inline __host__ __device__ bool is_on_gpu() {
  return get_device_type() == device_type::device;
}

}

namespace example::cuda::stream {

  struct context_t;
  struct scheduler_t;
  struct sender_base_t {};
  struct receiver_base_t {};

  template <class S>
    concept stream_sender = 
      std::execution::sender<S> &&
      std::is_base_of_v<sender_base_t, std::decay_t<S>>;

  template <class R>
    concept stream_receiver = 
      std::execution::receiver<R> &&
      std::is_base_of_v<receiver_base_t, std::decay_t<R>>;

  namespace detail {
    struct op_state_base_t{};

    template <class EnvId>
      class enqueue_receiver_t : receiver_base_t {
        using Env = std::__t<EnvId>;

        Env env_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

      public:
        template <std::__one_of<std::execution::set_value_t> Tag, class... As>
        friend void tag_invoke(Tag tag, enqueue_receiver_t&& self, As&&... as) noexcept {
          static_assert(sizeof...(As) == 0, "TODO Store data");
          self.producer_(self.task_);
        }

        template <std::__one_of<std::execution::set_error_t, 
                                std::execution::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, enqueue_receiver_t&& self, As&&... as) noexcept {
          self.producer_(self.task_);
        }

        friend Env tag_invoke(std::execution::get_env_t, const enqueue_receiver_t& self) {
          return self.env_;
        }

        enqueue_receiver_t(Env env, queue::task_base_t* task, queue::producer_t producer)
          : env_(env)
          , task_(task)
          , producer_(producer) {}
      };

    template <class Receiver, class Tag, class... As>
      __launch_bounds__(1) __global__ void continuation_kernel(Receiver receiver, Tag tag, As... as) {
        tag(std::move(receiver), (As&&)as...);
      }

    template <stream_receiver Receiver>
      struct continuation_task_t : queue::task_base_t {
        Receiver receiver_;

        continuation_task_t (Receiver receiver)
          : receiver_{receiver} {
          this->execute_ = [](task_base_t* t) noexcept {
            continuation_task_t &self = *reinterpret_cast<continuation_task_t*>(t);

            // TODO Load completion tag
            std::execution::set_value(std::move(self.receiver_));
          };
          this->next_ = nullptr;
        }
      };
  }

  template <class OuterReceiverId>
    struct operation_state_base_t : detail::op_state_base_t {
      using outer_receiver_t = std::__t<OuterReceiverId>;

      bool owner_{false};
      cudaStream_t stream_{0};
      outer_receiver_t receiver_;

      operation_state_base_t(outer_receiver_t receiver)
        : receiver_(receiver) {}

      template <class Tag, class... As _NVCXX_CAPTURE_PACK(As)>
      void propagate_completion_signal(Tag tag, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          if constexpr (stream_receiver<outer_receiver_t>) {
            tag((outer_receiver_t&&)receiver_, (As&&)as...);
          } else {
            detail::continuation_kernel<std::decay_t<outer_receiver_t>, Tag, std::decay_t<As>...><<<1, 1>>>(receiver_, tag, (As&&)as...);
          }
        );
      }

      cudaStream_t allocate() {
        if (stream_ == 0) {
          owner_ = true;
          cudaStreamCreate(&stream_);
        }

        return stream_;
      }

      ~operation_state_base_t() {
        if (owner_) {
          cudaStreamDestroy(stream_);
          stream_ = 0;
        }
      }
    };

  template <class ReceiverId>
    struct propagate_receiver_t : receiver_base_t {
      operation_state_base_t<ReceiverId>& operation_state_;

      template <std::__one_of<std::execution::set_value_t, 
                              std::execution::set_error_t, 
                              std::execution::set_stopped_t> Tag, 
                class... As  _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, propagate_receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.operation_state_.template propagate_completion_signal<Tag, As...>(tag, (As&&)as...);
        );
      }

      friend std::execution::env_of_t<std::__t<ReceiverId>> 
      tag_invoke(std::execution::get_env_t, const propagate_receiver_t& self) {
        return std::execution::get_env(self.operation_state_.receiver_);
      }
    };

  namespace detail {
    template <class SenderId, class InnerReceiverId, class OuterReceiverId>
      struct operation_state_t : operation_state_base_t<OuterReceiverId> {
        using sender_t = std::__t<SenderId>;
        using inner_receiver_t = std::__t<InnerReceiverId>;
        using outer_receiver_t = std::__t<OuterReceiverId>;
        using env_t = std::execution::env_of_t<outer_receiver_t>;
        using intermediate_receiver = std::__t<std::conditional_t<
          stream_sender<sender_t>, 
          std::__x<inner_receiver_t>,
          std::__x<detail::enqueue_receiver_t<std::__x<env_t>>>>>;
        using inner_op_state_t = std::execution::connect_result_t<sender_t, intermediate_receiver>;
        using task_t = detail::continuation_task_t<inner_receiver_t>;

        queue::task_hub_t* hub_;
        queue::host_ptr<task_t> task_;
        inner_op_state_t inner_op_;

        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
          op.stream_ = op.get_stream();
          std::execution::start(op.inner_op_);
        }

        cudaStream_t get_stream() {
          cudaStream_t stream{};

          if constexpr (std::is_base_of_v<detail::op_state_base_t, inner_op_state_t>) {
            stream = inner_op_.get_stream();
          } else {
            stream = this->allocate();
          }

          return stream;
        }

        template <std::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        operation_state_t(sender_t&& sender, queue::task_hub_t*, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , inner_op_{std::execution::connect((sender_t&&)sender, receiver_provider(*this))}
        {}

        template <std::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires (!stream_sender<sender_t>)
        operation_state_t(sender_t&& sender, queue::task_hub_t* hub, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , hub_(hub)
          , task_(queue::make_host<task_t>(receiver_provider(*this)))
          , inner_op_{
              std::execution::connect((sender_t&&)sender, 
              detail::enqueue_receiver_t<std::__x<env_t>>{
                std::execution::get_env(out_receiver), task_.get(), hub_->producer()})}
        {}
      };
  }

  template <class S>
    concept stream_completing_sender = 
      std::execution::sender<S> &&
      std::is_same_v<
          std::tag_invoke_result_t<
            std::execution::get_completion_scheduler_t<std::execution::set_value_t>, S>,
          scheduler_t>;

  template <class Sender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = detail::operation_state_t<std::__x<Sender>, 
                                                        std::__x<InnerReceiver>, 
                                                        std::__x<OuterReceiver>>;

  template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>, OuterReceiver> 
    stream_op_state(Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      detail::queue::task_hub_t* hub = std::execution::get_completion_scheduler<std::execution::set_value_t>(sndr).hub_;

      return stream_op_state_t<
        Sender, 
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr, 
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }

  template <class Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>, OuterReceiver> 
    stream_op_state(detail::queue::task_hub_t* hub, Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      return stream_op_state_t<
        Sender, 
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<std::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr, 
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }
}

